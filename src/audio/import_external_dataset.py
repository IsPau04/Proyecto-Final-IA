import argparse
import csv
import shutil
import unicodedata
from pathlib import Path

from src.config import SIMPLE_COMMANDS


VALID_CLASSES = set(SIMPLE_COMMANDS)
AUGMENTATION_MARKERS = ("ruido_leve", "ruido_fuerte", "aumentado", "noise")
IGNORED_NAME_TOKENS = {
    "audio",
    "comando",
    "cuarto",
    "fuerte",
    "leve",
    "noise",
    "original",
    "ruido",
    "ruido_fondo",
    "ruido_fuerte",
    "ruido_leve",
    "sample",
    "voz",
    "wav",
}

# La ruta se calcula desde este archivo para que funcione al ejecutarlo desde
# cualquier carpeta dentro del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_IMPORTS_DIR = PROJECT_ROOT / "data" / "external_imports" / "extracted"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
AUGMENTED_DATA_DIR = PROJECT_ROOT / "data" / "augmented"
REPORT_PATH = PROJECT_ROOT / "docs" / "import_report.csv"


def normalize_text(value: str) -> str:
    """Normaliza texto para comparar clases aunque tengan tildes o separadores."""
    without_accents = unicodedata.normalize("NFKD", value)
    ascii_text = without_accents.encode("ascii", "ignore").decode("ascii")

    normalized_chars = []
    for char in ascii_text.lower():
        if char.isalnum():
            normalized_chars.append(char)
        else:
            normalized_chars.append("_")

    return "_".join("".join(normalized_chars).split("_"))


CLASS_ALIASES = {normalize_text(command): command for command in VALID_CLASSES}
CLASS_ALIASES.update(
    {
        "abre": "ABRE",
        "baja": "BAJA",
        "cierra": "CIERRA",
        "sube": "SUBE",
        "posicion_inicial": "POSICION_INICIAL",
        "posicioninicial": "POSICION_INICIAL",
        "ruido_fondo": "RUIDO_FONDO",
        "ruidofondo": "RUIDO_FONDO",
    }
)


def split_tokens(value: str) -> list[str]:
    return [token for token in normalize_text(value).split("_") if token]


def detect_class(file_path: Path) -> str | None:
    """Detecta clase desde carpetas cercanas y luego desde el nombre del WAV."""
    candidates = [
        file_path.parent.name,
        *[parent.name for parent in file_path.parents if parent != file_path.parent],
        file_path.stem,
    ]

    for candidate in candidates:
        normalized = normalize_text(candidate)
        if normalized in CLASS_ALIASES:
            return CLASS_ALIASES[normalized]

        tokens = split_tokens(candidate)
        for token in tokens:
            if token in CLASS_ALIASES:
                return CLASS_ALIASES[token]

        for index in range(len(tokens) - 1):
            joined_pair = "_".join(tokens[index : index + 2])
            if joined_pair in CLASS_ALIASES:
                return CLASS_ALIASES[joined_pair]

    return None


def detect_augmentation_suffix(file_path: Path) -> str | None:
    normalized_name = normalize_text(file_path.stem)

    for marker in AUGMENTATION_MARKERS:
        if marker in normalized_name:
            return marker

    return None


def normalize_speaker(value: str) -> str:
    tokens = split_tokens(value)
    speaker_tokens = []
    class_tokens = {token for command in VALID_CLASSES for token in split_tokens(command)}

    for token in tokens:
        if token in class_tokens or token in IGNORED_NAME_TOKENS:
            continue
        if token.isdigit():
            continue
        speaker_tokens.append(token)

    return "_".join(speaker_tokens[:2]) or "hablante"


def detect_speaker(file_path: Path, command_class: str) -> str:
    parent_speaker = normalize_speaker(file_path.parent.name)
    if parent_speaker != "hablante" and normalize_text(file_path.parent.name) != normalize_text(command_class):
        return parent_speaker

    return normalize_speaker(file_path.stem)


def extract_existing_index(file_path: Path, command_class: str, speaker: str) -> int | None:
    prefix = f"{command_class}_{speaker}_"

    if not file_path.stem.startswith(prefix):
        return None

    remainder = file_path.stem[len(prefix) :]
    index_text = remainder.split("_", maxsplit=1)[0]

    try:
        return int(index_text)
    except ValueError:
        return None


def next_sample_index(class_dir: Path, command_class: str, speaker: str) -> int:
    existing_indices = []

    for file_path in class_dir.glob(f"{command_class}_{speaker}_*.wav"):
        existing_index = extract_existing_index(file_path, command_class, speaker)
        if existing_index is not None:
            existing_indices.append(existing_index)

    return max(existing_indices, default=0) + 1


def unique_destination_path(
    class_dir: Path,
    command_class: str,
    speaker: str,
    start_index: int,
    suffix: str | None,
    reserved_paths: set[Path],
) -> tuple[Path, int]:
    sample_index = start_index

    while True:
        suffix_part = f"_{suffix}" if suffix else ""
        filename = f"{command_class}_{speaker}_{sample_index:04d}{suffix_part}.wav"
        destination = class_dir / filename

        if not destination.exists() and destination not in reserved_paths:
            return destination, sample_index

        sample_index += 1


def find_wav_files() -> list[Path]:
    if not EXTERNAL_IMPORTS_DIR.exists():
        return []

    return sorted(EXTERNAL_IMPORTS_DIR.rglob("*.wav"))


def import_external_dataset(apply: bool) -> list[dict[str, str]]:
    report_rows = []
    next_indices: dict[tuple[Path, str, str], int] = {}
    reserved_paths: set[Path] = set()

    for source_path in find_wav_files():
        command_class = detect_class(source_path)

        if command_class not in VALID_CLASSES:
            report_rows.append(
                {
                    "archivo_origen": str(source_path),
                    "clase_detectada": "",
                    "tipo": "",
                    "archivo_destino": "",
                    "estado": "omitido_clase_no_detectada",
                }
            )
            continue

        augmentation_suffix = detect_augmentation_suffix(source_path)
        is_augmented = augmentation_suffix is not None and command_class != "RUIDO_FONDO"
        dataset_dir = AUGMENTED_DATA_DIR if is_augmented else RAW_DATA_DIR
        import_type = "aumentado" if is_augmented else "raw"
        speaker = detect_speaker(source_path, command_class)
        class_dir = dataset_dir / command_class
        index_key = (class_dir, command_class, speaker)

        if index_key not in next_indices:
            next_indices[index_key] = next_sample_index(class_dir, command_class, speaker)

        destination_path, used_index = unique_destination_path(
            class_dir,
            command_class,
            speaker,
            next_indices[index_key],
            augmentation_suffix if is_augmented else None,
            reserved_paths,
        )
        next_indices[index_key] = used_index + 1
        reserved_paths.add(destination_path)

        if apply:
            class_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            status = "copiado"
        else:
            status = "simulado"

        report_rows.append(
            {
                "archivo_origen": str(source_path),
                "clase_detectada": command_class,
                "tipo": import_type,
                "archivo_destino": str(destination_path),
                "estado": status,
            }
        )

    return report_rows


def write_report(rows: list[dict[str, str]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["archivo_origen", "clase_detectada", "tipo", "archivo_destino", "estado"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Importa y normaliza audios externos al dataset raw/augmented."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_true", help="Simula sin copiar y genera el reporte CSV.")
    mode_group.add_argument("--apply", action="store_true", help="Copia archivos y escribe el reporte CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = import_external_dataset(apply=args.apply)
    write_report(rows)

    copied_count = sum(1 for row in rows if row["estado"] == "copiado")
    simulated_count = sum(1 for row in rows if row["estado"] == "simulado")
    skipped_count = sum(1 for row in rows if row["estado"].startswith("omitido"))

    print(f"Archivos encontrados: {len(rows)}")
    print(f"Copiados: {copied_count}")
    print(f"Simulados: {simulated_count}")
    print(f"Omitidos: {skipped_count}")
    print(f"Reporte: {REPORT_PATH}")

    if not args.apply:
        print("Dry-run finalizado. Usa --apply para copiar los archivos.")


if __name__ == "__main__":
    main()

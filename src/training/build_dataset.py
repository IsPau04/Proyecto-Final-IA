"""Construye el dataset numerico para entrenar comandos de voz."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from src.config import SIMPLE_COMMANDS
from src.features.audio_features import (
    extract_log_mel_spectrogram,
    load_audio,
    normalize_audio,
    trim_silence_energy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "commands_dataset.npz"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "dataset_summary.csv"


def parse_args():
    """Define los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Construye un dataset NPZ de espectrogramas log-Mel.",
    )
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Incluye audios desde data/augmented/<CLASE>/ ademas de data/raw/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Ruta de salida del dataset .npz. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def find_wav_files(base_dir, class_name):
    """Devuelve archivos WAV ordenados para una clase."""
    class_dir = base_dir / class_name
    if not class_dir.exists():
        return []

    return sorted(class_dir.glob("*.wav"))


def build_summary(raw_counts, augmented_counts):
    """Crea filas de resumen por clase."""
    rows = []
    for class_name in SIMPLE_COMMANDS:
        raw_count = raw_counts[class_name]
        augmented_count = augmented_counts[class_name]
        rows.append(
            {
                "clase": class_name,
                "cantidad_raw": raw_count,
                "cantidad_augmented": augmented_count,
                "cantidad_total": raw_count + augmented_count,
            }
        )
    return rows


def save_summary_csv(summary_rows, output_path=SUMMARY_PATH):
    """Guarda el reporte CSV del dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "clase",
                "cantidad_raw",
                "cantidad_augmented",
                "cantidad_total",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def pad_or_truncate_spectrogram(spectrogram, target_frames):
    """Ajusta el eje temporal de un espectrograma con padding o truncamiento."""
    current_frames = spectrogram.shape[1]
    if current_frames == target_frames:
        return spectrogram

    if current_frames > target_frames:
        return spectrogram[:, :target_frames]

    pad_width = target_frames - current_frames
    return np.pad(
        spectrogram,
        pad_width=((0, 0), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )


def process_audio_file(file_path):
    """Carga un WAV y devuelve su espectrograma log-Mel."""
    audio = load_audio(file_path)
    audio = normalize_audio(audio)
    audio = trim_silence_energy(audio)
    return extract_log_mel_spectrogram(audio)


def collect_examples(include_augmented):
    """Lee audios y construye listas de espectrogramas, etiquetas y rutas."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    augmented_dir = PROJECT_ROOT / "data" / "augmented"
    class_to_label = {class_name: index for index, class_name in enumerate(SIMPLE_COMMANDS)}

    spectrograms = []
    labels = []
    file_paths = []
    raw_counts = Counter()
    augmented_counts = Counter()

    for class_name in SIMPLE_COMMANDS:
        for file_path in find_wav_files(raw_dir, class_name):
            spectrograms.append(process_audio_file(file_path))
            labels.append(class_to_label[class_name])
            file_paths.append(str(file_path))
            raw_counts[class_name] += 1

        if include_augmented:
            for file_path in find_wav_files(augmented_dir, class_name):
                spectrograms.append(process_audio_file(file_path))
                labels.append(class_to_label[class_name])
                file_paths.append(str(file_path))
                augmented_counts[class_name] += 1

    return spectrograms, labels, file_paths, raw_counts, augmented_counts


def build_dataset(include_augmented):
    """Construye los arreglos X, y, labels y file_paths."""
    spectrograms, y_values, file_paths, raw_counts, augmented_counts = collect_examples(
        include_augmented,
    )

    if not spectrograms:
        raise ValueError("No se encontraron archivos .wav para construir el dataset.")

    target_frames = max(spectrogram.shape[1] for spectrogram in spectrograms)
    fixed_spectrograms = [
        pad_or_truncate_spectrogram(spectrogram, target_frames)
        for spectrogram in spectrograms
    ]

    X = np.asarray(fixed_spectrograms, dtype=np.float32)
    X = X[..., np.newaxis]
    y = np.asarray(y_values, dtype=np.int64)
    labels = np.asarray(SIMPLE_COMMANDS)
    file_paths = np.asarray(file_paths)

    return X, y, labels, file_paths, raw_counts, augmented_counts


def print_dataset_info(X, y, labels, raw_counts, augmented_counts):
    """Muestra informacion basica del dataset construido."""
    total_counts = Counter(raw_counts)
    total_counts.update(augmented_counts)

    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    print(f"Etiquetas: {list(labels)}")
    print("Conteo por clase:")
    for class_name in SIMPLE_COMMANDS:
        print(f"  {class_name}: {total_counts[class_name]}")


def main():
    args = parse_args()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    X, y, labels, file_paths, raw_counts, augmented_counts = build_dataset(
        include_augmented=args.include_augmented,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        labels=labels,
        file_paths=file_paths,
    )

    summary_rows = build_summary(raw_counts, augmented_counts)
    save_summary_csv(summary_rows)
    print_dataset_info(X, y, labels, raw_counts, augmented_counts)
    print(f"Dataset guardado en: {output_path}")
    print(f"Reporte CSV guardado en: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

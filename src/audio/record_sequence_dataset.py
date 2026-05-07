from pathlib import Path

import sounddevice as sd
import soundfile as sf

from src.config import SAMPLE_RATE, SEQUENCE_COMMANDS, SEQUENCE_COMMAND_PHRASES


# Clases compuestas permitidas para mantener el corpus secuencial ordenado.
VALID_CLASSES = set(SEQUENCE_COMMANDS)
MIN_DURATION_SECONDS = 2.0
MAX_DURATION_SECONDS = 4.0

# La ruta se calcula desde este archivo para que el script funcione aunque se
# ejecute desde otra carpeta dentro del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SEQUENCE_DATA_DIR = PROJECT_ROOT / "data" / "raw_sequences"


def normalize_command_phrase(value: str) -> str:
    """Normaliza frases naturales para comparar sin mayusculas ni espacios extra."""
    return " ".join(value.lower().split())


def resolve_command_class(user_input: str) -> str | None:
    """Convierte una etiqueta o frase natural valida a la etiqueta interna."""
    command_class = user_input.strip().upper()

    if command_class in VALID_CLASSES:
        return command_class

    normalized_input = normalize_command_phrase(user_input)
    phrase_to_class = {
        normalize_command_phrase(phrase): command_class
        for command_class, phrase in SEQUENCE_COMMAND_PHRASES.items()
    }

    return phrase_to_class.get(normalized_input)


def ask_command_class() -> str:
    """Pide una clase compuesta o frase natural hasta recibir una permitida."""
    valid_options = ", ".join(sorted(VALID_CLASSES))

    while True:
        user_input = input(f"Clase o frase del comando compuesto ({valid_options}): ")
        command_class = resolve_command_class(user_input)

        if command_class is not None:
            return command_class

        print("Clase o frase no valida. Intenta de nuevo usando una de las opciones.")


def ask_speaker_name() -> str:
    """Pide el nombre del hablante y lo normaliza para usarlo en filenames."""
    while True:
        speaker = input("Nombre del hablante: ").strip().lower().replace(" ", "_")

        if speaker:
            return speaker

        print("El nombre del hablante no puede estar vacio.")


def ask_positive_integer(prompt: str) -> int:
    """Pide un entero positivo para evitar cantidades invalidas de muestras."""
    while True:
        value = input(prompt).strip()

        try:
            number = int(value)
        except ValueError:
            print("Ingresa un numero entero.")
            continue

        if number > 0:
            return number

        print("El numero debe ser mayor que cero.")


def ask_duration_seconds() -> float:
    """Permite elegir una duracion dentro del rango permitido de 2 a 4 s."""
    while True:
        value = input("Duracion por muestra en segundos (2.0 a 4.0): ").strip()

        try:
            duration = float(value)
        except ValueError:
            print("Ingresa una duracion numerica, por ejemplo 3.0.")
            continue

        if MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
            return duration

        print("La duracion debe estar entre 2.0 y 4.0 segundos.")


def next_sample_index(class_dir: Path, command_class: str, speaker: str) -> int:
    """Busca el siguiente indice disponible para no sobrescribir grabaciones."""
    pattern = f"{command_class}_{speaker}_*.wav"
    existing_files = sorted(class_dir.glob(pattern))

    if not existing_files:
        return 1

    used_indices = []
    for file_path in existing_files:
        try:
            used_indices.append(int(file_path.stem.rsplit("_", maxsplit=1)[1]))
        except (IndexError, ValueError):
            continue

    return max(used_indices, default=0) + 1


def record_sample(duration_seconds: float):
    """Graba una muestra mono y espera a que termine antes de guardarla."""
    frames = int(SAMPLE_RATE * duration_seconds)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio


def main() -> None:
    command_class = ask_command_class()
    phrase = SEQUENCE_COMMAND_PHRASES[command_class]

    print(f'Frase a decir: "{phrase}"')

    speaker = ask_speaker_name()
    sample_count = ask_positive_integer("Cantidad de muestras: ")
    duration_seconds = ask_duration_seconds()

    class_dir = RAW_SEQUENCE_DATA_DIR / command_class
    class_dir.mkdir(parents=True, exist_ok=True)

    start_index = next_sample_index(class_dir, command_class, speaker)

    print("\nPresiona Enter para grabar cada muestra.")
    print("Evita hablar antes del inicio de la grabacion.\n")

    for offset in range(sample_count):
        sample_index = start_index + offset
        filename = f"{command_class}_{speaker}_{sample_index:04d}.wav"
        output_path = class_dir / filename

        input(f"Muestra {offset + 1}/{sample_count}: presiona Enter para grabar...")
        print(f'Grabando: "{phrase}"...')

        audio = record_sample(duration_seconds)

        # soundfile escribe el WAV a 16 kHz para que el corpus secuencial quede
        # listo para extraccion posterior de caracteristicas temporales.
        sf.write(output_path, audio, SAMPLE_RATE)
        print(f"Guardado: {output_path}")

    print("\nGrabacion finalizada.")


if __name__ == "__main__":
    main()

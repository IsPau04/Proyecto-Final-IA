"""Detecta una secuencia de comandos base desde audio largo."""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAMPLE_RATE  # noqa: E402
from src.features.audio_features import (  # noqa: E402
    extract_log_mel_spectrogram,
    normalize_audio,
    trim_silence_energy,
)


MODEL_PATH = PROJECT_ROOT / "models" / "command_cnn.keras"
PREPROCESS_PARAMS_PATH = PROJECT_ROOT / "models" / "preprocess_params.npz"
LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"
BACKGROUND_LABEL = "RUIDO_FONDO"
DEFAULT_DURATION = 6.0
DEFAULT_WINDOW_DURATION = 2.0
DEFAULT_HOP_DURATION = 0.75
DEFAULT_THRESHOLD = 0.60


def parse_args():
    """Define argumentos para detectar comandos en ventanas consecutivas."""
    parser = argparse.ArgumentParser(
        description=(
            "Graba o carga audio y detecta comandos base en orden usando "
            "ventanas deslizantes."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Duracion de la grabacion en segundos. Default: {DEFAULT_DURATION}",
    )
    parser.add_argument(
        "--window-duration",
        type=float,
        default=DEFAULT_WINDOW_DURATION,
        help=(
            "Duracion de cada ventana de inferencia en segundos. "
            f"Default: {DEFAULT_WINDOW_DURATION}"
        ),
    )
    parser.add_argument(
        "--hop-duration",
        type=float,
        default=DEFAULT_HOP_DURATION,
        help=(
            "Separacion entre inicios de ventanas en segundos. "
            f"Default: {DEFAULT_HOP_DURATION}"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Confianza minima para aceptar comandos. Default: {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Dispositivo de entrada para sounddevice: indice o nombre parcial.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Lista dispositivos de audio disponibles y termina.",
    )
    parser.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="Archivo WAV para probar una frase grabada sin usar microfono.",
    )
    return parser.parse_args()


def validate_args(args):
    """Valida argumentos numericos antes de ejecutar inferencia."""
    if args.duration <= 0:
        raise ValueError("--duration debe ser mayor que 0.")
    if args.window_duration <= 0:
        raise ValueError("--window-duration debe ser mayor que 0.")
    if args.hop_duration <= 0:
        raise ValueError("--hop-duration debe ser mayor que 0.")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold debe estar entre 0 y 1.")


def parse_device(device):
    """Convierte indices numericos de dispositivo y conserva nombres como texto."""
    if device is None:
        return None

    try:
        return int(device)
    except ValueError:
        return device


def list_devices():
    """Muestra los dispositivos de audio conocidos por sounddevice."""
    print(sd.query_devices())


def load_labels(labels_path):
    """Carga las etiquetas del modelo en el orden usado durante entrenamiento."""
    if not labels_path.exists():
        raise FileNotFoundError(f"No existe el archivo de etiquetas: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as labels_file:
        labels = json.load(labels_file)

    if not isinstance(labels, list) or not labels:
        raise ValueError("labels.json debe contener una lista no vacia de etiquetas.")

    return labels


def load_preprocess_params(params_path):
    """Carga media y desviacion estandar calculadas sobre train."""
    if not params_path.exists():
        raise FileNotFoundError(f"No existe el archivo de normalizacion: {params_path}")

    with np.load(params_path) as params:
        mean = params["mean"].astype(np.float32)
        std = params["std"].astype(np.float32)

    if np.any(std == 0):
        raise ValueError("La desviacion estandar guardada contiene valores en 0.")

    return mean, std


def load_model(model_path):
    """Carga el modelo Keras entrenado."""
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado: {model_path}")

    from tensorflow import keras

    return keras.models.load_model(model_path)


def get_model_input_shape(model):
    """Devuelve la forma de entrada esperada sin el eje de batch."""
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(f"El modelo debe esperar entrada 4D, pero usa {input_shape}.")

    _, n_mels, frames, channels = input_shape
    if n_mels is None or frames is None or channels is None:
        raise ValueError(f"La forma de entrada del modelo debe estar definida: {input_shape}")
    if channels != 1:
        raise ValueError(f"El script espera un canal, pero el modelo usa {channels}.")

    return int(n_mels), int(frames), int(channels)


def validate_model_outputs(model, labels):
    """Comprueba que la salida del modelo coincida con las etiquetas."""
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    if output_shape[-1] != len(labels):
        raise ValueError(
            "La cantidad de etiquetas no coincide con la salida del modelo: "
            f"{len(labels)} etiquetas vs {output_shape[-1]} salidas."
        )


def record_audio(duration, device=None):
    """Graba audio mono desde el microfono usando SAMPLE_RATE."""
    samples = int(round(duration * SAMPLE_RATE))
    print(f"Grabando {duration:.2f} segundos a {SAMPLE_RATE} Hz...")
    recording = sd.rec(
        samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=parse_device(device),
    )
    sd.wait()
    print("Grabacion finalizada.")
    return np.squeeze(recording).astype(np.float32)


def load_wav_audio(audio_file):
    """Carga un WAV mono y lo remuestrea a SAMPLE_RATE si hace falta."""
    if not audio_file.exists():
        raise FileNotFoundError(f"No existe el archivo de audio: {audio_file}")

    audio, sample_rate = sf.read(audio_file, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

    return audio.astype(np.float32)


def pad_or_truncate_axis(array, target_size, axis):
    """Ajusta una dimension con padding en cero o truncamiento."""
    current_size = array.shape[axis]
    if current_size == target_size:
        return array

    if current_size > target_size:
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(0, target_size)
        return array[tuple(slices)]

    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target_size - current_size)
    return np.pad(array, pad_width=pad_width, mode="constant", constant_values=0)


def prepare_audio_for_model(audio, input_shape, mean, std):
    """Replica el preprocesamiento del dataset y normaliza para inferencia."""
    n_mels, frames, _channels = input_shape

    audio = normalize_audio(audio)
    audio = trim_silence_energy(audio)
    spectrogram = extract_log_mel_spectrogram(audio, n_mels=n_mels)
    spectrogram = pad_or_truncate_axis(spectrogram, n_mels, axis=0)
    spectrogram = pad_or_truncate_axis(spectrogram, frames, axis=1)
    spectrogram = spectrogram.astype(np.float32)
    spectrogram = spectrogram[..., np.newaxis]
    spectrogram = (spectrogram - mean) / std

    return spectrogram[np.newaxis, ...]


def iter_windows(audio, window_duration, hop_duration):
    """Genera ventanas deslizantes con sus tiempos de inicio y fin."""
    total_samples = len(audio)
    window_samples = max(1, int(round(window_duration * SAMPLE_RATE)))
    hop_samples = max(1, int(round(hop_duration * SAMPLE_RATE)))

    start_sample = 0
    while start_sample < total_samples:
        end_sample = min(start_sample + window_samples, total_samples)
        yield (
            start_sample / SAMPLE_RATE,
            end_sample / SAMPLE_RATE,
            audio[start_sample:end_sample],
        )
        start_sample += hop_samples


def predict_window(model, labels, audio, input_shape, mean, std):
    """Predice la clase ganadora y su confianza para una ventana."""
    X = prepare_audio_for_model(audio, input_shape, mean, std)
    probabilities = model.predict(X, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = labels[predicted_index]
    confidence = float(probabilities[predicted_index])
    return predicted_label, confidence


def classify_window_state(label, confidence, threshold, last_accepted):
    """Determina si la prediccion de una ventana entra a la secuencia final."""
    if confidence < threshold:
        return "BAJA_CONFIANZA", False
    if label == BACKGROUND_LABEL:
        return "RUIDO_IGNORADO", False
    if label == last_accepted:
        return "REPETIDO_IGNORADO", False

    return "ACEPTADO", True


def run_sequence_inference(model, labels, mean, std, input_shape, audio, args):
    """Ejecuta inferencia por ventanas y construye la secuencia final."""
    rows = []
    sequence = []
    last_accepted = None

    for start_time, end_time, window_audio in iter_windows(
        audio,
        args.window_duration,
        args.hop_duration,
    ):
        label, confidence = predict_window(model, labels, window_audio, input_shape, mean, std)
        state, accepted = classify_window_state(
            label,
            confidence,
            args.threshold,
            last_accepted,
        )

        if accepted:
            sequence.append(label)
            last_accepted = label

        rows.append(
            {
                "start": start_time,
                "end": end_time,
                "label": label,
                "confidence": confidence,
                "state": state,
            }
        )

    return rows, sequence


def print_window_table(rows):
    """Muestra una tabla compacta de predicciones por ventana."""
    headers = ("tiempo_inicio", "tiempo_fin", "clase_predicha", "confianza", "estado")
    print()
    print(
        f"{headers[0]:>13}  {headers[1]:>10}  "
        f"{headers[2]:<18}  {headers[3]:>10}  {headers[4]}"
    )
    print("-" * 72)

    for row in rows:
        print(
            f"{row['start']:>13.2f}  {row['end']:>10.2f}  "
            f"{row['label']:<18}  {row['confidence']:>10.4f}  {row['state']}"
        )


def print_sequence(sequence):
    """Muestra la secuencia final de comandos detectados."""
    print()
    if sequence:
        print("Secuencia final detectada:")
        print(" -> ".join(sequence))
    else:
        print("Secuencia final detectada: (sin comandos)")


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    validate_args(args)

    model = load_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
    mean, std = load_preprocess_params(PREPROCESS_PARAMS_PATH)
    input_shape = get_model_input_shape(model)
    validate_model_outputs(model, labels)

    if args.audio_file is not None:
        audio = load_wav_audio(args.audio_file)
        print(f"Audio cargado: {args.audio_file} ({len(audio) / SAMPLE_RATE:.2f} s)")
    else:
        audio = record_audio(args.duration, device=args.device)

    rows, sequence = run_sequence_inference(model, labels, mean, std, input_shape, audio, args)
    print_window_table(rows)
    print_sequence(sequence)


if __name__ == "__main__":
    main()

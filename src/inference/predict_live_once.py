"""Predice un comando de voz grabado una sola vez desde el microfono."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from tensorflow import keras


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
DEFAULT_THRESHOLD = 0.70


def parse_args():
    """Define argumentos para grabar y clasificar un audio corto."""
    parser = argparse.ArgumentParser(
        description="Graba audio del microfono y predice un comando con el modelo CNN.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duracion de la grabacion en segundos. Default: 2.0",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Confianza minima para aceptar la prediccion. Default: {DEFAULT_THRESHOLD}",
    )
    return parser.parse_args()


def validate_args(args):
    """Valida argumentos numericos antes de iniciar la grabacion."""
    if args.duration <= 0:
        raise ValueError("--duration debe ser mayor que 0.")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold debe estar entre 0 y 1.")


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


def record_audio(duration):
    """Graba audio mono desde el microfono usando SAMPLE_RATE."""
    samples = int(duration * SAMPLE_RATE)
    print(f"Grabando {duration:.2f} segundos a {SAMPLE_RATE} Hz...")
    recording = sd.rec(samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("Grabacion finalizada.")
    return np.squeeze(recording).astype(np.float32)


def pad_or_truncate_axis(array, target_size, axis):
    """Ajusta una dimension de un arreglo con padding en cero o truncamiento."""
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


def print_prediction(labels, probabilities, threshold):
    """Muestra clase ganadora, confianza y top 3 de probabilidades."""
    top_indices = np.argsort(probabilities)[::-1][:3]
    predicted_index = int(top_indices[0])
    predicted_label = labels[predicted_index]
    confidence = float(probabilities[predicted_index])

    print(f"Clase predicha: {predicted_label}")
    print(f"Confianza: {confidence:.4f}")

    if confidence < threshold:
        print(
            f"Prediccion insegura: confianza menor al umbral "
            f"({confidence:.4f} < {threshold:.4f})."
        )

    print("Top 3 probabilidades:")
    for index in top_indices:
        print(f"  {labels[int(index)]}: {float(probabilities[int(index)]):.4f}")


def main():
    args = parse_args()
    validate_args(args)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado: {MODEL_PATH}")

    model = keras.models.load_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
    mean, std = load_preprocess_params(PREPROCESS_PARAMS_PATH)
    input_shape = get_model_input_shape(model)

    if model.output_shape[-1] != len(labels):
        raise ValueError(
            "La cantidad de etiquetas no coincide con la salida del modelo: "
            f"{len(labels)} etiquetas vs {model.output_shape[-1]} salidas."
        )

    audio = record_audio(args.duration)
    X = prepare_audio_for_model(audio, input_shape, mean, std)
    probabilities = model.predict(X, verbose=0)[0]
    print_prediction(labels, probabilities, args.threshold)


if __name__ == "__main__":
    main()

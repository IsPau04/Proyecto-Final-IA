"""Detecta una secuencia de comandos base desde audio largo."""

import argparse
import json
import sys
import time
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
DEFAULT_ESP32_PORT = "COM4"
DEFAULT_COMMAND_DELAY = 0.50
DEFAULT_ENERGY_THRESHOLD = 0.003
DEFAULT_SEGMENTATION_MODE = "vad"
DEFAULT_MIN_SEGMENT_DURATION = 0.35
DEFAULT_MIN_SILENCE_DURATION = 0.35
DEFAULT_SEGMENT_PADDING = 0.20
DEFAULT_MIN_COMMAND_GAP = 0.6
DEFAULT_MIN_CONFIRMATIONS = 2
DEFAULT_CONFIRMATION_WINDOW = 1.0
DEFAULT_HIGH_CONFIDENCE_OVERRIDE = 0.98
VAD_FRAME_DURATION = 0.030


def parse_args():
    """Define argumentos para detectar comandos en audio largo."""
    parser = argparse.ArgumentParser(
        description=(
            "Graba o carga audio y detecta comandos base en orden usando "
            "segmentos de voz o ventanas deslizantes."
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
        "--segmentation-mode",
        choices=("windows", "vad"),
        default=DEFAULT_SEGMENTATION_MODE,
        help=(
            "Estrategia para dividir el audio antes de clasificar. "
            f"Default: {DEFAULT_SEGMENTATION_MODE}"
        ),
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=DEFAULT_ENERGY_THRESHOLD,
        help=(
            "Energia RMS minima para activar una ventana o frame VAD. "
            f"Default: {DEFAULT_ENERGY_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--min-segment-duration",
        type=float,
        default=DEFAULT_MIN_SEGMENT_DURATION,
        help=(
            "Duracion minima de voz para clasificar un segmento en segundos. "
            f"Default: {DEFAULT_MIN_SEGMENT_DURATION}"
        ),
    )
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=DEFAULT_MIN_SILENCE_DURATION,
        help=(
            "Silencio minimo para separar dos segmentos de voz en segundos. "
            f"Default: {DEFAULT_MIN_SILENCE_DURATION}"
        ),
    )
    parser.add_argument(
        "--segment-padding",
        type=float,
        default=DEFAULT_SEGMENT_PADDING,
        help=(
            "Contexto agregado antes y despues de cada segmento en segundos. "
            f"Default: {DEFAULT_SEGMENT_PADDING}"
        ),
    )
    parser.add_argument(
        "--auto-energy",
        action="store_true",
        help="Estima automaticamente el umbral RMS desde el audio.",
    )
    parser.add_argument(
        "--min-command-gap",
        type=float,
        default=DEFAULT_MIN_COMMAND_GAP,
        help=(
            "Separacion minima en segundos entre comandos aceptados. "
            f"Default: {DEFAULT_MIN_COMMAND_GAP}"
        ),
    )
    parser.add_argument(
        "--min-confirmations",
        type=int,
        default=DEFAULT_MIN_CONFIRMATIONS,
        help=(
            "Cantidad minima de ventanas cercanas con la misma clase para "
            f"confirmar un comando. Default: {DEFAULT_MIN_CONFIRMATIONS}"
        ),
    )
    parser.add_argument(
        "--confirmation-window",
        type=float,
        default=DEFAULT_CONFIRMATION_WINDOW,
        help=(
            "Ventana temporal en segundos para agrupar candidatas cercanas. "
            f"Default: {DEFAULT_CONFIRMATION_WINDOW}"
        ),
    )
    parser.add_argument(
        "--high-confidence-override",
        type=float,
        default=DEFAULT_HIGH_CONFIDENCE_OVERRIDE,
        help=(
            "Confianza minima para aceptar un grupo aunque no alcance las "
            f"confirmaciones requeridas. Default: {DEFAULT_HIGH_CONFIDENCE_OVERRIDE}"
        ),
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
    parser.add_argument(
        "--send-esp32",
        action="store_true",
        help="Envia la secuencia detectada a la ESP32 por Bluetooth clasico.",
    )
    parser.add_argument(
        "--esp32-port",
        default=DEFAULT_ESP32_PORT,
        help=f"Puerto COM de la ESP32. Default: {DEFAULT_ESP32_PORT}",
    )
    parser.add_argument(
        "--command-delay",
        type=float,
        default=DEFAULT_COMMAND_DELAY,
        help=(
            "Espera entre comandos enviados a la ESP32 en segundos. "
            f"Default: {DEFAULT_COMMAND_DELAY}"
        ),
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
    if args.energy_threshold < 0:
        raise ValueError("--energy-threshold no puede ser negativo.")
    if args.min_segment_duration <= 0:
        raise ValueError("--min-segment-duration debe ser mayor que 0.")
    if args.min_silence_duration < 0:
        raise ValueError("--min-silence-duration no puede ser negativo.")
    if args.segment_padding < 0:
        raise ValueError("--segment-padding no puede ser negativo.")
    if args.min_command_gap < 0:
        raise ValueError("--min-command-gap no puede ser negativo.")
    if args.min_confirmations <= 0:
        raise ValueError("--min-confirmations debe ser mayor que 0.")
    if args.confirmation_window < 0:
        raise ValueError("--confirmation-window no puede ser negativo.")
    if not 0 <= args.high_confidence_override <= 1:
        raise ValueError("--high-confidence-override debe estar entre 0 y 1.")
    if args.command_delay < 0:
        raise ValueError("--command-delay no puede ser negativo.")


def build_default_args(**overrides):
    """Construye argumentos por defecto para reutilizar inferencia desde la UI."""
    args = argparse.Namespace(
        duration=DEFAULT_DURATION,
        window_duration=DEFAULT_WINDOW_DURATION,
        hop_duration=DEFAULT_HOP_DURATION,
        threshold=DEFAULT_THRESHOLD,
        segmentation_mode=DEFAULT_SEGMENTATION_MODE,
        energy_threshold=DEFAULT_ENERGY_THRESHOLD,
        min_segment_duration=DEFAULT_MIN_SEGMENT_DURATION,
        min_silence_duration=DEFAULT_MIN_SILENCE_DURATION,
        segment_padding=DEFAULT_SEGMENT_PADDING,
        auto_energy=False,
        min_command_gap=DEFAULT_MIN_COMMAND_GAP,
        min_confirmations=DEFAULT_MIN_CONFIRMATIONS,
        confirmation_window=DEFAULT_CONFIRMATION_WINDOW,
        high_confidence_override=DEFAULT_HIGH_CONFIDENCE_OVERRIDE,
        device=None,
        list_devices=False,
        audio_file=None,
        send_esp32=False,
        esp32_port=DEFAULT_ESP32_PORT,
        command_delay=DEFAULT_COMMAND_DELAY,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    validate_args(args)
    return args


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


def calculate_rms_energy(audio):
    """Calcula la energia RMS de una ventana de audio cruda."""
    if len(audio) == 0:
        return 0.0

    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))


def estimate_energy_threshold(window_energies):
    """Estima un umbral RMS conservador a partir de energia baja del audio."""
    if not window_energies:
        return DEFAULT_ENERGY_THRESHOLD

    noise_estimate = float(np.percentile(window_energies, 25))
    return max(DEFAULT_ENERGY_THRESHOLD, noise_estimate * 3)


def iter_energy_frames(audio, sample_rate, frame_duration=VAD_FRAME_DURATION):
    """Genera frames cortos para detectar actividad por energia RMS."""
    frame_samples = max(1, int(round(frame_duration * sample_rate)))

    for start_sample in range(0, len(audio), frame_samples):
        end_sample = min(start_sample + frame_samples, len(audio))
        frame_audio = audio[start_sample:end_sample]
        yield start_sample, end_sample, calculate_rms_energy(frame_audio)


def estimate_frame_energy_threshold(audio, sample_rate):
    """Estima umbral RMS usando frames cortos del audio completo."""
    frame_energies = [
        energy
        for _start_sample, _end_sample, energy in iter_energy_frames(audio, sample_rate)
    ]
    return estimate_energy_threshold(frame_energies)


def segment_audio_by_energy(
    audio,
    sample_rate,
    energy_threshold,
    min_segment_duration,
    min_silence_duration,
):
    """Segmenta audio en zonas de voz usando energia RMS por frames de 30 ms."""
    frames = list(iter_energy_frames(audio, sample_rate))
    active_segments = []
    current_start = None
    current_end = None

    for start_sample, end_sample, energy in frames:
        is_active = energy >= energy_threshold
        if is_active and current_start is None:
            current_start = start_sample
            current_end = end_sample
            continue

        if is_active:
            current_end = end_sample
            continue

        if current_start is not None:
            active_segments.append((current_start, current_end))
            current_start = None
            current_end = None

    if current_start is not None:
        active_segments.append((current_start, current_end))

    min_silence_samples = int(round(min_silence_duration * sample_rate))
    merged_segments = []
    for start_sample, end_sample in active_segments:
        if not merged_segments:
            merged_segments.append([start_sample, end_sample])
            continue

        previous_start, previous_end = merged_segments[-1]
        silence_samples = start_sample - previous_end
        if silence_samples <= min_silence_samples:
            merged_segments[-1] = [previous_start, end_sample]
            continue

        merged_segments.append([start_sample, end_sample])

    min_segment_samples = int(round(min_segment_duration * sample_rate))
    segments = []
    for start_sample, end_sample in merged_segments:
        duration_samples = end_sample - start_sample
        if duration_samples < min_segment_samples:
            continue

        segment_audio = audio[start_sample:end_sample]
        segments.append(
            {
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start": start_sample / sample_rate,
                "end": end_sample / sample_rate,
                "duration": duration_samples / sample_rate,
                "energy": calculate_rms_energy(segment_audio),
            }
        )

    return segments


def predict_window(model, labels, audio, input_shape, mean, std):
    """Predice la clase ganadora y su confianza para una ventana."""
    X = prepare_audio_for_model(audio, input_shape, mean, std)
    probabilities = model.predict(X, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = labels[predicted_index]
    confidence = float(probabilities[predicted_index])
    return predicted_label, confidence


def classify_window_state(label, confidence, threshold, labels):
    """Determina si una prediccion es candidata valida para confirmacion."""
    if confidence < threshold:
        return "BAJA_CONFIANZA", False
    if label == BACKGROUND_LABEL:
        return "RUIDO_IGNORADO", False
    if label not in labels:
        return "CLASE_INVALIDA_IGNORADO", False

    return "CANDIDATO", True


def group_candidates(candidates, confirmation_window):
    """Agrupa candidatas cuyos inicios caen dentro de una ventana temporal."""
    groups = []
    current_group = []
    current_start = None

    for candidate in candidates:
        if not current_group:
            current_group = [candidate]
            current_start = candidate["start"]
            continue

        if candidate["start"] - current_start <= confirmation_window:
            current_group.append(candidate)
            continue

        groups.append(current_group)
        current_group = [candidate]
        current_start = candidate["start"]

    if current_group:
        groups.append(current_group)

    return groups


def summarize_candidate_group(group):
    """Elige la clase dominante y calcula metricas de confirmacion del grupo."""
    class_counts = {}
    class_max_confidences = {}

    for candidate in group:
        label = candidate["label"]
        confidence = candidate["confidence"]
        class_counts[label] = class_counts.get(label, 0) + 1
        class_max_confidences[label] = max(
            class_max_confidences.get(label, 0.0),
            confidence,
        )

    chosen_label = max(
        class_counts,
        key=lambda label: (class_counts[label], class_max_confidences[label]),
    )
    max_confidence = max(candidate["confidence"] for candidate in group)

    return {
        "start": group[0]["start"],
        "end": group[-1]["end"],
        "label": chosen_label,
        "confirmations": class_counts[chosen_label],
        "max_confidence": max_confidence,
        "rows": group,
    }


def confirm_candidate_groups(candidates, rows, args):
    """Confirma grupos de candidatas y construye la secuencia final."""
    sequence = []
    confirmed_groups = []
    last_accepted = None
    last_accepted_time = None

    for group in group_candidates(candidates, args.confirmation_window):
        summary = summarize_candidate_group(group)
        enough_confirmations = summary["confirmations"] >= args.min_confirmations
        high_confidence = summary["max_confidence"] >= args.high_confidence_override

        if not (enough_confirmations or high_confidence):
            for row in summary["rows"]:
                rows[row["row_index"]]["state"] = "CONFIRMACION_INSUFICIENTE"
            continue

        if (
            last_accepted_time is not None
            and summary["start"] - last_accepted_time < args.min_command_gap
        ):
            for row in summary["rows"]:
                rows[row["row_index"]]["state"] = "GAP_CORTO_IGNORADO"
            continue

        if summary["label"] == last_accepted:
            for row in summary["rows"]:
                rows[row["row_index"]]["state"] = "REPETIDO_IGNORADO"
            continue

        sequence.append(summary["label"])
        last_accepted = summary["label"]
        last_accepted_time = summary["start"]
        confirmed_groups.append(summary)

        for row in summary["rows"]:
            rows[row["row_index"]]["state"] = "ACEPTADO"

    return sequence, confirmed_groups


def run_sequence_inference(model, labels, mean, std, input_shape, audio, args):
    """Ejecuta inferencia por ventanas y construye la secuencia final."""
    rows = []
    candidates = []
    windows = list(iter_windows(audio, args.window_duration, args.hop_duration))
    window_energies = [calculate_rms_energy(window_audio) for _, _, window_audio in windows]
    energy_threshold = (
        estimate_energy_threshold(window_energies)
        if args.auto_energy
        else args.energy_threshold
    )

    if args.auto_energy:
        print(f"Umbral RMS automatico: {energy_threshold:.6f}")

    for (start_time, end_time, window_audio), energy in zip(windows, window_energies):
        if energy < energy_threshold:
            rows.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "label": "SIN_CLASIFICAR",
                    "confidence": 0.0,
                    "energy": energy,
                    "state": "ENERGIA_BAJA_IGNORADO",
                }
            )
            continue

        label, confidence = predict_window(model, labels, window_audio, input_shape, mean, std)
        state, accepted = classify_window_state(
            label,
            confidence,
            args.threshold,
            labels,
        )

        rows.append(
            {
                "start": start_time,
                "end": end_time,
                "label": label,
                "confidence": confidence,
                "energy": energy,
                "state": state,
            }
        )
        if accepted:
            candidates.append(
                {
                    "row_index": len(rows) - 1,
                    "start": start_time,
                    "end": end_time,
                    "label": label,
                    "confidence": confidence,
                }
            )

    sequence, confirmed_groups = confirm_candidate_groups(candidates, rows, args)
    return rows, sequence, confirmed_groups


def classify_segment_state(label, confidence, threshold, labels, last_accepted):
    """Determina si una prediccion de segmento entra a la secuencia final."""
    if confidence < threshold:
        return "BAJA_CONFIANZA", False
    if label == BACKGROUND_LABEL:
        return "RUIDO_IGNORADO", False
    if label not in labels:
        return "CLASE_INVALIDA_IGNORADO", False
    if label == last_accepted:
        return "REPETIDO_IGNORADO", False

    return "ACEPTADO", True


def extract_segment_with_padding(audio, segment, padding, sample_rate):
    """Extrae el segmento detectado agregando contexto a ambos lados."""
    padding_samples = int(round(padding * sample_rate))
    start_sample = max(0, segment["start_sample"] - padding_samples)
    end_sample = min(len(audio), segment["end_sample"] + padding_samples)
    return audio[start_sample:end_sample]


def run_vad_sequence_inference(model, labels, mean, std, input_shape, audio, args):
    """Segmenta por energia y clasifica cada segmento completo."""
    rows = []
    sequence = []
    last_accepted = None
    energy_threshold = (
        estimate_frame_energy_threshold(audio, SAMPLE_RATE)
        if args.auto_energy
        else args.energy_threshold
    )

    if args.auto_energy:
        print(f"Umbral RMS automatico: {energy_threshold:.6f}")

    segments = segment_audio_by_energy(
        audio,
        SAMPLE_RATE,
        energy_threshold,
        args.min_segment_duration,
        args.min_silence_duration,
    )

    for segment in segments:
        segment_audio = extract_segment_with_padding(
            audio,
            segment,
            args.segment_padding,
            SAMPLE_RATE,
        )
        label, confidence = predict_window(
            model,
            labels,
            segment_audio,
            input_shape,
            mean,
            std,
        )
        state, accepted = classify_segment_state(
            label,
            confidence,
            args.threshold,
            labels,
            last_accepted,
        )

        if accepted:
            sequence.append(label)
            last_accepted = label

        rows.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["duration"],
                "label": label,
                "confidence": confidence,
                "energy": segment["energy"],
                "state": state,
            }
        )

    return rows, sequence


def print_window_table(rows):
    """Muestra una tabla compacta de predicciones por ventana."""
    headers = (
        "tiempo_inicio",
        "tiempo_fin",
        "clase_predicha",
        "confianza",
        "energia_rms",
        "estado",
    )
    print()
    print(
        f"{headers[0]:>13}  {headers[1]:>10}  "
        f"{headers[2]:<18}  {headers[3]:>10}  {headers[4]:>11}  {headers[5]}"
    )
    print("-" * 86)

    for row in rows:
        print(
            f"{row['start']:>13.2f}  {row['end']:>10.2f}  "
            f"{row['label']:<18}  {row['confidence']:>10.4f}  "
            f"{row['energy']:>11.6f}  {row['state']}"
        )


def print_segment_table(rows):
    """Muestra una tabla compacta de predicciones por segmento de voz."""
    headers = (
        "tiempo_inicio",
        "tiempo_fin",
        "duracion",
        "clase_predicha",
        "confianza",
        "energia_rms",
        "estado",
    )
    print()
    print(
        f"{headers[0]:>13}  {headers[1]:>10}  {headers[2]:>8}  "
        f"{headers[3]:<18}  {headers[4]:>10}  {headers[5]:>11}  {headers[6]}"
    )
    print("-" * 98)

    if not rows:
        print("(sin segmentos de voz detectados)")
        return

    for row in rows:
        print(
            f"{row['start']:>13.2f}  {row['end']:>10.2f}  "
            f"{row['duration']:>8.2f}  {row['label']:<18}  "
            f"{row['confidence']:>10.4f}  {row['energy']:>11.6f}  {row['state']}"
        )


def print_sequence(sequence):
    """Muestra la secuencia final de comandos detectados."""
    print()
    if sequence:
        print("Secuencia final detectada:")
        print(" -> ".join(sequence))
    else:
        print("Secuencia final detectada: (sin comandos)")


def send_sequence_to_esp32(sequence, port, command_delay):
    """Envia comandos detectados a la ESP32 en el mismo orden."""
    from src.hardware.esp32_bluetooth_controller import ESP32BluetoothController

    commands = [command for command in sequence if command != BACKGROUND_LABEL]
    if not commands:
        print("ESP32: no hay comandos validos para enviar.")
        return

    controller = None
    try:
        controller = ESP32BluetoothController(port=port)
        for index, command in enumerate(commands):
            controller.send_command(command)
            for response in controller.read_responses():
                print(f"Respuesta ESP32: {response}")
            if index < len(commands) - 1 and command_delay > 0:
                time.sleep(command_delay)
    except ConnectionError as exc:
        print(f"Error ESP32: {exc}")
    finally:
        if controller is not None:
            controller.close()


def load_inference_resources():
    """Carga modelo, etiquetas y parametros para reutilizarlos desde la UI."""
    model = load_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
    mean, std = load_preprocess_params(PREPROCESS_PARAMS_PATH)
    input_shape = get_model_input_shape(model)
    validate_model_outputs(model, labels)
    return model, labels, mean, std, input_shape


def run_sequence_live_inference(args, resources=None):
    """Graba o carga audio largo y devuelve la secuencia detectada."""
    if resources is None:
        resources = load_inference_resources()

    model, labels, mean, std, input_shape = resources
    if args.audio_file is not None:
        audio = load_wav_audio(args.audio_file)
        print(f"Audio cargado: {args.audio_file} ({len(audio) / SAMPLE_RATE:.2f} s)")
    else:
        audio = record_audio(args.duration, device=args.device)

    confirmed_groups = None
    if args.segmentation_mode == "windows":
        rows, sequence, confirmed_groups = run_sequence_inference(
            model,
            labels,
            mean,
            std,
            input_shape,
            audio,
            args,
        )
        print_window_table(rows)
        print_confirmed_groups(confirmed_groups)
    else:
        rows, sequence = run_vad_sequence_inference(
            model,
            labels,
            mean,
            std,
            input_shape,
            audio,
            args,
        )
        print_segment_table(rows)

    print_sequence(sequence)
    return {
        "rows": rows,
        "sequence": sequence,
        "confirmed_groups": confirmed_groups,
    }


def print_confirmed_groups(groups):
    """Muestra los grupos que pasaron la confirmacion temporal."""
    print()
    print("Grupos confirmados:")
    if not groups:
        print("(sin grupos confirmados)")
        return

    headers = (
        "tiempo_inicio",
        "tiempo_fin",
        "clase_elegida",
        "cantidad_confirmaciones",
        "confianza_maxima",
    )
    print(
        f"{headers[0]:>13}  {headers[1]:>10}  "
        f"{headers[2]:<18}  {headers[3]:>24}  {headers[4]:>16}"
    )
    print("-" * 92)

    for group in groups:
        print(
            f"{group['start']:>13.2f}  {group['end']:>10.2f}  "
            f"{group['label']:<18}  {group['confirmations']:>24}  "
            f"{group['max_confidence']:>16.4f}"
        )


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    validate_args(args)

    result = run_sequence_live_inference(args)
    sequence = result["sequence"]
    if args.send_esp32:
        send_sequence_to_esp32(sequence, args.esp32_port, args.command_delay)


if __name__ == "__main__":
    main()

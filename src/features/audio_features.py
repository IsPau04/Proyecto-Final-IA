"""Funciones para extraer caracteristicas de audio desde archivos WAV."""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.config import SAMPLE_RATE


# Parametros configurables para mantener consistente la extraccion de
# caracteristicas durante entrenamiento, validacion e inferencia.
DEFAULT_N_MELS = 128
DEFAULT_N_MFCC = 13
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512


def load_audio(file_path):
    """Carga un archivo WAV mono usando el SAMPLE_RATE definido en src.config."""
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio


def normalize_audio(audio):
    """Normaliza la amplitud del audio al rango [-1, 1]."""
    max_amplitude = np.max(np.abs(audio))

    # Si el audio esta vacio o completamente en silencio, se devuelve igual para
    # evitar divisiones por cero.
    if max_amplitude == 0:
        return audio

    return audio / max_amplitude


def trim_silence_energy(
    audio,
    frame_length=DEFAULT_N_FFT,
    hop_length=DEFAULT_HOP_LENGTH,
    threshold_ratio=0.1,
):
    """Quita silencios al inicio y al final usando un VAD basico por energia."""
    if audio.size == 0:
        return audio

    # RMS estima la energia por ventanas cortas. Las ventanas con energia mayor
    # al umbral se consideran actividad de voz o sonido util.
    rms_energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    if rms_energy.size == 0:
        return audio

    threshold = np.max(rms_energy) * threshold_ratio
    active_frames = np.flatnonzero(rms_energy > threshold)

    # Si no hay ventanas activas, se conserva el audio original para no perder
    # ejemplos de RUIDO_FONDO o grabaciones de baja energia.
    if active_frames.size == 0:
        return audio

    start_sample = librosa.frames_to_samples(active_frames[0], hop_length=hop_length)
    end_sample = librosa.frames_to_samples(
        active_frames[-1] + 1,
        hop_length=hop_length,
    )

    return audio[start_sample:min(end_sample, len(audio))]


def extract_mel_spectrogram(
    audio,
    n_mels=DEFAULT_N_MELS,
    n_fft=DEFAULT_N_FFT,
    hop_length=DEFAULT_HOP_LENGTH,
):
    """Extrae un Mel-Spectrogram de potencia desde la senal de audio."""
    return librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )


def extract_log_mel_spectrogram(
    audio,
    n_mels=DEFAULT_N_MELS,
    n_fft=DEFAULT_N_FFT,
    hop_length=DEFAULT_HOP_LENGTH,
):
    """Convierte un Mel-Spectrogram de potencia a escala logaritmica."""
    mel_spectrogram = extract_mel_spectrogram(
        audio,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def extract_mfcc(
    audio,
    n_mfcc=DEFAULT_N_MFCC,
    n_mels=DEFAULT_N_MELS,
    n_fft=DEFAULT_N_FFT,
    hop_length=DEFAULT_HOP_LENGTH,
):
    """Extrae coeficientes MFCC desde la senal de audio."""
    return librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )


def save_spectrogram_image(log_mel, output_path):
    """Guarda una imagen PNG del espectrograma log-Mel para documentacion."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    # El espectrograma representa como cambia la energia de las frecuencias a
    # traves del tiempo: el eje X es tiempo, el eje Y son bandas Mel y el color
    # indica energia en decibeles.
    librosa.display.specshow(
        log_mel,
        sr=SAMPLE_RATE,
        hop_length=DEFAULT_HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    plt.close()

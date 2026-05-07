"""Extraccion de caracteristicas de audio para comandos de voz."""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.config import SAMPLE_RATE


def load_audio(file_path):
    """Carga un archivo WAV usando la frecuencia de muestreo del proyecto."""
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio


def normalize_audio(audio):
    """Normaliza la amplitud al rango [-1, 1] sin cambiar la forma de la senal."""
    max_amplitude = np.max(np.abs(audio))

    if max_amplitude == 0:
        return audio

    return audio / max_amplitude


def trim_silence_energy(audio, frame_length=2048, hop_length=512, threshold_ratio=0.1):
    """Quita silencio al inicio y final usando un VAD basico por energia."""
    if audio.size == 0:
        return audio

    rms_energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    if rms_energy.size == 0:
        return audio

    threshold = np.max(rms_energy) * threshold_ratio
    active_frames = np.flatnonzero(rms_energy > threshold)

    if active_frames.size == 0:
        return audio

    start_sample = librosa.frames_to_samples(active_frames[0], hop_length=hop_length)
    end_sample = librosa.frames_to_samples(
        active_frames[-1] + 1,
        hop_length=hop_length,
    )

    return audio[start_sample:min(end_sample, len(audio))]


def extract_mel_spectrogram(audio, n_mels=128, n_fft=2048, hop_length=512):
    """Extrae un Mel-Spectrogram de potencia desde la senal de audio."""
    return librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )


def extract_log_mel_spectrogram(audio, n_mels=128, n_fft=2048, hop_length=512):
    """Convierte el Mel-Spectrogram a escala logaritmica en decibeles."""
    mel_spectrogram = extract_mel_spectrogram(
        audio,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def extract_mfcc(audio, n_mfcc=13, n_mels=128, n_fft=2048, hop_length=512):
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
    # traves del tiempo: eje X = tiempo, eje Y = bandas Mel, color = energia.
    librosa.display.specshow(
        log_mel,
        sr=SAMPLE_RATE,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

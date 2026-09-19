import os

import numpy as np
import librosa
import librosa.display

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

N_FFT = 512

HOP_LENGTH = 128

WIN_LENGTH = 512


# ============================================================
# FILES
# ============================================================

CLEAN_FILE = (
    "data/processed/train/"
    "sample_0003_clean.wav"
)

NOISY_FILE = (
    "data/processed/train/"
    "sample_0003_snr_5dB.wav"
)


OUTPUT_FILE = (
    "outputs/spectrograms/"
    "clean_vs_noisy_5dB.png"
)


# ============================================================
# LOAD AUDIO
# ============================================================

clean, sr = librosa.load(
    CLEAN_FILE,
    sr=SAMPLE_RATE,
    mono=True
)

noisy, _ = librosa.load(
    NOISY_FILE,
    sr=SAMPLE_RATE,
    mono=True
)


# ============================================================
# STFT
# ============================================================

clean_stft = librosa.stft(
    clean,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH
)

noisy_stft = librosa.stft(
    noisy,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH
)


# ============================================================
# MAGNITUDE
# ============================================================

clean_magnitude = np.abs(
    clean_stft
)

noisy_magnitude = np.abs(
    noisy_stft
)


# ============================================================
# DECIBEL CONVERSION
# ============================================================

clean_db = librosa.amplitude_to_db(
    clean_magnitude,
    ref=np.max
)

noisy_db = librosa.amplitude_to_db(
    noisy_magnitude,
    ref=np.max
)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 10)
)


# ------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------

img1 = librosa.display.specshow(
    clean_db,
    sr=sr,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="hz",
    ax=axes[0]
)

axes[0].set_title(
    "Clean Speech Spectrogram"
)

fig.colorbar(
    img1,
    ax=axes[0],
    format="%+2.0f dB"
)


# ------------------------------------------------------------
# NOISY
# ------------------------------------------------------------

img2 = librosa.display.specshow(
    noisy_db,
    sr=sr,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="hz",
    ax=axes[1]
)

axes[1].set_title(
    "Noisy Speech Spectrogram - 5 dB SNR"
)

fig.colorbar(
    img2,
    ax=axes[1],
    format="%+2.0f dB"
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "Comparison saved to:"
)

print(
    OUTPUT_FILE
)
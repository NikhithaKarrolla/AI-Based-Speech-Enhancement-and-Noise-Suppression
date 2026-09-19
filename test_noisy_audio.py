import os

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import librosa
import librosa.display


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
    "outputs/audio/"
    "clean_vs_noisy.png"
)


# ============================================================
# LOAD
# ============================================================

clean, sr = librosa.load(
    CLEAN_FILE,
    sr=None,
    mono=True
)

noisy, _ = librosa.load(
    NOISY_FILE,
    sr=sr,
    mono=True
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(14, 8)
)


plt.subplot(2, 1, 1)

librosa.display.waveshow(
    clean,
    sr=sr
)

plt.title(
    "Clean Speech"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)


plt.subplot(2, 1, 2)

librosa.display.waveshow(
    noisy,
    sr=sr
)

plt.title(
    "Noisy Speech - 5 dB SNR"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

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
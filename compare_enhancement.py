import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np


# ============================================
# FILES
# ============================================

CLEAN_FILE = "data/processed/train/sample_0003_clean.wav"

NOISY_FILE = "data/processed/train/sample_0003_snr_5dB.wav"

ENHANCED_FILE = (
    "outputs/audio/spectral_subtraction_5dB.wav"
)

OUTPUT_FILE = (
    "outputs/spectrograms/"
    "clean_noisy_enhanced_5dB.png"
)


# ============================================
# LOAD AUDIO
# ============================================

clean, sr = librosa.load(
    CLEAN_FILE,
    sr=16000,
    mono=True
)

noisy, _ = librosa.load(
    NOISY_FILE,
    sr=16000,
    mono=True
)

enhanced, _ = librosa.load(
    ENHANCED_FILE,
    sr=16000,
    mono=True
)


# ============================================
# COMPUTE SPECTROGRAMS
# ============================================

clean_stft = librosa.stft(
    clean,
    n_fft=512,
    hop_length=128,
    win_length=512
)

noisy_stft = librosa.stft(
    noisy,
    n_fft=512,
    hop_length=128,
    win_length=512
)

enhanced_stft = librosa.stft(
    enhanced,
    n_fft=512,
    hop_length=128,
    win_length=512
)


clean_db = librosa.amplitude_to_db(
    abs(clean_stft),
    ref=np.max
)

noisy_db = librosa.amplitude_to_db(
    abs(noisy_stft),
    ref=np.max
)

enhanced_db = librosa.amplitude_to_db(
    abs(enhanced_stft),
    ref=np.max
)

# ============================================
# PLOT
# ============================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 12)
)


# Clean
librosa.display.specshow(
    clean_db,
    sr=sr,
    hop_length=128,
    x_axis="time",
    y_axis="hz",
    ax=axes[0]
)

axes[0].set_title("Clean Speech")


# Noisy
librosa.display.specshow(
    noisy_db,
    sr=sr,
    hop_length=128,
    x_axis="time",
    y_axis="hz",
    ax=axes[1]
)

axes[1].set_title("Noisy Speech - 5 dB SNR")


# Enhanced
librosa.display.specshow(
    enhanced_db,
    sr=sr,
    hop_length=128,
    x_axis="time",
    y_axis="hz",
    ax=axes[2]
)

axes[2].set_title("Spectral Subtraction Enhanced")


plt.tight_layout()

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

plt.savefig(
    OUTPUT_FILE,
    dpi=150
)

plt.close()

print("Comparison saved to:")
print(OUTPUT_FILE)
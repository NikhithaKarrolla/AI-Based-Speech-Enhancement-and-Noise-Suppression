import os

import numpy as np
import librosa
import librosa.display

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.audio.stft import (
    compute_stft,
    get_magnitude,
    get_phase,
    reconstruct_audio
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

N_FFT = 512

HOP_LENGTH = 128

WIN_LENGTH = 512


# ============================================================
# INPUT FILE
# ============================================================

INPUT_FILE = (
    "data/processed/train/"
    "sample_0003_snr_5dB.wav"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    "outputs/spectrograms"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD AUDIO
# ============================================================

print()
print("=" * 60)
print("STFT TEST")
print("=" * 60)

print()
print("Loading audio:")

print(INPUT_FILE)


audio, sr = librosa.load(
    INPUT_FILE,
    sr=SAMPLE_RATE,
    mono=True
)


print()
print("Sample rate:", sr)

print(
    "Duration:",
    len(audio) / sr,
    "seconds"
)

print(
    "Samples:",
    len(audio)
)


# ============================================================
# COMPUTE STFT
# ============================================================

print()
print("Computing STFT...")

stft_matrix = compute_stft(
    audio,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH
)


print(
    "STFT shape:",
    stft_matrix.shape
)


# ============================================================
# MAGNITUDE
# ============================================================

magnitude = get_magnitude(
    stft_matrix
)


print(
    "Magnitude shape:",
    magnitude.shape
)


# ============================================================
# PHASE
# ============================================================

phase = get_phase(
    stft_matrix
)


print(
    "Phase shape:",
    phase.shape
)


# ============================================================
# CONVERT TO DECIBELS
# ============================================================

magnitude_db = librosa.amplitude_to_db(
    magnitude,
    ref=np.max
)


# ============================================================
# PLOT SPECTROGRAM
# ============================================================

plt.figure(
    figsize=(14, 6)
)


librosa.display.specshow(
    magnitude_db,
    sr=sr,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="hz"
)


plt.colorbar(
    format="%+2.0f dB"
)


plt.title(
    "Noisy Speech Spectrogram - 5 dB SNR"
)


plt.xlabel(
    "Time (seconds)"
)


plt.ylabel(
    "Frequency (Hz)"
)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "sample_0003_snr_5dB_spectrogram.png"
)


plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


print()
print(
    "Spectrogram saved to:"
)

print(
    output_file
)


# ============================================================
# TEST RECONSTRUCTION
# ============================================================

print()
print("Testing iSTFT reconstruction...")


reconstructed_audio = reconstruct_audio(
    magnitude,
    phase,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH
)


print(
    "Reconstructed samples:",
    len(reconstructed_audio)
)


# ============================================================
# SAVE RECONSTRUCTED AUDIO
# ============================================================

import soundfile as sf


reconstructed_file = (
    "outputs/audio/"
    "reconstructed_5dB.wav"
)


os.makedirs(
    "outputs/audio",
    exist_ok=True
)


sf.write(
    reconstructed_file,
    reconstructed_audio,
    SAMPLE_RATE
)


print(
    "Reconstructed audio saved to:"
)

print(
    reconstructed_file
)


print()
print("=" * 60)
print("STFT TEST COMPLETED")
print("=" * 60)
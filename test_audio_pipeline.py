import os
import sys

import numpy as np
import librosa
import librosa.display

# IMPORTANT:
# Use a non-GUI backend so Matplotlib does not require Tkinter.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.audio.preprocessing import (
    load_audio,
    normalize_audio,
    save_audio
)

from src.audio.utils import (
    get_audio_info
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "clean"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train"
)

WAVEFORM_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "audio"
)


# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not os.path.exists(INPUT_DIR):

    print(
        "ERROR: Clean audio directory does not exist:"
    )

    print(INPUT_DIR)

    sys.exit(1)


# ============================================================
# FIND WAV FILES
# ============================================================

wav_files = [
    file
    for file in os.listdir(INPUT_DIR)
    if file.lower().endswith(".wav")
]


if len(wav_files) == 0:

    print("\nNo WAV files found.")

    print(
        "Please place a clean speech WAV file inside:"
    )

    print(INPUT_DIR)

    sys.exit(1)


# ============================================================
# SELECT INPUT FILE
# ============================================================

input_file = os.path.join(
    INPUT_DIR,
    wav_files[0]
)

output_file = os.path.join(
    OUTPUT_DIR,
    "clean_0001.wav"
)

waveform_file = os.path.join(
    WAVEFORM_DIR,
    "clean_0001_waveform.png"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    WAVEFORM_DIR,
    exist_ok=True
)


# ============================================================
# LOAD AUDIO
# ============================================================

print("\nLoading audio...")

audio, sample_rate = load_audio(
    input_file,
    sample_rate=16000
)


print("\nOriginal input:")

print(
    "File:",
    input_file
)

print(
    "Sample rate:",
    sample_rate
)

print(
    "Samples:",
    len(audio)
)

print(
    "Duration:",
    len(audio) / sample_rate,
    "seconds"
)


# ============================================================
# NORMALIZE AUDIO
# ============================================================

audio = normalize_audio(
    audio
)


# ============================================================
# SAVE PROCESSED AUDIO
# ============================================================

save_audio(
    output_file,
    audio,
    sample_rate
)


# ============================================================
# AUDIO INFORMATION
# ============================================================

info = get_audio_info(
    audio,
    sample_rate
)


print("\nProcessed audio information:")

for key, value in info.items():

    print(
        f"{key}: {value}"
    )


print(
    "\nProcessed audio saved to:"
)

print(
    output_file
)


# ============================================================
# CREATE WAVEFORM
# ============================================================

print(
    "\nCreating waveform plot..."
)

plt.figure(
    figsize=(14, 4)
)

librosa.display.waveshow(
    audio,
    sr=sample_rate
)

plt.title(
    "Processed Clean Speech Waveform"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.tight_layout()


# ============================================================
# SAVE WAVEFORM
# ============================================================

plt.savefig(
    waveform_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "\nWaveform saved to:"
)

print(
    waveform_file
)


print(
    "\n========================================"
)

print(
    "AUDIO PREPROCESSING COMPLETED SUCCESSFULLY"
)

print(
    "========================================"
)
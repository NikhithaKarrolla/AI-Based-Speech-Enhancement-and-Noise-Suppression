import os
import numpy as np
import soundfile as sf

from src.audio.preprocessing import load_audio
from src.audio.spectral_subtraction import spectral_subtraction


# ============================================
# CONFIGURATION
# ============================================

INPUT_FILE = "data/processed/train/sample_0003_snr_5dB.wav"

OUTPUT_FILE = "outputs/audio/spectral_subtraction_5dB.wav"

SAMPLE_RATE = 16000


# ============================================
# CHECK INPUT
# ============================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Run generate_noisy_dataset.py first."
    )


# ============================================
# LOAD AUDIO
# ============================================

print("Loading noisy audio...")

noisy_audio, sr = load_audio(
    INPUT_FILE,
    SAMPLE_RATE
)

print(f"Sample rate: {sr}")
print(f"Samples: {len(noisy_audio)}")
print(f"Duration: {len(noisy_audio) / sr:.2f} seconds")


# ============================================
# SPECTRAL SUBTRACTION
# ============================================

print("\nRunning spectral subtraction...")

enhanced_audio = spectral_subtraction(
    noisy_audio=noisy_audio,
    sample_rate=sr,
    n_fft=512,
    hop_length=128,
    win_length=512,
    noise_duration=0.5,
    alpha=1.0,
    beta=0.02
)


# ============================================
# NORMALIZATION
# ============================================

peak = np.max(np.abs(enhanced_audio))

if peak > 0.99:
    enhanced_audio = enhanced_audio / peak


# ============================================
# SAVE
# ============================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

sf.write(
    OUTPUT_FILE,
    enhanced_audio,
    sr
)


print("\nSpectral subtraction completed.")

print(f"Output saved to:")
print(OUTPUT_FILE)
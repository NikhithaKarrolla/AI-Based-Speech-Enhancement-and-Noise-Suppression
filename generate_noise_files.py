import os

import numpy as np
import soundfile as sf
from scipy import signal


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

DURATION = 60

OUTPUT_DIR = os.path.join(
    "data",
    "raw",
    "noise"
)

SEED = 42

np.random.seed(SEED)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# NUMBER OF SAMPLES
# ============================================================

num_samples = int(
    SAMPLE_RATE * DURATION
)


# ============================================================
# 1. WHITE NOISE
# ============================================================

print("Generating white noise...")

white_noise = np.random.randn(
    num_samples
).astype(np.float32)


# Normalize
white_noise = (
    white_noise /
    np.max(np.abs(white_noise))
)


white_noise_path = os.path.join(
    OUTPUT_DIR,
    "white_noise.wav"
)


sf.write(
    white_noise_path,
    white_noise,
    SAMPLE_RATE
)


# ============================================================
# 2. LOW-FREQUENCY NOISE
# ============================================================

print(
    "Generating low-frequency noise..."
)

raw_noise = np.random.randn(
    num_samples
)

# Low-pass filter
b, a = signal.butter(
    4,
    300,
    btype="low",
    fs=SAMPLE_RATE
)

low_frequency_noise = signal.filtfilt(
    b,
    a,
    raw_noise
)

low_frequency_noise = (
    low_frequency_noise /
    np.max(
        np.abs(
            low_frequency_noise
        )
    )
)

low_frequency_noise = (
    low_frequency_noise
    .astype(np.float32)
)


low_frequency_path = os.path.join(
    OUTPUT_DIR,
    "low_frequency_noise.wav"
)


sf.write(
    low_frequency_path,
    low_frequency_noise,
    SAMPLE_RATE
)


# ============================================================
# 3. MIXED SYNTHETIC NOISE
# ============================================================

print(
    "Generating mixed noise..."
)

# White noise
component_1 = np.random.randn(
    num_samples
)

# Low-frequency component
component_2 = signal.filtfilt(
    b,
    a,
    np.random.randn(num_samples)
)

# Combine
mixed_noise = (
    0.7 * component_1 +
    0.3 * component_2
)

mixed_noise = (
    mixed_noise /
    np.max(
        np.abs(mixed_noise)
    )
)

mixed_noise = (
    mixed_noise
    .astype(np.float32)
)


mixed_noise_path = os.path.join(
    OUTPUT_DIR,
    "mixed_noise.wav"
)


sf.write(
    mixed_noise_path,
    mixed_noise,
    SAMPLE_RATE
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 50)
print("NOISE GENERATION COMPLETED")
print("=" * 50)

print(
    f"Sample rate: {SAMPLE_RATE} Hz"
)

print(
    f"Duration: {DURATION} seconds"
)

print()
print("Generated files:")

print(
    white_noise_path
)

print(
    low_frequency_path
)

print(
    mixed_noise_path
)
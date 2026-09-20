import os

import numpy as np
import librosa
from pystoi import stoi


# ============================================
# CONFIGURATION
# ============================================

SAMPLE_RATE = 16000

TEST_DIR = "data/test"

CLEAN_FILE = "clean_0002_clean.wav"

NOISY_FILE = "clean_0002_snr_5dB.wav"

ENHANCED_FILE = (
    "cnn_enhanced_clean_0002_5dB.wav"
)

ENHANCED_DIR = "outputs/audio"


# ============================================
# PATHS
# ============================================

clean_path = os.path.join(
    TEST_DIR,
    "clean",
    CLEAN_FILE
)

noisy_path = os.path.join(
    TEST_DIR,
    "noisy",
    NOISY_FILE
)

enhanced_path = os.path.join(
    ENHANCED_DIR,
    ENHANCED_FILE
)


# ============================================
# CHECK FILES
# ============================================

for path in [
    clean_path,
    noisy_path,
    enhanced_path
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File not found: {path}"
        )


# ============================================
# LOAD AUDIO
# ============================================

clean, _ = librosa.load(
    clean_path,
    sr=SAMPLE_RATE,
    mono=True
)

noisy, _ = librosa.load(
    noisy_path,
    sr=SAMPLE_RATE,
    mono=True
)

enhanced, _ = librosa.load(
    enhanced_path,
    sr=SAMPLE_RATE,
    mono=True
)


# ============================================
# MATCH LENGTHS
# ============================================

length = min(
    len(clean),
    len(noisy),
    len(enhanced)
)

clean = clean[:length]
noisy = noisy[:length]
enhanced = enhanced[:length]


# ============================================
# SNR
# ============================================

def calculate_snr(
    clean_signal,
    estimated_signal
):

    noise = (
        estimated_signal
        - clean_signal
    )

    signal_power = np.mean(
        clean_signal ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    if noise_power == 0:

        return float("inf")

    return 10 * np.log10(
        signal_power / noise_power
    )


# ============================================
# SI-SDR
# ============================================

def calculate_si_sdr(
    reference,
    estimate
):

    reference = (
        reference
        - np.mean(reference)
    )

    estimate = (
        estimate
        - np.mean(estimate)
    )

    reference_energy = np.sum(
        reference ** 2
    )

    if reference_energy == 0:

        return float("-inf")

    scale = (
        np.dot(estimate, reference)
        / reference_energy
    )

    target = (
        scale * reference
    )

    noise = (
        estimate - target
    )

    target_energy = np.sum(
        target ** 2
    )

    noise_energy = np.sum(
        noise ** 2
    )

    if noise_energy == 0:

        return float("inf")

    return 10 * np.log10(
        target_energy
        / noise_energy
    )


# ============================================
# CALCULATE METRICS
# ============================================

noisy_snr = calculate_snr(
    clean,
    noisy
)

enhanced_snr = calculate_snr(
    clean,
    enhanced
)

noisy_si_sdr = calculate_si_sdr(
    clean,
    noisy
)

enhanced_si_sdr = calculate_si_sdr(
    clean,
    enhanced
)


# ============================================
# STOI
# ============================================

noisy_stoi = stoi(
    clean,
    noisy,
    SAMPLE_RATE,
    extended=False
)

enhanced_stoi = stoi(
    clean,
    enhanced,
    SAMPLE_RATE,
    extended=False
)


# ============================================
# IMPROVEMENTS
# ============================================

snr_improvement = (
    enhanced_snr
    - noisy_snr
)

si_sdr_improvement = (
    enhanced_si_sdr
    - noisy_si_sdr
)

stoi_improvement = (
    enhanced_stoi
    - noisy_stoi
)


# ============================================
# RESULTS
# ============================================

print("=" * 60)
print("CNN SPEECH ENHANCEMENT EVALUATION")
print("=" * 60)

print("\nInput:")
print("clean_0002_snr_5dB.wav")

print("\n--------------------------------------------")
print("SNR")
print("--------------------------------------------")

print(
    f"Noisy SNR:       {noisy_snr:.4f} dB"
)

print(
    f"Enhanced SNR:    {enhanced_snr:.4f} dB"
)

print(
    f"SNR Improvement: {snr_improvement:.4f} dB"
)


print("\n--------------------------------------------")
print("SI-SDR")
print("--------------------------------------------")

print(
    f"Noisy SI-SDR:       {noisy_si_sdr:.4f} dB"
)

print(
    f"Enhanced SI-SDR:    {enhanced_si_sdr:.4f} dB"
)

print(
    f"SI-SDR Improvement: {si_sdr_improvement:.4f} dB"
)


print("\n--------------------------------------------")
print("STOI")
print("--------------------------------------------")

print(
    f"Noisy STOI:       {noisy_stoi:.4f}"
)

print(
    f"Enhanced STOI:    {enhanced_stoi:.4f}"
)

print(
    f"STOI Improvement: {stoi_improvement:.4f}"
)


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
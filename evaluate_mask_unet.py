import os

import numpy as np
import pandas as pd
import librosa
from pystoi import stoi


# ============================================
# Configuration
# ============================================

NOISY_DIR = "data/test/noisy"

CLEAN_DIR = "data/test/clean"

ENHANCED_DIR = "outputs/audio"

OUTPUT_CSV = (
    "outputs/metrics/mask_unet_test_results.csv"
)

SAMPLE_RATE = 16000


# ============================================
# SNR
# ============================================

def calculate_snr(clean, estimate):

    clean = np.asarray(
        clean,
        dtype=np.float64
    )

    estimate = np.asarray(
        estimate,
        dtype=np.float64
    )

    noise = clean - estimate

    signal_power = np.sum(
        clean ** 2
    )

    noise_power = np.sum(
        noise ** 2
    )

    if noise_power < 1e-12:

        return 100.0

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

    reference = np.asarray(
        reference,
        dtype=np.float64
    )

    estimate = np.asarray(
        estimate,
        dtype=np.float64
    )

    length = min(
        len(reference),
        len(estimate)
    )

    reference = reference[:length]

    estimate = estimate[:length]

    reference_energy = np.sum(
        reference ** 2
    )

    if reference_energy < 1e-12:

        return 0.0

    scale = (
        np.dot(
            estimate,
            reference
        )
        /
        reference_energy
    )

    projection = (
        scale * reference
    )

    error = (
        estimate - projection
    )

    projection_energy = np.sum(
        projection ** 2
    )

    error_energy = np.sum(
        error ** 2
    )

    if error_energy < 1e-12:

        return 100.0

    return 10 * np.log10(
        projection_energy
        /
        error_energy
    )


# ============================================
# STOI
# ============================================

def calculate_stoi(
    clean,
    estimate
):

    length = min(
        len(clean),
        len(estimate)
    )

    clean = clean[:length]

    estimate = estimate[:length]

    return stoi(
        clean,
        estimate,
        SAMPLE_RATE,
        extended=False
    )


# ============================================
# Get matching clean file
# ============================================

def get_clean_filename(
    noisy_filename
):

    base_name = noisy_filename.split(
        "_snr_"
    )[0]

    return (
        f"{base_name}_clean.wav"
    )


# ============================================
# Evaluate all files
# ============================================

results = []


noisy_files = sorted(
    file_name
    for file_name in os.listdir(
        NOISY_DIR
    )
    if file_name.lower().endswith(".wav")
)


print(
    "Found",
    len(noisy_files),
    "test files."
)

print()


for noisy_filename in noisy_files:

    # ----------------------------------------
    # Paths
    # ----------------------------------------

    noisy_path = os.path.join(
        NOISY_DIR,
        noisy_filename
    )

    clean_filename = get_clean_filename(
        noisy_filename
    )

    clean_path = os.path.join(
        CLEAN_DIR,
        clean_filename
    )

    enhanced_filename = (
        noisy_filename
        .replace(
            "_snr_",
            "_"
        )
        .replace(
            ".wav",
            ""
        )
    )

    enhanced_filename = (
        "mask_unet_enhanced_"
        + enhanced_filename
        + ".wav"
    )

    enhanced_path = os.path.join(
        ENHANCED_DIR,
        enhanced_filename
    )


    # ----------------------------------------
    # Check files
    # ----------------------------------------

    if not os.path.exists(
        clean_path
    ):

        print(
            "Missing clean file:",
            clean_path
        )

        continue


    if not os.path.exists(
        enhanced_path
    ):

        print(
            "Missing enhanced file:",
            enhanced_path
        )

        continue


    # ----------------------------------------
    # Load audio
    # ----------------------------------------

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


    # ----------------------------------------
    # Align lengths
    # ----------------------------------------

    length = min(
        len(clean),
        len(noisy),
        len(enhanced)
    )

    clean = clean[:length]

    noisy = noisy[:length]

    enhanced = enhanced[:length]


    # ----------------------------------------
    # Calculate metrics
    # ----------------------------------------

    noisy_snr = calculate_snr(
        clean,
        noisy
    )

    enhanced_snr = calculate_snr(
        clean,
        enhanced
    )

    snr_improvement = (
        enhanced_snr
        -
        noisy_snr
    )


    noisy_si_sdr = calculate_si_sdr(
        clean,
        noisy
    )

    enhanced_si_sdr = calculate_si_sdr(
        clean,
        enhanced
    )

    si_sdr_improvement = (
        enhanced_si_sdr
        -
        noisy_si_sdr
    )


    noisy_stoi = calculate_stoi(
        clean,
        noisy
    )

    enhanced_stoi = calculate_stoi(
        clean,
        enhanced
    )

    stoi_improvement = (
        enhanced_stoi
        -
        noisy_stoi
    )


    # ----------------------------------------
    # Print
    # ----------------------------------------

    print(
        noisy_filename
    )

    print(
        f"  SNR: "
        f"{noisy_snr:.3f} → "
        f"{enhanced_snr:.3f} "
        f"({snr_improvement:+.3f} dB)"
    )

    print(
        f"  SI-SDR: "
        f"{noisy_si_sdr:.3f} → "
        f"{enhanced_si_sdr:.3f} "
        f"({si_sdr_improvement:+.3f} dB)"
    )

    print(
        f"  STOI: "
        f"{noisy_stoi:.3f} → "
        f"{enhanced_stoi:.3f} "
        f"({stoi_improvement:+.3f})"
    )

    print()


    # ----------------------------------------
    # Store result
    # ----------------------------------------

    results.append(
        {
            "file": noisy_filename,

            "noisy_snr": noisy_snr,
            "enhanced_snr": enhanced_snr,
            "snr_improvement": snr_improvement,

            "noisy_si_sdr": noisy_si_sdr,
            "enhanced_si_sdr": enhanced_si_sdr,
            "si_sdr_improvement": si_sdr_improvement,

            "noisy_stoi": noisy_stoi,
            "enhanced_stoi": enhanced_stoi,
            "stoi_improvement": stoi_improvement,
        }
    )


# ============================================
# Save results
# ============================================

# ============================================
# Save results
# ============================================

if not results:

    print()
    print("ERROR: No test files were evaluated.")
    print(
        "Please check that the clean and enhanced "
        "audio files exist."
    )

    raise SystemExit(1)


df = pd.DataFrame(
    results
)


os.makedirs(
    os.path.dirname(
        OUTPUT_CSV
    ),
    exist_ok=True
)


df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================
# Average results
# ============================================

print(
    "========================================"
)

print(
    "Average SNR improvement:",
    f"{df['snr_improvement'].mean():+.4f}",
    "dB"
)

print(
    "Average SI-SDR improvement:",
    f"{df['si_sdr_improvement'].mean():+.4f}",
    "dB"
)

print(
    "Average STOI improvement:",
    f"{df['stoi_improvement'].mean():+.4f}"
)

print(
    "========================================"
)

print(
    "Results saved to:",
    OUTPUT_CSV
)


os.makedirs(
    os.path.dirname(
        OUTPUT_CSV
    ),
    exist_ok=True
)


df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================
# Average results
# ============================================

print(
    "========================================"
)

print(
    "Average SNR improvement:",
    f"{df['snr_improvement'].mean():+.4f}",
    "dB"
)

print(
    "Average SI-SDR improvement:",
    f"{df['si_sdr_improvement'].mean():+.4f}",
    "dB"
)

print(
    "Average STOI improvement:",
    f"{df['stoi_improvement'].mean():+.4f}"
)

print(
    "========================================"
)

print(
    "Results saved to:",
    OUTPUT_CSV
)
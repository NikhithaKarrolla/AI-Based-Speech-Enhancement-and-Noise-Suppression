import os
import csv

import numpy as np
import librosa
from pystoi import stoi


# ============================================
# CONFIGURATION
# ============================================

SAMPLE_RATE = 16000

CLEAN_DIR = "data/test/clean"
NOISY_DIR = "data/test/noisy"

ENHANCED_DIR = "outputs/audio"

OUTPUT_DIR = "outputs/metrics"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "cnn_test_results.csv"
)


# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================
# METRICS
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

    target = scale * reference

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
# FIND TEST FILES
# ============================================

noisy_files = sorted(
    file
    for file in os.listdir(NOISY_DIR)
    if file.endswith(".wav")
)


print("=" * 70)
print("CNN TEST SET EVALUATION")
print("=" * 70)

print("\nNumber of noisy test samples:")
print(len(noisy_files))


# ============================================
# RESULTS
# ============================================

results = []


for noisy_file in noisy_files:

    # ----------------------------------------
    # Extract base name
    # ----------------------------------------

    base_name = noisy_file.split(
        "_snr_"
    )[0]

    clean_file = (
        f"{base_name}_clean.wav"
    )

    # Example:
    #
    # clean_0002_snr_5dB.wav
    #
    # →
    #
    # cnn_enhanced_clean_0002_5dB.wav

    snr_part = noisy_file.split(
        "_snr_"
    )[1]

    enhanced_file = (
        f"cnn_enhanced_"
        f"{base_name}_"
        f"{snr_part}"
    )


    # ----------------------------------------
    # Paths
    # ----------------------------------------

    clean_path = os.path.join(
        CLEAN_DIR,
        clean_file
    )

    noisy_path = os.path.join(
        NOISY_DIR,
        noisy_file
    )

    enhanced_path = os.path.join(
        ENHANCED_DIR,
        enhanced_file
    )


    # ----------------------------------------
    # Check files
    # ----------------------------------------

    if not os.path.exists(
        clean_path
    ):

        print(
            f"\nSkipping {noisy_file}: "
            f"clean file missing"
        )

        continue


    if not os.path.exists(
        enhanced_path
    ):

        print(
            f"\nSkipping {noisy_file}: "
            f"enhanced file missing"
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
    # Match lengths
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
    # Metrics
    # ----------------------------------------

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


    # ----------------------------------------
    # Store
    # ----------------------------------------

    results.append({

        "file": noisy_file,

        "noisy_snr": noisy_snr,

        "enhanced_snr": enhanced_snr,

        "snr_improvement":
            enhanced_snr - noisy_snr,

        "noisy_si_sdr":
            noisy_si_sdr,

        "enhanced_si_sdr":
            enhanced_si_sdr,

        "si_sdr_improvement":
            enhanced_si_sdr - noisy_si_sdr,

        "noisy_stoi":
            noisy_stoi,

        "enhanced_stoi":
            enhanced_stoi,

        "stoi_improvement":
            enhanced_stoi - noisy_stoi
    })


# ============================================
# SAVE CSV
# ============================================

fieldnames = [
    "file",
    "noisy_snr",
    "enhanced_snr",
    "snr_improvement",
    "noisy_si_sdr",
    "enhanced_si_sdr",
    "si_sdr_improvement",
    "noisy_stoi",
    "enhanced_stoi",
    "stoi_improvement"
]


with open(
    RESULTS_FILE,
    "w",
    newline=""
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================
# PRINT RESULTS
# ============================================

print("\n" + "=" * 70)
print("INDIVIDUAL RESULTS")
print("=" * 70)

for result in results:

    print(
        f"\n{result['file']}"
    )

    print(
        f"  SNR: "
        f"{result['noisy_snr']:.3f} → "
        f"{result['enhanced_snr']:.3f} "
        f"({result['snr_improvement']:+.3f} dB)"
    )

    print(
        f"  SI-SDR: "
        f"{result['noisy_si_sdr']:.3f} → "
        f"{result['enhanced_si_sdr']:.3f} "
        f"({result['si_sdr_improvement']:+.3f} dB)"
    )

    print(
        f"  STOI: "
        f"{result['noisy_stoi']:.3f} → "
        f"{result['enhanced_stoi']:.3f} "
        f"({result['stoi_improvement']:+.3f})"
    )


# ============================================
# AVERAGES
# ============================================

if results:

    average_snr = np.mean([
        r["snr_improvement"]
        for r in results
    ])

    average_si_sdr = np.mean([
        r["si_sdr_improvement"]
        for r in results
    ])

    average_stoi = np.mean([
        r["stoi_improvement"]
        for r in results
    ])


    print(
        "\n" + "=" * 70
    )

    print(
        "AVERAGE IMPROVEMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAverage SNR improvement: "
        f"{average_snr:+.4f} dB"
    )

    print(
        f"Average SI-SDR improvement: "
        f"{average_si_sdr:+.4f} dB"
    )

    print(
        f"Average STOI improvement: "
        f"{average_stoi:+.4f}"
    )


print(
    "\nResults saved to:"
)

print(
    RESULTS_FILE
)

print(
    "\n" + "=" * 70
)
print(
    "EVALUATION COMPLETE"
)
print(
    "=" * 70
)
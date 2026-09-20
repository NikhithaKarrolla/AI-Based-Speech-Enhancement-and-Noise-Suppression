import os

import numpy as np
import pandas as pd
import librosa

#from src.audio.mixing import calculate_snr

def calculate_snr(clean, noise):
    """
    Calculate Signal-to-Noise Ratio in dB.
    """

    clean = np.asarray(
        clean,
        dtype=np.float64
    )

    noise = np.asarray(
        noise,
        dtype=np.float64
    )

    signal_power = np.sum(
        clean ** 2
    )

    noise_power = np.sum(
        noise ** 2
    )

    if noise_power < 1e-12:
        return 100.0

    return 10 * np.log10(
        signal_power /
        noise_power
    )
# ============================================
# Configuration
# ============================================

TEST_NOISY_DIR = "data/test/noisy"
TEST_CLEAN_DIR = "data/test/clean"

ENHANCED_DIR = "outputs/audio"

OUTPUT_CSV = "outputs/metrics/unet_test_results.csv"

SAMPLE_RATE = 16000


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

    reference_energy = np.sum(
        reference ** 2
    )

    if reference_energy < 1e-12:
        return 0.0

    scale = np.dot(
        estimate,
        reference
    ) / reference_energy

    target = scale * reference

    noise = estimate - target

    target_energy = np.sum(
        target ** 2
    )

    noise_energy = np.sum(
        noise ** 2
    )

    if noise_energy < 1e-12:
        return 100.0

    return 10 * np.log10(
        target_energy /
        noise_energy
    )


# ============================================
# STOI
# ============================================

def calculate_stoi(
    clean,
    enhanced
):

    try:

        from pystoi import stoi

        return stoi(
            clean,
            enhanced,
            SAMPLE_RATE,
            extended=False
        )

    except Exception as error:

        print(
            f"STOI calculation failed: {error}"
        )

        return np.nan


# ============================================
# Main
# ============================================

def main():

    print("=" * 70)
    print("U-NET TEST SET EVALUATION")
    print("=" * 70)

    noisy_files = sorted(
        file_name
        for file_name in os.listdir(
            TEST_NOISY_DIR
        )
        if file_name.lower().endswith(".wav")
    )

    print(
        f"\nNumber of noisy test samples:"
    )

    print(len(noisy_files))

    results = []

    print("\n" + "=" * 70)
    print("INDIVIDUAL RESULTS")
    print("=" * 70)

    for noisy_file in noisy_files:

        # --------------------------------
        # Determine base name
        # --------------------------------

        base_name = noisy_file.split(
            "_snr_"
        )[0]

        snr_part = noisy_file.split(
            "_snr_"
        )[1]

        clean_file = (
            f"{base_name}_clean.wav"
        )

        enhanced_file = (
            f"unet_enhanced_"
            f"{base_name}_"
            f"{snr_part}"
        )

        noisy_path = os.path.join(
            TEST_NOISY_DIR,
            noisy_file
        )

        clean_path = os.path.join(
            TEST_CLEAN_DIR,
            clean_file
        )

        enhanced_path = os.path.join(
            ENHANCED_DIR,
            enhanced_file
        )

        # --------------------------------
        # Check files
        # --------------------------------

        if not os.path.exists(
            clean_path
        ):

            print(
                f"\nMissing clean file: "
                f"{clean_file}"
            )

            continue

        if not os.path.exists(
            enhanced_path
        ):

            print(
                f"\nMissing enhanced file: "
                f"{enhanced_file}"
            )

            continue

        # --------------------------------
        # Load audio
        # --------------------------------

        noisy, _ = librosa.load(
            noisy_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        clean, _ = librosa.load(
            clean_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        enhanced, _ = librosa.load(
            enhanced_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # --------------------------------
        # Match lengths
        # --------------------------------

        min_length = min(
            len(clean),
            len(noisy),
            len(enhanced)
        )

        clean = clean[:min_length]

        noisy = noisy[:min_length]

        enhanced = enhanced[:min_length]

        # --------------------------------
        # SNR
        # --------------------------------

        noisy_snr = calculate_snr(
            clean,
            noisy - clean
        )

        enhanced_snr = calculate_snr(
            clean,
            enhanced - clean
        )

        snr_improvement = (
            enhanced_snr -
            noisy_snr
        )

        # --------------------------------
        # SI-SDR
        # --------------------------------

        noisy_si_sdr = calculate_si_sdr(
            clean,
            noisy
        )

        enhanced_si_sdr = calculate_si_sdr(
            clean,
            enhanced
        )

        si_sdr_improvement = (
            enhanced_si_sdr -
            noisy_si_sdr
        )

        # --------------------------------
        # STOI
        # --------------------------------

        noisy_stoi = calculate_stoi(
            clean,
            noisy
        )

        enhanced_stoi = calculate_stoi(
            clean,
            enhanced
        )

        stoi_improvement = (
            enhanced_stoi -
            noisy_stoi
        )

        # --------------------------------
        # Store result
        # --------------------------------

        results.append(
            {
                "file": noisy_file,

                "noisy_snr": noisy_snr,
                "enhanced_snr": enhanced_snr,
                "snr_improvement": snr_improvement,

                "noisy_si_sdr": noisy_si_sdr,
                "enhanced_si_sdr": enhanced_si_sdr,
                "si_sdr_improvement": si_sdr_improvement,

                "noisy_stoi": noisy_stoi,
                "enhanced_stoi": enhanced_stoi,
                "stoi_improvement": stoi_improvement
            }
        )

        # --------------------------------
        # Print result
        # --------------------------------

        print(f"\n{noisy_file}")

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

    # ========================================
    # DataFrame
    # ========================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================
    # Average improvements
    # ========================================

    print("\n" + "=" * 70)
    print("AVERAGE IMPROVEMENT")
    print("=" * 70)

    average_snr = (
        results_df[
            "snr_improvement"
        ].mean()
    )

    average_si_sdr = (
        results_df[
            "si_sdr_improvement"
        ].mean()
    )

    average_stoi = (
        results_df[
            "stoi_improvement"
        ].mean()
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

    # ========================================
    # Save
    # ========================================

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nResults saved to:"
    )

    print(OUTPUT_CSV)

    print("\n" + "=" * 70)
    print("U-NET EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
import os

import pandas as pd


CNN_RESULTS = "outputs/metrics/cnn_test_results.csv"
UNET_RESULTS = "outputs/metrics/unet_test_results.csv"

OUTPUT_CSV = "outputs/metrics/cnn_vs_unet_comparison.csv"


def extract_snr(file_name):
    """
    Extract the target SNR from filenames such as:

    clean_0002_snr_-5dB.wav
    clean_0002_snr_0dB.wav
    clean_0002_snr_5dB.wav
    clean_0002_snr_10dB.wav
    """

    snr_part = file_name.split("_snr_")[1]

    snr_value = snr_part.replace(
        "dB.wav",
        ""
    )

    return float(snr_value)


def main():

    print("=" * 70)
    print("CNN vs U-NET COMPARISON")
    print("=" * 70)

    # --------------------------------
    # Load results
    # --------------------------------

    cnn = pd.read_csv(
        CNN_RESULTS
    )

    unet = pd.read_csv(
        UNET_RESULTS
    )

    # --------------------------------
    # Extract target SNR
    # --------------------------------

    cnn["target_snr"] = cnn["file"].apply(
        extract_snr
    )

    unet["target_snr"] = unet["file"].apply(
        extract_snr
    )

    # --------------------------------
    # Average by target SNR
    # --------------------------------

    cnn_grouped = (
        cnn
        .groupby("target_snr")
        [
            [
                "snr_improvement",
                "si_sdr_improvement",
                "stoi_improvement"
            ]
        ]
        .mean()
        .reset_index()
    )

    unet_grouped = (
        unet
        .groupby("target_snr")
        [
            [
                "snr_improvement",
                "si_sdr_improvement",
                "stoi_improvement"
            ]
        ]
        .mean()
        .reset_index()
    )

    # --------------------------------
    # Rename columns
    # --------------------------------

    cnn_grouped = cnn_grouped.rename(
        columns={
            "snr_improvement":
                "cnn_snr_improvement",

            "si_sdr_improvement":
                "cnn_si_sdr_improvement",

            "stoi_improvement":
                "cnn_stoi_improvement"
        }
    )

    unet_grouped = unet_grouped.rename(
        columns={
            "snr_improvement":
                "unet_snr_improvement",

            "si_sdr_improvement":
                "unet_si_sdr_improvement",

            "stoi_improvement":
                "unet_stoi_improvement"
        }
    )

    # --------------------------------
    # Merge
    # --------------------------------

    comparison = pd.merge(
        cnn_grouped,
        unet_grouped,
        on="target_snr"
    )

    # --------------------------------
    # Calculate U-Net advantage
    # --------------------------------

    comparison[
        "unet_snr_advantage"
    ] = (
        comparison["unet_snr_improvement"]
        -
        comparison["cnn_snr_improvement"]
    )

    comparison[
        "unet_si_sdr_advantage"
    ] = (
        comparison["unet_si_sdr_improvement"]
        -
        comparison["cnn_si_sdr_improvement"]
    )

    comparison[
        "unet_stoi_advantage"
    ] = (
        comparison["unet_stoi_improvement"]
        -
        comparison["cnn_stoi_improvement"]
    )

    # --------------------------------
    # Print
    # --------------------------------

    print("\nPER-SNR COMPARISON")
    print("=" * 70)

    for _, row in comparison.iterrows():

        snr = row["target_snr"]

        print(
            f"\nInput SNR: {snr:+.0f} dB"
        )

        print(
            f"  CNN SNR improvement: "
            f"{row['cnn_snr_improvement']:+.3f} dB"
        )

        print(
            f"  U-Net SNR improvement: "
            f"{row['unet_snr_improvement']:+.3f} dB"
        )

        print(
            f"  U-Net SNR advantage: "
            f"{row['unet_snr_advantage']:+.3f} dB"
        )

        print(
            f"  CNN SI-SDR improvement: "
            f"{row['cnn_si_sdr_improvement']:+.3f} dB"
        )

        print(
            f"  U-Net SI-SDR improvement: "
            f"{row['unet_si_sdr_improvement']:+.3f} dB"
        )

        print(
            f"  U-Net SI-SDR advantage: "
            f"{row['unet_si_sdr_advantage']:+.3f} dB"
        )

        print(
            f"  CNN STOI improvement: "
            f"{row['cnn_stoi_improvement']:+.3f}"
        )

        print(
            f"  U-Net STOI improvement: "
            f"{row['unet_stoi_improvement']:+.3f}"
        )

        print(
            f"  U-Net STOI advantage: "
            f"{row['unet_stoi_advantage']:+.3f}"
        )

    # --------------------------------
    # Save
    # --------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    comparison.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        f"Comparison saved to:\n"
        f"{OUTPUT_CSV}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
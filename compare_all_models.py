import pandas as pd
import numpy as np
import os


# ============================================
# Configuration
# ============================================

CNN_CSV = (
    "outputs/metrics/cnn_test_results.csv"
)

UNET_CSV = (
    "outputs/metrics/unet_test_results.csv"
)

MASK_UNET_CSV = (
    "outputs/metrics/mask_unet_test_results.csv"
)

OUTPUT_CSV = (
    "outputs/metrics/all_models_comparison.csv"
)


# ============================================
# Load results
# ============================================

cnn = pd.read_csv(CNN_CSV)

unet = pd.read_csv(UNET_CSV)

mask_unet = pd.read_csv(MASK_UNET_CSV)


# ============================================
# Extract input SNR
# ============================================

def extract_snr(filename):

    part = filename.split("_snr_")[1]

    value = part.replace(
        "dB.wav",
        ""
    )

    return int(value)


cnn["input_snr"] = (
    cnn["file"]
    .apply(extract_snr)
)

unet["input_snr"] = (
    unet["file"]
    .apply(extract_snr)
)

mask_unet["input_snr"] = (
    mask_unet["file"]
    .apply(extract_snr)
)


# ============================================
# Group by input SNR
# ============================================

cnn_grouped = (
    cnn
    .groupby("input_snr")
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
    .groupby("input_snr")
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

mask_grouped = (
    mask_unet
    .groupby("input_snr")
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


# ============================================
# Rename columns
# ============================================

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

mask_grouped = mask_grouped.rename(
    columns={
        "snr_improvement":
            "mask_unet_snr_improvement",

        "si_sdr_improvement":
            "mask_unet_si_sdr_improvement",

        "stoi_improvement":
            "mask_unet_stoi_improvement"
    }
)


# ============================================
# Merge
# ============================================

comparison = cnn_grouped.merge(
    unet_grouped,
    on="input_snr"
)

comparison = comparison.merge(
    mask_grouped,
    on="input_snr"
)


# ============================================
# Sort
# ============================================

comparison = comparison.sort_values(
    "input_snr"
)


# ============================================
# Save
# ============================================

os.makedirs(
    os.path.dirname(
        OUTPUT_CSV
    ),
    exist_ok=True
)

comparison.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================
# Print comparison
# ============================================

print()
print("========================================")
print("CNN vs U-Net vs Mask U-Net")
print("========================================")

for _, row in comparison.iterrows():

    snr = int(
        row["input_snr"]
    )

    print()
    print(
        f"Input SNR: {snr:+d} dB"
    )

    print()

    print(
        "CNN:"
    )

    print(
        f"  SNR improvement: "
        f"{row['cnn_snr_improvement']:+.3f} dB"
    )

    print(
        f"  SI-SDR improvement: "
        f"{row['cnn_si_sdr_improvement']:+.3f} dB"
    )

    print(
        f"  STOI improvement: "
        f"{row['cnn_stoi_improvement']:+.3f}"
    )

    print()

    print(
        "U-Net:"
    )

    print(
        f"  SNR improvement: "
        f"{row['unet_snr_improvement']:+.3f} dB"
    )

    print(
        f"  SI-SDR improvement: "
        f"{row['unet_si_sdr_improvement']:+.3f} dB"
    )

    print(
        f"  STOI improvement: "
        f"{row['unet_stoi_improvement']:+.3f}"
    )

    print()

    print(
        "Mask U-Net:"
    )

    print(
        f"  SNR improvement: "
        f"{row['mask_unet_snr_improvement']:+.3f} dB"
    )

    print(
        f"  SI-SDR improvement: "
        f"{row['mask_unet_si_sdr_improvement']:+.3f} dB"
    )

    print(
        f"  STOI improvement: "
        f"{row['mask_unet_stoi_improvement']:+.3f}"
    )


# ============================================
# Overall averages
# ============================================

print()
print("========================================")
print("OVERALL AVERAGES")
print("========================================")

print()

print(
    "CNN:"
)

print(
    f"  SNR: "
    f"{cnn['snr_improvement'].mean():+.4f} dB"
)

print(
    f"  SI-SDR: "
    f"{cnn['si_sdr_improvement'].mean():+.4f} dB"
)

print(
    f"  STOI: "
    f"{cnn['stoi_improvement'].mean():+.4f}"
)

print()

print(
    "U-Net:"
)

print(
    f"  SNR: "
    f"{unet['snr_improvement'].mean():+.4f} dB"
)

print(
    f"  SI-SDR: "
    f"{unet['si_sdr_improvement'].mean():+.4f} dB"
)

print(
    f"  STOI: "
    f"{unet['stoi_improvement'].mean():+.4f}"
)

print()

print(
    "Mask U-Net:"
)

print(
    f"  SNR: "
    f"{mask_unet['snr_improvement'].mean():+.4f} dB"
)

print(
    f"  SI-SDR: "
    f"{mask_unet['si_sdr_improvement'].mean():+.4f} dB"
)

print(
    f"  STOI: "
    f"{mask_unet['stoi_improvement'].mean():+.4f}"
)

print()
print("========================================")

print(
    "Comparison saved to:",
    OUTPUT_CSV
)
import os
import random

import numpy as np
import librosa

from src.data.mixing import (
    mix_audio,
    calculate_snr,
    save_audio
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

CLEAN_DIR = os.path.join(
    "data",
    "raw",
    "clean"
)

NOISE_DIR = os.path.join(
    "data",
    "raw",
    "noise"
)

OUTPUT_DIR = os.path.join(
    "data",
    "processed",
    "train"
)

METADATA_DIR = os.path.join(
    "data",
    "metadata"
)

SNR_LEVELS = [
    -5,
    0,
    5,
    10
]

RANDOM_SEED = 42

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    METADATA_DIR,
    exist_ok=True
)


# ============================================================
# FIND CLEAN SPEECH FILES
# ============================================================

clean_files = [
    file
    for file in os.listdir(CLEAN_DIR)
    if file.lower().endswith(".wav")
]


if len(clean_files) == 0:

    raise FileNotFoundError(
        "No clean WAV files found in "
        f"{CLEAN_DIR}"
    )


# ============================================================
# FIND NOISE FILES
# ============================================================

noise_files = [
    file
    for file in os.listdir(NOISE_DIR)
    if file.lower().endswith(".wav")
]


if len(noise_files) == 0:

    raise FileNotFoundError(
        "No noise WAV files found in "
        f"{NOISE_DIR}"
    )


print()
print("=" * 60)
print("NOISY DATASET GENERATION")
print("=" * 60)

print(
    f"Clean files: {len(clean_files)}"
)

print(
    f"Noise files: {len(noise_files)}"
)

print(
    f"SNR levels: {SNR_LEVELS}"
)

print()


# ============================================================
# METADATA STORAGE
# ============================================================

metadata = []


sample_counter = 1


# ============================================================
# GENERATE MIXTURES
# ============================================================

for clean_file in clean_files:

    clean_path = os.path.join(
        CLEAN_DIR,
        clean_file
    )

    # Load clean speech
    clean_audio, sr = librosa.load(
        clean_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    clean_audio = (
        clean_audio.astype(
            np.float32
        )
    )


    for snr_db in SNR_LEVELS:

        # Select random noise
        noise_file = random.choice(
            noise_files
        )

        noise_path = os.path.join(
            NOISE_DIR,
            noise_file
        )


        noise_audio, _ = librosa.load(
            noise_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        noise_audio = (
            noise_audio.astype(
                np.float32
            )
        )


        # ================================================
        # MIX
        # ================================================

        noisy_audio, scaled_noise = mix_audio(
            clean_audio,
            noise_audio,
            snr_db
        )


        # ================================================
        # CALCULATE ACTUAL SNR
        # ================================================

        actual_snr = calculate_snr(
            clean_audio,
            scaled_noise
        )


        # ================================================
        # FILE NAME
        # ================================================

        sample_name = (
            f"sample_{sample_counter:04d}"
        )


        noisy_filename = (
            f"{sample_name}_"
            f"snr_{snr_db}dB.wav"
        )


        clean_filename = (
            f"{sample_name}_"
            f"clean.wav"
        )


        noisy_path = os.path.join(
            OUTPUT_DIR,
            noisy_filename
        )


        clean_output_path = os.path.join(
            OUTPUT_DIR,
            clean_filename
        )


        # ================================================
        # SAVE
        # ================================================

        save_audio(
            noisy_path,
            noisy_audio,
            SAMPLE_RATE
        )


        save_audio(
            clean_output_path,
            clean_audio,
            SAMPLE_RATE
        )


        # ================================================
        # METADATA
        # ================================================

        metadata.append({
            "sample_id": sample_name,
            "clean_file": clean_filename,
            "noisy_file": noisy_filename,
            "noise_file": noise_file,
            "target_snr_db": snr_db,
            "actual_snr_db": actual_snr,
            "sample_rate": SAMPLE_RATE
        })


        print(
            f"Created {sample_name} | "
            f"Target SNR: {snr_db} dB | "
            f"Actual SNR: {actual_snr:.2f} dB | "
            f"Noise: {noise_file}"
        )


        sample_counter += 1


# ============================================================
# SAVE METADATA
# ============================================================

import pandas as pd


metadata_df = pd.DataFrame(
    metadata
)


metadata_path = os.path.join(
    METADATA_DIR,
    "training_metadata.csv"
)


metadata_df.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATASET GENERATION COMPLETED")
print("=" * 60)

print(
    f"Total samples created: "
    f"{len(metadata)}"
)

print(
    f"Output directory: "
    f"{OUTPUT_DIR}"
)

print(
    f"Metadata: "
    f"{metadata_path}"
)
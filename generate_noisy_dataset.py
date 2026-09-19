import os
import random
import numpy as np
import pandas as pd

from src.audio.preprocessing import load_audio
from src.data.mixing import mix_audio, peak_normalize, calculate_snr


# ============================================
# CONFIGURATION
# ============================================

CLEAN_DIR = "data/raw/clean"
NOISE_DIR = "data/raw/noise"

OUTPUT_DIR = "data"

SAMPLE_RATE = 16000

SNR_LEVELS = [-5, 0, 5, 10]

RANDOM_SEED = 42


# ============================================
# RANDOM SEED
# ============================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================
# CREATE DIRECTORIES
# ============================================

for split in ["train", "validation", "test"]:

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            split,
            "clean"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            split,
            "noisy"
        ),
        exist_ok=True
    )


# ============================================
# FIND CLEAN FILES
# ============================================

clean_files = [
    file
    for file in os.listdir(CLEAN_DIR)
    if file.lower().endswith(".wav")
]

clean_files.sort()

if len(clean_files) == 0:
    raise ValueError(
        "No clean WAV files found."
    )


# ============================================
# SHUFFLE
# ============================================

random.shuffle(clean_files)


# ============================================
# SPLIT
# ============================================

total = len(clean_files)

train_count = int(total * 0.8)
validation_count = int(total * 0.1)

train_files = clean_files[:train_count]

validation_files = clean_files[
    train_count:
    train_count + validation_count
]

test_files = clean_files[
    train_count + validation_count:
]


splits = {
    "train": train_files,
    "validation": validation_files,
    "test": test_files
}


print("\nDataset split:")

for split, files in splits.items():
    print(
        f"{split}: {len(files)} files"
    )


# ============================================
# FIND NOISE FILES
# ============================================

noise_files = [
    file
    for file in os.listdir(NOISE_DIR)
    if file.lower().endswith(".wav")
]

noise_files.sort()

if len(noise_files) == 0:
    raise ValueError(
        "No noise WAV files found."
    )


# ============================================
# METADATA
# ============================================

metadata = []


# ============================================
# GENERATE DATASET
# ============================================

for split, files in splits.items():

    print(
        f"\nGenerating {split} dataset..."
    )

    for clean_filename in files:

        clean_path = os.path.join(
            CLEAN_DIR,
            clean_filename
        )

        clean_audio, sr = load_audio(
            clean_path,
            SAMPLE_RATE
        )

        # Remove extension
        base_name = os.path.splitext(
            clean_filename
        )[0]

        # Save clean reference
        clean_output = os.path.join(
            OUTPUT_DIR,
            split,
            "clean",
            f"{base_name}_clean.wav"
        )

        clean_audio = peak_normalize(
            clean_audio
        )

        import soundfile as sf

        sf.write(
            clean_output,
            clean_audio,
            sr
        )

        # Generate noisy versions
        for snr in SNR_LEVELS:

            noise_filename = random.choice(
                noise_files
            )

            noise_path = os.path.join(
                NOISE_DIR,
                noise_filename
            )

            noise_audio, _ = load_audio(
                noise_path,
                SAMPLE_RATE
            )

            noisy_audio, scaled_noise = mix_audio(
                clean_audio,
                noise_audio,
                snr
            )

            actual_snr = calculate_snr(
                clean_audio,
                scaled_noise
            )

            noisy_filename = (
                f"{base_name}_snr_{snr}dB.wav"
            )

            noisy_output = os.path.join(
                OUTPUT_DIR,
                split,
                "noisy",
                noisy_filename
            )

            sf.write(
                noisy_output,
                noisy_audio,
                sr
            )

            metadata.append({
                "split": split,
                "clean_file": clean_output,
                "noisy_file": noisy_output,
                "noise_file": noise_filename,
                "target_snr_db": snr,
                "actual_snr_db": actual_snr
            })

            print(
                f"{split}: "
                f"{base_name} | "
                f"SNR={snr} dB | "
                f"actual={actual_snr:.2f} dB"
            )


# ============================================
# SAVE METADATA
# ============================================

os.makedirs(
    "data/metadata",
    exist_ok=True
)

metadata_path = (
    "data/metadata/"
    "training_metadata.csv"
)

df = pd.DataFrame(metadata)

df.to_csv(
    metadata_path,
    index=False
)


print("\n================================")
print("Dataset generation completed.")
print("================================")

print(
    f"Metadata saved to: {metadata_path}"
)
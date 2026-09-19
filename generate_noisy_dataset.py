import os
import random

import librosa
import soundfile as sf

from configs.config import (
    SAMPLE_RATE,
    SNR_LEVELS,
    RANDOM_SEED
)

from src.data.mixing import (
    calculate_snr,
    mix_audio
)


# ============================================
# DIRECTORIES
# ============================================

RAW_CLEAN_DIR = "data/raw/clean"
RAW_NOISE_DIR = "data/raw/noise"


# ============================================
# DATASET SPLITS
# ============================================

SPLITS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1
}


# ============================================
# RANDOM SEED
# ============================================

random.seed(RANDOM_SEED)


# ============================================
# NOISE FILES
# ============================================

NOISE_FILES = [
    "white_noise.wav",
    "low_frequency_noise.wav",
    "mixed_noise.wav"
]


# ============================================
# FIND CLEAN RECORDINGS
# ============================================

clean_files = [
    file
    for file in os.listdir(RAW_CLEAN_DIR)
    if file.lower().endswith(".wav")
]

clean_files.sort()


print("=" * 60)
print("DATASET GENERATION")
print("=" * 60)

print("\nTotal clean recordings:")
print(len(clean_files))

for file in clean_files:
    print(" ", file)


# ============================================
# RECORDING-LEVEL SPLIT
# ============================================

random.shuffle(clean_files)

total_files = len(clean_files)

train_count = int(total_files * 0.8)
validation_count = int(total_files * 0.1)

# Make sure validation has at least one recording
if validation_count < 1:
    validation_count = 1

test_count = total_files - train_count - validation_count

# Make sure test has at least one recording
if test_count < 1:
    test_count = 1
    train_count = total_files - validation_count - test_count


train_files = clean_files[:train_count]

validation_files = clean_files[
    train_count:
    train_count + validation_count
]

test_files = clean_files[
    train_count + validation_count:
]


# ============================================
# DISPLAY SPLIT
# ============================================

print("\n" + "=" * 60)
print("RECORDING-LEVEL SPLIT")
print("=" * 60)

print("\nTrain recordings:")
for file in train_files:
    print(" ", file)

print("\nValidation recordings:")
for file in validation_files:
    print(" ", file)

print("\nTest recordings:")
for file in test_files:
    print(" ", file)


# ============================================
# CREATE DIRECTORIES
# ============================================

for split in SPLITS:

    os.makedirs(
        f"data/{split}/clean",
        exist_ok=True
    )

    os.makedirs(
        f"data/{split}/noisy",
        exist_ok=True
    )


# ============================================
# CLEAR OLD GENERATED DATA
# ============================================

for split in SPLITS:

    clean_dir = f"data/{split}/clean"
    noisy_dir = f"data/{split}/noisy"

    for file in os.listdir(clean_dir):

        if file.lower().endswith(".wav"):

            os.remove(
                os.path.join(clean_dir, file)
            )

    for file in os.listdir(noisy_dir):

        if file.lower().endswith(".wav"):

            os.remove(
                os.path.join(noisy_dir, file)
            )


# ============================================
# LOAD NOISE FILES
# ============================================

noise_data = {}

for noise_file in NOISE_FILES:

    noise_path = os.path.join(
        RAW_NOISE_DIR,
        noise_file
    )

    noise, _ = librosa.load(
        noise_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    noise_data[noise_file] = noise


# ============================================
# GENERATE DATA FOR ONE SPLIT
# ============================================

def generate_split(split_name, files):

    clean_output_dir = f"data/{split_name}/clean"
    noisy_output_dir = f"data/{split_name}/noisy"

    sample_counter = 0

    for clean_file in files:

        clean_path = os.path.join(
            RAW_CLEAN_DIR,
            clean_file
        )

        # Load clean speech
        clean_audio, _ = librosa.load(
            clean_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        base_name = os.path.splitext(
            clean_file
        )[0]

        # ====================================
        # CREATE SAMPLES FOR EACH SNR
        # ====================================

        for snr in SNR_LEVELS:

            # Randomly select noise type
            noise_name = random.choice(
                NOISE_FILES
            )

            noise = noise_data[noise_name]

            # Create noisy speech
            noisy_audio = mix_audio(
                clean_audio,
                noise,
                snr
            )

            # Output filenames
            clean_output_name = (
                f"{base_name}_clean.wav"
            )

            noisy_output_name = (
                f"{base_name}_snr_{snr}dB.wav"
            )

            clean_output_path = os.path.join(
                clean_output_dir,
                clean_output_name
            )

            noisy_output_path = os.path.join(
                noisy_output_dir,
                noisy_output_name
            )

            # Save clean recording once
            if not os.path.exists(
                clean_output_path
            ):

                sf.write(
                    clean_output_path,
                    clean_audio,
                    SAMPLE_RATE
                )

            # Save noisy recording
            sf.write(
                noisy_output_path,
                noisy_audio,
                SAMPLE_RATE
            )

            # Calculate actual SNR
            actual_snr = calculate_snr(
                clean_audio,
                noisy_audio - clean_audio
            )

            print(
                f"{split_name:10s} | "
                f"{base_name:20s} | "
                f"SNR={snr:3d} dB | "
                f"Actual={actual_snr:6.2f} dB | "
                f"Noise={noise_name}"
            )

            sample_counter += 1

    return sample_counter


# ============================================
# GENERATE ALL SPLITS
# ============================================

train_samples = generate_split(
    "train",
    train_files
)

validation_samples = generate_split(
    "validation",
    validation_files
)

test_samples = generate_split(
    "test",
    test_files
)


# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)

print("\nTrain noisy samples:")
print(train_samples)

print("\nValidation noisy samples:")
print(validation_samples)

print("\nTest noisy samples:")
print(test_samples)
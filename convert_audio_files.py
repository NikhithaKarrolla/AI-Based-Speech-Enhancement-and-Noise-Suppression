import os
import subprocess
import tempfile
import shutil

import imageio_ffmpeg


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CLEAN_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "clean"
)

NOISE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "noise"
)


# ============================================================
# FFMPEG
# ============================================================

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================
# AUDIO CONVERSION
# ============================================================

def convert_audio(input_file, output_file):
    """
    Convert audio to:

    Sample rate : 16000 Hz
    Channels    : Mono
    Format      : PCM 16-bit WAV
    """

    command = [
        FFMPEG,
        "-y",
        "-i",
        input_file,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        output_file
    ]

    subprocess.run(
        command,
        check=True
    )


# ============================================================
# STANDARDIZE EXISTING CLEAN WAV FILES
# ============================================================

def standardize_existing_wavs():

    wav_files = [
        file
        for file in os.listdir(CLEAN_DIR)
        if file.lower().endswith(".wav")
    ]

    print("=" * 60)
    print("STANDARDIZING EXISTING CLEAN WAV FILES")
    print("=" * 60)

    print(f"\nFound {len(wav_files)} WAV files.")

    for wav_file in wav_files:

        input_path = os.path.join(
            CLEAN_DIR,
            wav_file
        )

        # Temporary file
        temp_path = os.path.join(
            CLEAN_DIR,
            "_temp_" + wav_file
        )

        print(f"\nConverting: {wav_file}")

        convert_audio(
            input_path,
            temp_path
        )

        # Replace original with standardized version
        os.remove(input_path)

        os.replace(
            temp_path,
            input_path
        )

        print("  -> 16000 Hz mono WAV")


# ============================================================
# CONVERT NEW SPEECH MP3 FILES
# ============================================================

def convert_speech_mp3_files():

    mp3_files = [
        file
        for file in os.listdir(CLEAN_DIR)
        if file.lower().endswith(".mp3")
    ]

    print("\n" + "=" * 60)
    print("CONVERTING SPEECH MP3 FILES")
    print("=" * 60)

    print(f"\nFound {len(mp3_files)} MP3 files.")

    # Find existing clean numbers
    existing_numbers = []

    for file in os.listdir(CLEAN_DIR):

        if file.startswith("clean_") and file.endswith(".wav"):

            number_text = (
                file
                .replace("clean_", "")
                .replace(".wav", "")
            )

            try:
                existing_numbers.append(
                    int(number_text)
                )
            except ValueError:
                pass

    if existing_numbers:
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1

    for mp3_file in mp3_files:

        input_path = os.path.join(
            CLEAN_DIR,
            mp3_file
        )

        output_name = f"clean_{next_number:04d}.wav"

        output_path = os.path.join(
            CLEAN_DIR,
            output_name
        )

        print(f"\nConverting: {mp3_file}")
        print(f"  -> {output_name}")

        convert_audio(
            input_path,
            output_path
        )

        next_number += 1


# ============================================================
# CONVERT CROWD NOISE
# ============================================================

def convert_crowd_noise():

    crowd_noise_file = "soundreality-crowd-noise-375725.mp3"

    input_path = os.path.join(
        NOISE_DIR,
        crowd_noise_file
    )

    if not os.path.exists(input_path):

        print("\nCrowd noise MP3 not found.")

        return

    output_path = os.path.join(
        NOISE_DIR,
        "crowd_noise.wav"
    )

    print("\n" + "=" * 60)
    print("CONVERTING CROWD NOISE")
    print("=" * 60)

    convert_audio(
        input_path,
        output_path
    )

    print("\nCreated:")
    print("crowd_noise.wav")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # First convert any MP3 speech files
    convert_speech_mp3_files()

    # Then standardize ALL WAV speech files
    standardize_existing_wavs()

    # Convert real-world crowd noise
    convert_crowd_noise()

    print("\n" + "=" * 60)
    print("AUDIO STANDARDIZATION COMPLETE")
    print("=" * 60)
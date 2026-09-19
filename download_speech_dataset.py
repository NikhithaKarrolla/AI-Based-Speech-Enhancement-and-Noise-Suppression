import os
import shutil

import librosa


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = os.path.join(
    "data",
    "raw",
    "clean"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LIBROSA EXAMPLE SPEECH FILES
# ============================================================

example_names = [
    "libri1",
    "libri2",
    "libri3"
]


# ============================================================
# DOWNLOAD / COPY
# ============================================================

for index, example_name in enumerate(
    example_names,
    start=1
):

    print(
        f"Getting speech sample: "
        f"{example_name}"
    )

    source_path = librosa.example(
        example_name
    )


    output_path = os.path.join(
        OUTPUT_DIR,
        f"clean_{index:04d}.wav"
    )


    shutil.copy2(
        source_path,
        output_path
    )


    print(
        f"Saved: {output_path}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("CLEAN SPEECH DATASET CREATED")
print("=" * 60)

files = [
    file
    for file in os.listdir(OUTPUT_DIR)
    if file.endswith(".wav")
]

for file in files:

    print(file)
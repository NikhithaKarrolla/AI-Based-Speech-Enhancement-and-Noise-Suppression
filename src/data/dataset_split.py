import os
import random
import shutil


def split_files(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split audio files into train, validation,
    and test directories.

    The split is performed at the file level.
    """

    # --------------------------------------------
    # Validate ratios
    # --------------------------------------------

    total_ratio = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if abs(total_ratio - 1.0) > 1e-6:

        raise ValueError(
            "Train, validation and test "
            "ratios must sum to 1."
        )


    # --------------------------------------------
    # Get WAV files
    # --------------------------------------------

    files = [
        file
        for file in os.listdir(input_dir)
        if file.lower().endswith(".wav")
    ]


    if len(files) == 0:

        raise ValueError(
            f"No WAV files found in {input_dir}"
        )


    # --------------------------------------------
    # Reproducible shuffle
    # --------------------------------------------

    random.seed(seed)

    random.shuffle(files)


    # --------------------------------------------
    # Calculate split sizes
    # --------------------------------------------

    total_files = len(files)

    train_count = int(
        total_files * train_ratio
    )

    validation_count = int(
        total_files * validation_ratio
    )


    # --------------------------------------------
    # Split
    # --------------------------------------------

    train_files = files[
        :train_count
    ]

    validation_files = files[
        train_count:
        train_count + validation_count
    ]

    test_files = files[
        train_count + validation_count:
    ]


    # --------------------------------------------
    # Create directories
    # --------------------------------------------

    train_dir = os.path.join(
        output_dir,
        "train"
    )

    validation_dir = os.path.join(
        output_dir,
        "validation"
    )

    test_dir = os.path.join(
        output_dir,
        "test"
    )


    os.makedirs(
        train_dir,
        exist_ok=True
    )

    os.makedirs(
        validation_dir,
        exist_ok=True
    )

    os.makedirs(
        test_dir,
        exist_ok=True
    )


    # --------------------------------------------
    # Copy files
    # --------------------------------------------

    for file in train_files:

        shutil.copy2(
            os.path.join(input_dir, file),
            os.path.join(train_dir, file)
        )


    for file in validation_files:

        shutil.copy2(
            os.path.join(input_dir, file),
            os.path.join(validation_dir, file)
        )


    for file in test_files:

        shutil.copy2(
            os.path.join(input_dir, file),
            os.path.join(test_dir, file)
        )


    # --------------------------------------------
    # Return information
    # --------------------------------------------

    return {
        "train": train_files,
        "validation": validation_files,
        "test": test_files
    }
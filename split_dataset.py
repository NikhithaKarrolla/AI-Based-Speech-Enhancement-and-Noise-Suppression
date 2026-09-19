import os

from src.data.dataset_split import split_files


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "clean"
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "splits"
)


# ============================================================
# SPLIT DATASET
# ============================================================

result = split_files(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1,
    seed=42
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print(
    f"Training files: "
    f"{len(result['train'])}"
)

print(
    f"Validation files: "
    f"{len(result['validation'])}"
)

print(
    f"Test files: "
    f"{len(result['test'])}"
)

print()
print("Dataset split completed.")
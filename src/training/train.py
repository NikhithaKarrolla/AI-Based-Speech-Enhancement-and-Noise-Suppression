import os

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.spectrogram_dataset import (
    ChunkedSpeechEnhancementDataset
)

from src.models.cnn_baseline import (
    SpeechEnhancementCNN
)

from src.training.loss import (
    SpectrogramMSELoss
)


# ============================================
# CONFIGURATION
# ============================================

TRAIN_DIR = "data/train"
VALIDATION_DIR = "data/validation"

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"

BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

CHUNK_SIZE = 256
HOP_SIZE = 128


# ============================================
# DEVICE
# ============================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("CNN SPEECH ENHANCEMENT TRAINING")
print("=" * 60)

print("\nDevice:")
print(device)


# ============================================
# CREATE OUTPUT DIRECTORIES
# ============================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================
# TRAINING DATASET
# ============================================

train_dataset = ChunkedSpeechEnhancementDataset(
    data_dir=TRAIN_DIR,
    chunk_size=CHUNK_SIZE,
    hop_size=HOP_SIZE
)


# ============================================
# VALIDATION DATASET
# ============================================

validation_dataset = ChunkedSpeechEnhancementDataset(
    data_dir=VALIDATION_DIR,
    chunk_size=CHUNK_SIZE,
    hop_size=HOP_SIZE
)


print("\nTraining dataset size:")
print(len(train_dataset))

print("\nValidation dataset size:")
print(len(validation_dataset))


# ============================================
# DATA LOADERS
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nTraining batches:")
print(len(train_loader))

print("\nValidation batches:")
print(len(validation_loader))


# ============================================
# MODEL
# ============================================

model = SpeechEnhancementCNN()

model = model.to(device)


# ============================================
# LOSS FUNCTION
# ============================================

criterion = SpectrogramMSELoss()


# ============================================
# OPTIMIZER
# ============================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================
# TRAINING HISTORY
# ============================================

train_losses = []
validation_losses = []

best_validation_loss = float("inf")


# ============================================
# TRAINING LOOP
# ============================================

for epoch in range(NUM_EPOCHS):

    # ----------------------------------------
    # TRAINING
    # ----------------------------------------

    model.train()

    running_train_loss = 0.0

    for noisy, clean in train_loader:

        noisy = noisy.to(device)
        clean = clean.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        predicted = model(noisy)

        # Calculate loss
        loss = criterion(
            predicted,
            clean
        )

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        running_train_loss += (
            loss.item()
        )

    average_train_loss = (
        running_train_loss
        / len(train_loader)
    )


    # ----------------------------------------
    # VALIDATION
    # ----------------------------------------

    model.eval()

    running_validation_loss = 0.0

    with torch.no_grad():

        for noisy, clean in validation_loader:

            noisy = noisy.to(device)
            clean = clean.to(device)

            predicted = model(noisy)

            loss = criterion(
                predicted,
                clean
            )

            running_validation_loss += (
                loss.item()
            )

    average_validation_loss = (
        running_validation_loss
        / len(validation_loader)
    )


    # ----------------------------------------
    # STORE HISTORY
    # ----------------------------------------

    train_losses.append(
        average_train_loss
    )

    validation_losses.append(
        average_validation_loss
    )


    # ----------------------------------------
    # PRINT RESULTS
    # ----------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {average_train_loss:.6f} | "
        f"Validation Loss: {average_validation_loss:.6f}"
    )


    # ----------------------------------------
    # SAVE BEST MODEL
    # ----------------------------------------

    if average_validation_loss < best_validation_loss:

        best_validation_loss = (
            average_validation_loss
        )

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            "cnn_baseline_best.pth"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            f"  Best model saved → "
            f"{checkpoint_path}"
        )


# ============================================
# SAVE FINAL MODEL
# ============================================

final_model_path = os.path.join(
    CHECKPOINT_DIR,
    "cnn_baseline_final.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)


# ============================================
# PLOT LOSS CURVES
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    validation_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title(
    "CNN Speech Enhancement Training"
)

plt.legend()
plt.grid(True)

loss_curve_path = os.path.join(
    OUTPUT_DIR,
    "training_validation_loss.png"
)

plt.savefig(
    loss_curve_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nBest validation loss:")
print(f"{best_validation_loss:.6f}")

print("\nFinal model:")
print(final_model_path)

print("\nBest model:")
print(
    os.path.join(
        CHECKPOINT_DIR,
        "cnn_baseline_best.pth"
    )
)

print("\nLoss curve:")
print(loss_curve_path)
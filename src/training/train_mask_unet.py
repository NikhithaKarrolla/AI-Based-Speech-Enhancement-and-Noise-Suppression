import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.spectrogram_dataset import (
    ChunkedSpeechEnhancementDataset
)

from src.models.mask_unet import MaskUNet

from src.training.loss import SpectrogramMSELoss


# ============================================
# Configuration
# ============================================

TRAIN_DIR = "data/train"
VALIDATION_DIR = "data/validation"

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"

BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

CHUNK_SIZE = 256
HOP_SIZE = 128

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "mask_unet_best.pth"
)

FINAL_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "mask_unet_final.pth"
)

LOSS_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "mask_unet_training_validation_loss.png"
)


# ============================================
# Device
# ============================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================
# Dataset
# ============================================

train_dataset = ChunkedSpeechEnhancementDataset(
    data_dir=TRAIN_DIR,
    chunk_size=CHUNK_SIZE,
    hop_size=HOP_SIZE
)

validation_dataset = ChunkedSpeechEnhancementDataset(
    data_dir=VALIDATION_DIR,
    chunk_size=CHUNK_SIZE,
    hop_size=HOP_SIZE
)

print(
    "Training dataset size:",
    len(train_dataset)
)

print(
    "Validation dataset size:",
    len(validation_dataset)
)


# ============================================
# DataLoaders
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

print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(validation_loader)
)


# ============================================
# Model
# ============================================

model = MaskUNet().to(device)


# ============================================
# Loss and optimizer
# ============================================

criterion = SpectrogramMSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================
# Training history
# ============================================

train_losses = []
validation_losses = []

best_validation_loss = float("inf")


# ============================================
# Training loop
# ============================================

for epoch in range(NUM_EPOCHS):

    # ----------------------------------------
    # Training
    # ----------------------------------------

    model.train()

    running_train_loss = 0.0

    for noisy, clean in train_loader:

        noisy = noisy.to(device)
        clean = clean.to(device)

        # Dataset contains log magnitudes.
        # Convert log magnitude back to magnitude.
        noisy_magnitude = torch.expm1(noisy)

        clean_magnitude = torch.expm1(clean)

        # Predict enhancement mask
        mask = model(noisy)

        # Apply mask to noisy magnitude
        enhanced_magnitude = (
            noisy_magnitude * mask
        )

        # Convert enhanced magnitude to
        # log magnitude for loss calculation.
        enhanced_log_magnitude = torch.log1p(
            enhanced_magnitude
        )

        # Calculate loss
        loss = criterion(
            enhanced_log_magnitude,
            clean
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_train_loss += (
            loss.item()
        )

    train_loss = (
        running_train_loss
        / len(train_loader)
    )


    # ----------------------------------------
    # Validation
    # ----------------------------------------

    model.eval()

    running_validation_loss = 0.0

    with torch.no_grad():

        for noisy, clean in validation_loader:

            noisy = noisy.to(device)
            clean = clean.to(device)

            noisy_magnitude = torch.expm1(
                noisy
            )

            mask = model(noisy)

            enhanced_magnitude = (
                noisy_magnitude * mask
            )

            enhanced_log_magnitude = torch.log1p(
                enhanced_magnitude
            )

            loss = criterion(
                enhanced_log_magnitude,
                clean
            )

            running_validation_loss += (
                loss.item()
            )

    validation_loss = (
        running_validation_loss
        / len(validation_loader)
    )


    # ----------------------------------------
    # Store losses
    # ----------------------------------------

    train_losses.append(
        train_loss
    )

    validation_losses.append(
        validation_loss
    )


    print(
        f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Validation Loss: {validation_loss:.6f}"
    )


    # ----------------------------------------
    # Save best model
    # ----------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            "  Saved best model."
        )


# ============================================
# Save final model
# ============================================

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)


# ============================================
# Plot training history
# ============================================

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    validation_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title(
    "Mask U-Net Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    LOSS_PLOT_PATH
)

plt.close()


# ============================================
# Final summary
# ============================================

print()
print("TRAINING COMPLETE")

print(
    "Best validation loss:",
    f"{best_validation_loss:.6f}"
)

print(
    "Best model:",
    BEST_MODEL_PATH
)

print(
    "Final model:",
    FINAL_MODEL_PATH
)

print(
    "Loss plot:",
    LOSS_PLOT_PATH
)
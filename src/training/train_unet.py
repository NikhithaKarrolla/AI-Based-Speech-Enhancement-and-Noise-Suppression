import os

import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.data.spectrogram_dataset import ChunkedSpeechEnhancementDataset
from src.models.unet import SpeechEnhancementUNet
from src.training.loss import SpectrogramMSELoss


# ============================================
# Configuration
# ============================================

TRAIN_DIR = "data/train"
VALIDATION_DIR = "data/validation"

BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

CHUNK_SIZE = 256
HOP_SIZE = 128

BEST_CHECKPOINT = "checkpoints/unet_best.pth"
FINAL_CHECKPOINT = "checkpoints/unet_final.pth"

LOSS_PLOT = "outputs/unet_training_validation_loss.png"


# ============================================
# Training
# ============================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    for noisy, clean in dataloader:

        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        predicted = model(noisy)

        loss = criterion(
            predicted,
            clean
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * noisy.size(0)
        )

    epoch_loss = (
        running_loss /
        len(dataloader.dataset)
    )

    return epoch_loss


# ============================================
# Validation
# ============================================

def validate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for noisy, clean in dataloader:

            noisy = noisy.to(device)
            clean = clean.to(device)

            predicted = model(noisy)

            loss = criterion(
                predicted,
                clean
            )

            running_loss += (
                loss.item() * noisy.size(0)
            )

    epoch_loss = (
        running_loss /
        len(dataloader.dataset)
    )

    return epoch_loss


# ============================================
# Main
# ============================================

def main():

    print("=" * 70)
    print("U-NET SPEECH ENHANCEMENT TRAINING")
    print("=" * 70)

    # -------------------------
    # Device
    # -------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:")
    print(device)

    # -------------------------
    # Datasets
    # -------------------------

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

    print("\nTraining dataset size:")
    print(len(train_dataset))

    print("\nValidation dataset size:")
    print(len(validation_dataset))

    # -------------------------
    # DataLoaders
    # -------------------------

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

    # -------------------------
    # Model
    # -------------------------

    model = SpeechEnhancementUNet()

    model = model.to(device)

    # -------------------------
    # Loss
    # -------------------------

    criterion = SpectrogramMSELoss()

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -------------------------
    # Training history
    # -------------------------

    train_losses = []
    validation_losses = []

    best_validation_loss = float("inf")

    # -------------------------
    # Training loop
    # -------------------------

    for epoch in range(NUM_EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        validation_loss = validate(
            model,
            validation_loader,
            criterion,
            device
        )

        train_losses.append(
            train_loss
        )

        validation_losses.append(
            validation_loss
        )

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} "
            f"| Train Loss: {train_loss:.6f} "
            f"| Validation Loss: {validation_loss:.6f}"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            torch.save(
                model.state_dict(),
                BEST_CHECKPOINT
            )

            print(
                "  → Best model saved."
            )

    # -------------------------
    # Save final model
    # -------------------------

    torch.save(
        model.state_dict(),
        FINAL_CHECKPOINT
    )

    # -------------------------
    # Plot losses
    # -------------------------

    plt.figure(figsize=(8, 5))

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
        "U-Net Training and Validation Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        LOSS_PLOT,
        dpi=150
    )

    plt.close()

    print("\n" + "=" * 70)

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"\nBest model saved to:\n"
        f"{BEST_CHECKPOINT}"
    )

    print(
        f"\nFinal model saved to:\n"
        f"{FINAL_CHECKPOINT}"
    )

    print(
        f"\nLoss plot saved to:\n"
        f"{LOSS_PLOT}"
    )

    print("\nU-NET TRAINING COMPLETE")

    print("=" * 70)


if __name__ == "__main__":
    main()
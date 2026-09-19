import os

import torch
from torch.utils.data import DataLoader

from src.data.spectrogram_dataset import ChunkedSpeechEnhancementDataset
from src.models.cnn_baseline import SpeechEnhancementCNN
from src.training.loss import SpectrogramMSELoss


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/train"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 10


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SPEECH ENHANCEMENT CNN TRAINING")
print("=" * 60)

print("\nDevice:")
print(device)


# ============================================================
# DATASET
# ============================================================

dataset = ChunkedSpeechEnhancementDataset(
    data_dir=DATA_DIR,
    chunk_size=256,
    hop_size=128
)

print("\nDataset size:")
print(len(dataset))


# ============================================================
# DATALOADER
# ============================================================

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\nNumber of batches:")
print(len(dataloader))


# ============================================================
# MODEL
# ============================================================

model = SpeechEnhancementCNN()

model = model.to(device)

print("\nModel:")
print(model)


# ============================================================
# LOSS
# ============================================================

criterion = SpectrogramMSELoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    print("\n" + "-" * 60)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 60)

    for batch_index, (noisy, clean) in enumerate(dataloader):

        noisy = noisy.to(device)
        clean = clean.to(device)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        predicted = model(noisy)

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            predicted,
            clean
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # ----------------------------------------------------
        # Display progress
        # ----------------------------------------------------

        if (batch_index + 1) % 5 == 0:

            print(
                f"Batch "
                f"{batch_index + 1}/{len(dataloader)} "
                f"- Loss: {loss.item():.6f}"
            )

    # ========================================================
    # EPOCH LOSS
    # ========================================================

    epoch_loss = running_loss / len(dataloader)

    print(
        f"\nEpoch {epoch + 1} Loss: "
        f"{epoch_loss:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "cnn_baseline.pth"
)

torch.save(
    model.state_dict(),
    checkpoint_path
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nModel saved to:")
print(checkpoint_path)
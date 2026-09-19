import torch

from src.models.cnn_baseline import (
    SpeechEnhancementCNN
)


# ============================================
# CREATE MODEL
# ============================================

model = SpeechEnhancementCNN()


print("=" * 60)
print("CNN MODEL")
print("=" * 60)

print(model)


# ============================================
# CREATE TEST INPUT
# ============================================

batch_size = 2

channels = 1

frequency_bins = 257

time_frames = 7501


x = torch.randn(
    batch_size,
    channels,
    frequency_bins,
    time_frames
)


print("\nInput shape:")
print(x.shape)


# ============================================
# FORWARD PASS
# ============================================

with torch.no_grad():

    output = model(x)


print("\nOutput shape:")
print(output.shape)


# ============================================
# CHECK
# ============================================

if output.shape == x.shape:

    print("\nSUCCESS!")

    print(
        "Input and output dimensions match."
    )

else:

    print("\nERROR!")

    print(
        "Input and output dimensions do not match."
    )
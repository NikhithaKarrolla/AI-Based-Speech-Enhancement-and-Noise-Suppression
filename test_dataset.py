import torch

from src.data.speech_dataset import SpeechEnhancementDataset


# ============================================
# CREATE DATASET
# ============================================

dataset = SpeechEnhancementDataset(
    data_dir="data/train",
    sample_rate=16000,
    n_fft=512,
    hop_length=128,
    win_length=512
)


# ============================================
# DATASET SIZE
# ============================================

print("\nDataset size:")
print(len(dataset))


# ============================================
# GET ONE SAMPLE
# ============================================

noisy, clean = dataset[0]


print("\nTensor information:")

print("Noisy shape:")
print(noisy.shape)

print("\nClean shape:")
print(clean.shape)

print("\nNoisy dtype:")
print(noisy.dtype)

print("\nClean dtype:")
print(clean.dtype)
from torch.utils.data import DataLoader

from src.data.spectrogram_dataset import ChunkedSpeechEnhancementDataset


dataset = ChunkedSpeechEnhancementDataset(
    data_dir="data/train",
    chunk_size=256,
    hop_size=128
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

print("=" * 60)
print("DATALOADER TEST")
print("=" * 60)

print("\nDataset size:")
print(len(dataset))

print("\nBatch size:")
print(dataloader.batch_size)


noisy_batch, clean_batch = next(iter(dataloader))

print("\nNoisy batch shape:")
print(noisy_batch.shape)

print("\nClean batch shape:")
print(clean_batch.shape)

print("\nNoisy dtype:")
print(noisy_batch.dtype)

print("\nClean dtype:")
print(clean_batch.dtype)


expected_shape = (16, 1, 257, 256)

if noisy_batch.shape == expected_shape and clean_batch.shape == expected_shape:

    print("\nSUCCESS!")
    print("DataLoader is producing correct batches.")

else:

    print("\nERROR!")
    print("Unexpected batch dimensions.")
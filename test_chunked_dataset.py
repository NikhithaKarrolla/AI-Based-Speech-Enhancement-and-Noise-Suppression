from src.data.spectrogram_dataset import ChunkedSpeechEnhancementDataset


dataset = ChunkedSpeechEnhancementDataset(
    data_dir="data/train",
    chunk_size=256,
    hop_size=128
)

print("=" * 60)
print("CHUNKED SPEECH ENHANCEMENT DATASET")
print("=" * 60)

print("\nNumber of chunks:")
print(len(dataset))

noisy, clean = dataset[0]

print("\nFirst chunk:")

print("Noisy shape:")
print(noisy.shape)

print("\nClean shape:")
print(clean.shape)

print("\nNoisy dtype:")
print(noisy.dtype)

print("\nClean dtype:")
print(clean.dtype)


if noisy.shape == (1, 257, 256) and clean.shape == (1, 257, 256):

    print("\nSUCCESS!")
    print("Spectrogram chunking is working correctly.")

else:

    print("\nERROR!")
    print("Unexpected chunk dimensions.")
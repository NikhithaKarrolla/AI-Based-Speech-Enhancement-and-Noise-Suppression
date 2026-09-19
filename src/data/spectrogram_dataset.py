import torch
from torch.utils.data import Dataset
from src.data.speech_dataset import SpeechEnhancementDataset


class ChunkedSpeechEnhancementDataset(Dataset):
    """
    Splits full spectrograms into fixed-size time chunks.

    Each sample returned has shape:

        [1, frequency_bins, chunk_size]
    """

    def __init__(
        self,
        data_dir="data/train",
        chunk_size=256,
        hop_size=128
    ):
        self.base_dataset = SpeechEnhancementDataset(
            data_dir=data_dir
        )

        self.chunk_size = chunk_size
        self.hop_size = hop_size

        self.chunks = []

        self._create_chunk_index()

    def _create_chunk_index(self):
        for sample_index in range(len(self.base_dataset)):

            noisy, clean = self.base_dataset[sample_index]

            time_frames = noisy.shape[-1]

            start = 0

            while start + self.chunk_size <= time_frames:

                self.chunks.append(
                    (sample_index, start)
                )

                start += self.hop_size

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):

        sample_index, start = self.chunks[index]

        noisy, clean = self.base_dataset[sample_index]

        end = start + self.chunk_size

        noisy_chunk = noisy[:, :, start:end]
        clean_chunk = clean[:, :, start:end]

        return noisy_chunk, clean_chunk
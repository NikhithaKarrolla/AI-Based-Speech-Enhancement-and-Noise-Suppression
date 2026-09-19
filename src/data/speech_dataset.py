import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np


class SpeechEnhancementDataset(Dataset):

    def __init__(
        self,
        data_dir,
        sample_rate=16000,
        n_fft=512,
        hop_length=128,
        win_length=512
    ):
        self.data_dir = data_dir

        self.clean_dir = os.path.join(
            data_dir,
            "clean"
        )

        self.noisy_dir = os.path.join(
            data_dir,
            "noisy"
        )

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # ------------------------------------
        # Find noisy files
        # ------------------------------------

        self.files = [
            file
            for file in os.listdir(self.noisy_dir)
            if file.lower().endswith(".wav")
        ]

        self.files.sort()

        if len(self.files) == 0:
            raise ValueError(
                f"No WAV files found in {self.noisy_dir}"
            )

        # ------------------------------------
        # Build clean-file lookup
        # ------------------------------------

        self.clean_files = [
            file
            for file in os.listdir(self.clean_dir)
            if file.lower().endswith(".wav")
        ]

        self.clean_files.sort()

        if len(self.clean_files) == 0:
            raise ValueError(
                f"No WAV files found in {self.clean_dir}"
            )

        print(
            f"Found {len(self.files)} noisy samples "
            f"in {data_dir}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        noisy_filename = self.files[index]

        # ------------------------------------
        # Extract original clean filename
        #
        # Example:
        # clean_0002_snr_-5dB.wav
        #
        # becomes:
        # clean_0002_clean.wav
        # ------------------------------------

        base_name = noisy_filename.split("_snr_")[0]

        clean_filename = (
            f"{base_name}_clean.wav"
        )

        noisy_path = os.path.join(
            self.noisy_dir,
            noisy_filename
        )

        clean_path = os.path.join(
            self.clean_dir,
            clean_filename
        )

        # ------------------------------------
        # Verify clean file exists
        # ------------------------------------

        if not os.path.exists(clean_path):
            raise FileNotFoundError(
                f"Clean file not found:\n"
                f"{clean_path}\n\n"
                f"Noisy file:\n"
                f"{noisy_path}"
            )

        # ------------------------------------
        # Load audio
        # ------------------------------------

        noisy_audio, _ = librosa.load(
            noisy_path,
            sr=self.sample_rate,
            mono=True
        )

        clean_audio, _ = librosa.load(
            clean_path,
            sr=self.sample_rate,
            mono=True
        )

        # ------------------------------------
        # Make lengths equal
        # ------------------------------------

        min_length = min(
            len(noisy_audio),
            len(clean_audio)
        )

        noisy_audio = noisy_audio[:min_length]
        clean_audio = clean_audio[:min_length]

        # ------------------------------------
        # STFT
        # ------------------------------------

        noisy_stft = librosa.stft(
            noisy_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann"
        )

        clean_stft = librosa.stft(
            clean_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann"
        )

        # ------------------------------------
        # Magnitude
        # ------------------------------------

        noisy_magnitude = np.abs(
            noisy_stft
        )

        clean_magnitude = np.abs(
            clean_stft
        )

        # ------------------------------------
        # Log magnitude
        # ------------------------------------

        noisy_magnitude = np.log1p(
            noisy_magnitude
        )

        clean_magnitude = np.log1p(
            clean_magnitude
        )

        # ------------------------------------
        # Convert to PyTorch tensors
        # ------------------------------------

        noisy_tensor = torch.tensor(
            noisy_magnitude,
            dtype=torch.float32
        )

        clean_tensor = torch.tensor(
            clean_magnitude,
            dtype=torch.float32
        )

        # ------------------------------------
        # Add channel dimension
        #
        # [Frequency, Time]
        #       ↓
        # [1, Frequency, Time]
        # ------------------------------------

        noisy_tensor = noisy_tensor.unsqueeze(0)

        clean_tensor = clean_tensor.unsqueeze(0)

        return noisy_tensor, clean_tensor
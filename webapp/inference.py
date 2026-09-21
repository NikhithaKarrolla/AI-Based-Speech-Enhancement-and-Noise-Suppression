from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from src.models.mask_unet import MaskUNet


SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

CHECKPOINT_PATH = Path("checkpoints/mask_unet_best.pth")


class SpeechEnhancer:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = MaskUNet().to(self.device)

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=True
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        print(f"Speech enhancement model loaded on: {self.device}")

    def enhance(self, input_path: str, output_path: str):
        # Load audio
        audio, _ = librosa.load(
            input_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # STFT
        stft = librosa.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window="hann"
        )

        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Log magnitude
        log_magnitude = np.log1p(magnitude)

        # Model input
        input_tensor = torch.tensor(
            log_magnitude,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        input_tensor = input_tensor.to(self.device)

        # Model inference
        with torch.no_grad():
            mask = self.model(input_tensor)

        mask = mask.squeeze().cpu().numpy()

        # Apply predicted speech mask
        enhanced_magnitude = magnitude * mask

        # Reconstruct complex STFT
        enhanced_stft = (
            enhanced_magnitude *
            np.exp(1j * phase)
        )

        # iSTFT
        enhanced_audio = librosa.istft(
            enhanced_stft,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window="hann"
        )

        # Normalize
        max_value = np.max(np.abs(enhanced_audio))

        if max_value > 0:
            enhanced_audio = enhanced_audio / max_value

        # Save
        sf.write(
            output_path,
            enhanced_audio.astype(np.float32),
            SAMPLE_RATE
        )

        return output_path
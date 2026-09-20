import os

import numpy as np
import torch
import librosa
import soundfile as sf

from src.models.unet import SpeechEnhancementUNet


# ============================================
# Configuration
# ============================================

TEST_DIR = "data/test/noisy"

OUTPUT_DIR = "outputs/audio"

CHECKPOINT = "checkpoints/unet_best.pth"

SAMPLE_RATE = 16000

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512


# ============================================
# Load model
# ============================================

def load_model(device):

    model = SpeechEnhancementUNet()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()

    return model


# ============================================
# Enhance one audio file
# ============================================

def enhance_audio(
    model,
    input_path,
    output_path,
    device
):

    # -------------------------
    # Load audio
    # -------------------------

    audio, sample_rate = librosa.load(
        input_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # -------------------------
    # STFT
    # -------------------------

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann"
    )

    magnitude = np.abs(stft)

    phase = np.angle(stft)

    # -------------------------
    # Log magnitude
    # -------------------------

    log_magnitude = np.log1p(
        magnitude
    )

    # -------------------------
    # Tensor
    # -------------------------

    input_tensor = torch.from_numpy(
        log_magnitude
    ).float()

    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # -------------------------
    # U-Net inference
    # -------------------------

    with torch.no_grad():

        predicted = model(
            input_tensor
        )

    # -------------------------
    # Convert back to numpy
    # -------------------------

    predicted_log_magnitude = (
        predicted
        .squeeze()
        .cpu()
        .numpy()
    )

    # -------------------------
    # Convert log magnitude
    # back to magnitude
    # -------------------------

    enhanced_magnitude = np.expm1(
        predicted_log_magnitude
    )

    # Avoid negative values
    enhanced_magnitude = np.maximum(
        enhanced_magnitude,
        0.0
    )

    # -------------------------
    # Reconstruct complex STFT
    # -------------------------

    enhanced_stft = (
        enhanced_magnitude *
        np.exp(1j * phase)
    )

    # -------------------------
    # iSTFT
    # -------------------------

    enhanced_audio = librosa.istft(
        enhanced_stft,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann"
    )

    # -------------------------
    # Normalize
    # -------------------------

    peak = np.max(
        np.abs(enhanced_audio)
    )

    if peak > 1.0:

        enhanced_audio = (
            enhanced_audio / peak
        )

    # -------------------------
    # Save
    # -------------------------

    sf.write(
        output_path,
        enhanced_audio.astype(
            np.float32
        ),
        SAMPLE_RATE
    )

    return len(audio), len(enhanced_audio)


# ============================================
# Main
# ============================================

def main():

    print("=" * 70)
    print("U-NET TEST SET INFERENCE")
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
    # Check checkpoint
    # -------------------------

    if not os.path.exists(CHECKPOINT):

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    # -------------------------
    # Create output directory
    # -------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # -------------------------
    # Load model
    # -------------------------

    model = load_model(
        device
    )

    print("\nModel loaded:")
    print(CHECKPOINT)

    # -------------------------
    # Find test files
    # -------------------------

    noisy_files = sorted(
        file_name
        for file_name in os.listdir(TEST_DIR)
        if file_name.lower().endswith(".wav")
    )

    print("\nNumber of test files:")
    print(len(noisy_files))

    # -------------------------
    # Process files
    # -------------------------

    for file_name in noisy_files:

        input_path = os.path.join(
            TEST_DIR,
            file_name
        )

        # clean_0002_snr_5dB.wav
        #
        # Remove "_snr_" from filename:
        #
        # clean_0002_snr_5dB.wav
        #        ↓
        # clean_0002_5dB.wav

        if "_snr_" in file_name:

            base_name, snr_part = (
                file_name.split(
                    "_snr_",
                    1
                )
            )

            output_name = (
                f"unet_enhanced_"
                f"{base_name}_"
                f"{snr_part}"
            )

        else:

            output_name = (
                f"unet_enhanced_"
                f"{file_name}"
            )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        input_samples, output_samples = (
            enhance_audio(
                model,
                input_path,
                output_path,
                device
            )
        )

        print(
            f"\n{file_name}"
        )

        print(
            f"  Input samples: "
            f"{input_samples}"
        )

        print(
            f"  Enhanced samples: "
            f"{output_samples}"
        )

        print(
            f"  Saved: "
            f"{output_path}"
        )

    print("\n" + "=" * 70)

    print(
        "U-NET INFERENCE COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
import os

import numpy as np
import torch
import soundfile as sf
import librosa

from src.models.mask_unet import MaskUNet


# ============================================
# Configuration
# ============================================

TEST_DIR = "data/test/noisy"

OUTPUT_DIR = "outputs/audio"

CHECKPOINT_PATH = (
    "checkpoints/mask_unet_best.pth"
)

SAMPLE_RATE = 16000

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512


# ============================================
# Device
# ============================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================
# Load model
# ============================================

model = MaskUNet()

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print(
    "Loaded model:",
    CHECKPOINT_PATH
)


# ============================================
# Create output directory
# ============================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================
# Enhancement function
# ============================================

def enhance_audio(audio):

    # ----------------------------------------
    # STFT
    # ----------------------------------------

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann"
    )

    # ----------------------------------------
    # Magnitude and phase
    # ----------------------------------------

    noisy_magnitude = np.abs(stft)

    noisy_phase = np.angle(stft)

    # ----------------------------------------
    # Convert magnitude to log magnitude
    # ----------------------------------------

    noisy_log_magnitude = np.log1p(
        noisy_magnitude
    )

    # ----------------------------------------
    # Tensor
    # ----------------------------------------

    input_tensor = torch.from_numpy(
        noisy_log_magnitude
    ).float()

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    print(
        "Model input:",
        input_tensor.shape
    )

    # ----------------------------------------
    # Predict enhancement mask
    # ----------------------------------------

    with torch.no_grad():

        mask = model(
            input_tensor
        )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "Mask range:",
        f"{mask.min().item():.4f}",
        "to",
        f"{mask.max().item():.4f}"
    )

    # ----------------------------------------
    # Remove batch/channel dimensions
    # ----------------------------------------

    mask = mask.squeeze(
        0
    ).squeeze(
        0
    ).cpu().numpy()

    # ----------------------------------------
    # Apply mask
    # ----------------------------------------

    enhanced_magnitude = (
        noisy_magnitude * mask
    )

    # ----------------------------------------
    # Reconstruct complex STFT
    # using noisy phase
    # ----------------------------------------

    enhanced_stft = (
        enhanced_magnitude
        *
        np.exp(
            1j * noisy_phase
        )
    )

    # ----------------------------------------
    # iSTFT
    # ----------------------------------------

    enhanced_audio = librosa.istft(
        enhanced_stft,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann"
    )

    # ----------------------------------------
    # Peak normalization
    # ----------------------------------------

    peak = np.max(
        np.abs(enhanced_audio)
    )

    if peak > 1.0:

        enhanced_audio = (
            enhanced_audio / peak
        )

    return enhanced_audio.astype(
        np.float32
    )


# ============================================
# Process all test files
# ============================================

test_files = sorted(
    file_name
    for file_name in os.listdir(TEST_DIR)
    if file_name.lower().endswith(".wav")
)


print()
print(
    "Found",
    len(test_files),
    "test files."
)

print()


for file_name in test_files:

    input_path = os.path.join(
        TEST_DIR,
        file_name
    )

    # ----------------------------------------
    # Load audio
    # ----------------------------------------

    audio, sr = librosa.load(
        input_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    print(
        "Processing:",
        file_name
    )

    print(
        "Input samples:",
        len(audio)
    )

    print(
        "Duration:",
        f"{len(audio) / SAMPLE_RATE:.2f}",
        "seconds"
    )

    # ----------------------------------------
    # Enhance
    # ----------------------------------------

    enhanced_audio = enhance_audio(
        audio
    )

    # ----------------------------------------
    # Output name
    # ----------------------------------------

    output_name = (
        file_name
        .replace(
            "_snr_",
            "_"
        )
        .replace(
            ".wav",
            ""
        )
    )

    output_name = (
        "mask_unet_enhanced_"
        + output_name
        + ".wav"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    # ----------------------------------------
    # Save
    # ----------------------------------------

    sf.write(
        output_path,
        enhanced_audio,
        SAMPLE_RATE
    )

    print(
        "Enhanced samples:",
        len(enhanced_audio)
    )

    print(
        "Saved:",
        output_path
    )

    print()


print(
    "MASK U-NET INFERENCE COMPLETE"
)
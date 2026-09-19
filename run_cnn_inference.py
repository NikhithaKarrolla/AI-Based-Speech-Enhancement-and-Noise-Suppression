import os

import numpy as np
import torch
import librosa
import soundfile as sf

from src.models.cnn_baseline import SpeechEnhancementCNN


# ============================================
# CONFIGURATION
# ============================================

SAMPLE_RATE = 16000

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

CHECKPOINT_PATH = (
    "checkpoints/cnn_baseline_best.pth"
)

INPUT_DIR = "data/test/noisy"

OUTPUT_DIR = "outputs/audio"

INPUT_FILE = "clean_0002_snr_5dB.wav"

OUTPUT_FILE = (
    "cnn_enhanced_clean_0002_5dB.wav"
)


# ============================================
# DEVICE
# ============================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("CNN SPEECH ENHANCEMENT INFERENCE")
print("=" * 60)

print("\nDevice:")
print(device)


# ============================================
# PATHS
# ============================================

input_path = os.path.join(
    INPUT_DIR,
    INPUT_FILE
)

output_path = os.path.join(
    OUTPUT_DIR,
    OUTPUT_FILE
)


# ============================================
# CHECK INPUT
# ============================================

if not os.path.exists(input_path):

    raise FileNotFoundError(
        f"Input file not found: {input_path}"
    )


if not os.path.exists(
    CHECKPOINT_PATH
):

    raise FileNotFoundError(
        f"Checkpoint not found: "
        f"{CHECKPOINT_PATH}"
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================
# LOAD MODEL
# ============================================

model = SpeechEnhancementCNN()

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


print("\nModel loaded successfully.")


# ============================================
# LOAD AUDIO
# ============================================

noisy_audio, _ = librosa.load(
    input_path,
    sr=SAMPLE_RATE,
    mono=True
)

print("\nInput audio:")
print(input_path)

print("Samples:")
print(len(noisy_audio))

print("Duration:")
print(
    f"{len(noisy_audio) / SAMPLE_RATE:.2f} seconds"
)


# ============================================
# COMPUTE STFT
# ============================================

noisy_stft = librosa.stft(
    noisy_audio,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window="hann"
)


# ============================================
# MAGNITUDE + PHASE
# ============================================

noisy_magnitude = np.abs(
    noisy_stft
)

noisy_phase = np.angle(
    noisy_stft
)


# ============================================
# LOG MAGNITUDE
# ============================================

noisy_log_magnitude = np.log1p(
    noisy_magnitude
)


# ============================================
# CONVERT TO PYTORCH TENSOR
# ============================================

input_tensor = torch.tensor(
    noisy_log_magnitude,
    dtype=torch.float32
)


# Shape:

# [Frequency, Time]

input_tensor = input_tensor.unsqueeze(
    0
)

# Shape:

# [Channel, Frequency, Time]

input_tensor = input_tensor.unsqueeze(
    0
)

# Shape:

# [Batch, Channel, Frequency, Time]

input_tensor = input_tensor.to(
    device
)


print("\nModel input shape:")
print(input_tensor.shape)


# ============================================
# CNN INFERENCE
# ============================================

with torch.no_grad():

    predicted_log_magnitude = model(
        input_tensor
    )


print("\nCNN output shape:")
print(
    predicted_log_magnitude.shape
)


# ============================================
# REMOVE BATCH + CHANNEL DIMENSIONS
# ============================================

predicted_log_magnitude = (
    predicted_log_magnitude
    .squeeze(0)
    .squeeze(0)
    .cpu()
    .numpy()
)


# ============================================
# CONVERT LOG MAGNITUDE BACK
# ============================================

predicted_magnitude = np.expm1(
    predicted_log_magnitude
)


# Prevent negative values caused by
# numerical/model output behavior

predicted_magnitude = np.maximum(
    predicted_magnitude,
    0.0
)


# ============================================
# RECONSTRUCT COMPLEX STFT
# ============================================

enhanced_stft = (
    predicted_magnitude
    *
    np.exp(1j * noisy_phase)
)


# ============================================
# iSTFT
# ============================================

enhanced_audio = librosa.istft(
    enhanced_stft,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window="hann"
)


# ============================================
# MATCH INPUT LENGTH
# ============================================

enhanced_audio = enhanced_audio[
    :len(noisy_audio)
]


# ============================================
# NORMALIZE
# ============================================

peak = np.max(
    np.abs(enhanced_audio)
)

if peak > 1.0:

    enhanced_audio = (
        enhanced_audio / peak
    )


# ============================================
# SAVE
# ============================================

sf.write(
    output_path,
    enhanced_audio.astype(
        np.float32
    ),
    SAMPLE_RATE
)


# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("INFERENCE COMPLETE")
print("=" * 60)

print("\nEnhanced audio:")
print(output_path)

print("\nEnhanced samples:")
print(len(enhanced_audio))

print("\nEnhanced duration:")
print(
    f"{len(enhanced_audio) / SAMPLE_RATE:.2f} seconds"
)
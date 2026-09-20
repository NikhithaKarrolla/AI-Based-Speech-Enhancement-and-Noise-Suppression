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


# ============================================
# DEVICE
# ============================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("CNN TEST SET INFERENCE")
print("=" * 60)

print("\nDevice:")
print(device)


# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================

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
# FIND TEST FILES
# ============================================

input_files = sorted(
    file
    for file in os.listdir(INPUT_DIR)
    if file.lower().endswith(".wav")
)


print("\nTest files found:")
print(len(input_files))

for file in input_files:
    print(" ", file)


# ============================================
# PROCESS EACH FILE
# ============================================

for index, input_file in enumerate(
    input_files,
    start=1
):

    print("\n" + "-" * 60)

    print(
        f"Processing {index}/{len(input_files)}:"
    )

    print(input_file)


    # ----------------------------------------
    # INPUT PATH
    # ----------------------------------------

    input_path = os.path.join(
        INPUT_DIR,
        input_file
    )


    # ----------------------------------------
    # OUTPUT NAME
    # ----------------------------------------

    # clean_0002_snr_5dB.wav
    #
    # →
    #
    # cnn_enhanced_clean_0002_5dB.wav

    base_name = input_file.replace(
        "_snr_",
        "_"
    )

    output_file = (
        "cnn_enhanced_"
        + base_name
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_file
    )


    # ----------------------------------------
    # LOAD AUDIO
    # ----------------------------------------

    noisy_audio, _ = librosa.load(
        input_path,
        sr=SAMPLE_RATE,
        mono=True
    )


    # ----------------------------------------
    # STFT
    # ----------------------------------------

    noisy_stft = librosa.stft(
        noisy_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann"
    )


    # ----------------------------------------
    # MAGNITUDE + PHASE
    # ----------------------------------------

    noisy_magnitude = np.abs(
        noisy_stft
    )

    noisy_phase = np.angle(
        noisy_stft
    )


    # ----------------------------------------
    # LOG MAGNITUDE
    # ----------------------------------------

    noisy_log_magnitude = np.log1p(
        noisy_magnitude
    )


    # ----------------------------------------
    # TENSOR
    # ----------------------------------------

    input_tensor = torch.tensor(
        noisy_log_magnitude,
        dtype=torch.float32
    )

    input_tensor = input_tensor.unsqueeze(
        0
    )

    input_tensor = input_tensor.unsqueeze(
        0
    )

    input_tensor = input_tensor.to(
        device
    )


    # ----------------------------------------
    # CNN
    # ----------------------------------------

    with torch.no_grad():

        predicted_log_magnitude = model(
            input_tensor
        )


    # ----------------------------------------
    # NUMPY
    # ----------------------------------------

    predicted_log_magnitude = (
        predicted_log_magnitude
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
    )


    # ----------------------------------------
    # LOG → MAGNITUDE
    # ----------------------------------------

    predicted_magnitude = np.expm1(
        predicted_log_magnitude
    )

    predicted_magnitude = np.maximum(
        predicted_magnitude,
        0.0
    )


    # ----------------------------------------
    # RECONSTRUCT STFT
    # ----------------------------------------

    enhanced_stft = (
        predicted_magnitude
        *
        np.exp(1j * noisy_phase)
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
    # MATCH LENGTH
    # ----------------------------------------

    enhanced_audio = enhanced_audio[
        :len(noisy_audio)
    ]


    # ----------------------------------------
    # NORMALIZE IF NECESSARY
    # ----------------------------------------

    peak = np.max(
        np.abs(enhanced_audio)
    )

    if peak > 1.0:

        enhanced_audio = (
            enhanced_audio / peak
        )


    # ----------------------------------------
    # SAVE
    # ----------------------------------------

    sf.write(
        output_path,
        enhanced_audio.astype(
            np.float32
        ),
        SAMPLE_RATE
    )


    print(
        "Saved:"
    )

    print(
        output_path
    )


# ============================================
# COMPLETE
# ============================================

print("\n" + "=" * 60)
print("ALL TEST FILES PROCESSED")
print("=" * 60)

print("\nEnhanced files:")
print(
    len(input_files)
)

print("\nOutput directory:")
print(OUTPUT_DIR)
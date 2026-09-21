import sys
import os
import queue
import threading

import numpy as np
import torch
import librosa
import sounddevice as sd


sys.path.append(
    os.path.abspath(".")
)

from src.models.mask_unet import MaskUNet


# ============================================
# Configuration
# ============================================

SAMPLE_RATE = 16000

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

BLOCK_SIZE = 2048

CHECKPOINT = (
    "checkpoints/mask_unet_best.pth"
)

QUEUE_SIZE = 8


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
        CHECKPOINT,
        map_location=device
    )
)

model.to(device)

model.eval()

print(
    "Loaded:",
    CHECKPOINT
)


# ============================================
# Queues
# ============================================

input_queue = queue.Queue(
    maxsize=QUEUE_SIZE
)

output_queue = queue.Queue(
    maxsize=QUEUE_SIZE
)


# ============================================
# Enhancement
# ============================================

def enhance_chunk(audio):

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann"
    )

    magnitude = np.abs(stft)

    phase = np.angle(stft)

    log_magnitude = np.log1p(
        magnitude
    )

    tensor = torch.from_numpy(
        log_magnitude
    ).float()

    tensor = tensor.unsqueeze(0)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        mask = model(tensor)

    mask = (
        mask
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    enhanced_magnitude = (
        magnitude * mask
    )

    enhanced_stft = (
        enhanced_magnitude
        *
        np.exp(1j * phase)
    )

    enhanced = librosa.istft(
        enhanced_stft,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann"
    )

    enhanced = enhanced.astype(
        np.float32
    )

    return enhanced


# ============================================
# Worker thread
# ============================================

def processing_worker():

    while True:

        audio = input_queue.get()

        if audio is None:

            break

        try:

            enhanced = enhance_chunk(
                audio
            )

            try:

                output_queue.put_nowait(
                    enhanced
                )

            except queue.Full:

                # Drop old output if necessary
                try:

                    output_queue.get_nowait()

                except queue.Empty:

                    pass

                try:

                    output_queue.put_nowait(
                        enhanced
                    )

                except queue.Full:

                    pass

        except Exception as error:

            print(
                "Processing error:",
                error
            )


# ============================================
# Input callback
# ============================================

def input_callback(
    indata,
    frames,
    time,
    status
):

    if status:

        print(
            "Input:",
            status
        )

    audio = indata[:, 0].copy()

    try:

        input_queue.put_nowait(
            audio
        )

    except queue.Full:

        # Drop newest block instead of
        # blocking the audio callback.
        pass


# ============================================
# Output callback
# ============================================

def output_callback(
    outdata,
    frames,
    time,
    status
):

    if status:

        print(
            "Output:",
            status
        )

    outdata.fill(0)

    try:

        enhanced = (
            output_queue.get_nowait()
        )

        length = min(
            len(enhanced),
            frames
        )

        outdata[
            :length,
            0
        ] = enhanced[
            :length
        ]

    except queue.Empty:

        pass


# ============================================
# Main
# ============================================

print()
print(
    "======================================"
)

print(
    "REAL-TIME SPEECH ENHANCEMENT"
)

print(
    "======================================"
)

print(
    "Sample rate:",
    SAMPLE_RATE
)

print(
    "Block size:",
    BLOCK_SIZE
)

print()
print(
    "Speak into your microphone."
)

print(
    "Press Ctrl+C to stop."
)

print()


worker = threading.Thread(
    target=processing_worker,
    daemon=True
)

worker.start()


try:

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=input_callback
    ):

        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=output_callback
        ):

            while True:

                sd.sleep(1000)


except KeyboardInterrupt:

    print()
    print(
        "Stopping real-time enhancement..."
    )


finally:

    try:

        input_queue.put_nowait(
            None
        )

    except queue.Full:

        pass

    worker.join(
        timeout=2
    )

    print(
        "Real-time enhancement stopped."
    )
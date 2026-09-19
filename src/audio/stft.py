import numpy as np
import librosa


def compute_stft(
    audio: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512
):
    """
    Compute the Short-Time Fourier Transform.

    Returns:
        stft_matrix: complex-valued STFT
    """

    stft_matrix = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann"
    )

    return stft_matrix


def get_magnitude(
    stft_matrix: np.ndarray
):
    """
    Extract magnitude from complex STFT.
    """

    magnitude = np.abs(
        stft_matrix
    )

    return magnitude


def get_phase(
    stft_matrix: np.ndarray
):
    """
    Extract phase from complex STFT.
    """

    phase = np.angle(
        stft_matrix
    )

    return phase


def reconstruct_audio(
    magnitude: np.ndarray,
    phase: np.ndarray,
    hop_length: int = 128,
    win_length: int = 512
):
    """
    Reconstruct time-domain audio from
    magnitude and phase.
    """

    complex_stft = (
        magnitude *
        np.exp(1j * phase)
    )

    audio = librosa.istft(
        complex_stft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann"
    )

    return audio.astype(
        np.float32
    )
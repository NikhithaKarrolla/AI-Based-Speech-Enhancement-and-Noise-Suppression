import numpy as np
import librosa
import soundfile as sf


def load_audio(
    file_path: str,
    sample_rate: int = 16000
):
    """
    Load an audio file.

    The audio is:
    - converted to mono
    - resampled to the required sample rate

    Returns:
        audio: NumPy array containing audio samples
        sr: sample rate
    """

    audio, sr = librosa.load(
        file_path,
        sr=sample_rate,
        mono=True
    )

    audio = audio.astype(np.float32)

    return audio, sr


def normalize_audio(audio: np.ndarray):
    """
    Peak-normalize an audio signal.

    The maximum absolute amplitude
    becomes approximately 1.0.
    """

    max_amplitude = np.max(
        np.abs(audio)
    )

    if max_amplitude == 0:
        return audio

    normalized_audio = (
        audio / max_amplitude
    )

    return normalized_audio.astype(
        np.float32
    )


def save_audio(
    file_path: str,
    audio: np.ndarray,
    sample_rate: int = 16000
):
    """
    Save an audio signal as a WAV file.
    """

    sf.write(
        file_path,
        audio,
        sample_rate
    )


def preprocess_audio(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000
):
    """
    Complete preprocessing pipeline:

    Input audio
        ↓
    Load
        ↓
    Mono
        ↓
    Resample
        ↓
    Normalize
        ↓
    Save
    """

    audio, sr = load_audio(
        input_path,
        sample_rate
    )

    audio = normalize_audio(audio)

    save_audio(
        output_path,
        audio,
        sr
    )

    return audio, sr
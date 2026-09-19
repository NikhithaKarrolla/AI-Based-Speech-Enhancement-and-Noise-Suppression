import numpy as np


def get_audio_info(
    audio: np.ndarray,
    sample_rate: int
):
    """
    Return basic information about an audio signal.
    """

    duration = (
        len(audio) / sample_rate
    )

    rms = np.sqrt(
        np.mean(audio ** 2)
    )

    return {
        "samples": len(audio),
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "min_amplitude": float(
            np.min(audio)
        ),
        "max_amplitude": float(
            np.max(audio)
        ),
        "rms": float(rms)
    }
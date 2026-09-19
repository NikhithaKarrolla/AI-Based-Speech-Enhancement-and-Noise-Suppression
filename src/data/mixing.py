import numpy as np
import soundfile as sf


def calculate_rms(signal: np.ndarray) -> float:
    """
    Calculate RMS energy of an audio signal.

    RMS = sqrt(mean(signal^2))
    """

    signal = signal.astype(np.float64)

    return float(
        np.sqrt(
            np.mean(signal ** 2)
        )
    )


def calculate_snr(
    clean: np.ndarray,
    noise: np.ndarray
) -> float:
    """
    Calculate SNR in decibels.

    SNR(dB) =
        20 * log10(RMS(clean) / RMS(noise))
    """

    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)

    if noise_rms == 0:
        return float("inf")

    snr = 20 * np.log10(
        clean_rms / noise_rms
    )

    return float(snr)


def scale_noise_to_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float
) -> np.ndarray:
    """
    Scale the noise so that the resulting
    clean/noise pair has the requested SNR.
    """

    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)

    if clean_rms == 0:
        raise ValueError(
            "Clean speech has zero energy."
        )

    if noise_rms == 0:
        raise ValueError(
            "Noise has zero energy."
        )

    # Convert target SNR from dB to linear ratio
    target_ratio = 10 ** (
        target_snr_db / 20
    )

    # Desired noise RMS
    desired_noise_rms = (
        clean_rms / target_ratio
    )

    # Scaling factor
    scale_factor = (
        desired_noise_rms / noise_rms
    )

    scaled_noise = (
        noise * scale_factor
    )

    return scaled_noise.astype(
        np.float32
    )


def mix_audio(
    clean: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float
):
    """
    Mix clean speech and noise at a target SNR.

    Returns:
        noisy_audio
        scaled_noise
    """

    # Make sure both signals are NumPy arrays
    clean = np.asarray(
        clean,
        dtype=np.float32
    )

    noise = np.asarray(
        noise,
        dtype=np.float32
    )

    # Make noise at least as long as clean speech
    if len(noise) < len(clean):

        repetitions = int(
            np.ceil(
                len(clean) / len(noise)
            )
        )

        noise = np.tile(
            noise,
            repetitions
        )

    # Random starting point
    max_start = (
        len(noise) - len(clean)
    )

    if max_start > 0:

        start = np.random.randint(
            0,
            max_start + 1
        )

        noise = noise[
            start:start + len(clean)
        ]

    else:

        noise = noise[:len(clean)]

    # Scale noise
    scaled_noise = scale_noise_to_snr(
        clean,
        noise,
        target_snr_db
    )

    # Mix
    noisy_audio = (
        clean + scaled_noise
    )

    return noisy_audio.astype(np.float32)


def peak_normalize(
    audio: np.ndarray
) -> np.ndarray:
    """
    Prevent clipping by peak-normalizing audio.
    """

    peak = np.max(
        np.abs(audio)
    )

    if peak == 0:
        return audio

    return (
        audio / peak
    ).astype(np.float32)


def save_audio(
    file_path: str,
    audio: np.ndarray,
    sample_rate: int
):
    """
    Save audio as WAV.
    """

    sf.write(
        file_path,
        audio,
        sample_rate
    )
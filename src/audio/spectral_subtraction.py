import numpy as np
import librosa


def estimate_noise_spectrum(
    noisy_audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    noise_duration: float = 0.5
):
    """
    Estimate the noise magnitude spectrum from the
    initial noise_duration seconds of the recording.

    Assumption:
    The beginning of the recording contains mostly noise.
    """

    noise_samples = int(noise_duration * sample_rate)

    if len(noisy_audio) < noise_samples:
        noise_samples = len(noisy_audio)

    noise_segment = noisy_audio[:noise_samples]

    noise_stft = librosa.stft(
        noise_segment,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann"
    )

    noise_magnitude = np.abs(noise_stft)

    # Average noise magnitude over time
    noise_spectrum = np.mean(noise_magnitude, axis=1)

    return noise_spectrum


def spectral_subtraction(
    noisy_audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    noise_duration: float = 0.5,
    alpha: float = 1.0,
    beta: float = 0.02
):
    """
    Perform spectral subtraction.

    alpha:
        Noise subtraction strength.

    beta:
        Spectral floor to prevent the magnitude
        from becoming exactly zero.
    """

    # -----------------------------------------
    # 1. Compute STFT
    # -----------------------------------------

    noisy_stft = librosa.stft(
        noisy_audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann"
    )

    # -----------------------------------------
    # 2. Separate magnitude and phase
    # -----------------------------------------

    noisy_magnitude = np.abs(noisy_stft)
    noisy_phase = np.angle(noisy_stft)

    # -----------------------------------------
    # 3. Estimate noise spectrum
    # -----------------------------------------

    noise_spectrum = estimate_noise_spectrum(
        noisy_audio,
        sample_rate,
        n_fft,
        hop_length,
        win_length,
        noise_duration
    )

    # Make noise spectrum compatible with
    # all time frames
    noise_spectrum = noise_spectrum[:, np.newaxis]

    # -----------------------------------------
    # 4. Spectral subtraction
    # -----------------------------------------

    enhanced_magnitude = noisy_magnitude - (
        alpha * noise_spectrum
    )

    # -----------------------------------------
    # 5. Apply spectral floor
    # -----------------------------------------

    minimum_magnitude = beta * noisy_magnitude

    enhanced_magnitude = np.maximum(
        enhanced_magnitude,
        minimum_magnitude
    )

    # -----------------------------------------
    # 6. Reconstruct complex STFT
    # -----------------------------------------

    enhanced_stft = (
        enhanced_magnitude *
        np.exp(1j * noisy_phase)
    )

    # -----------------------------------------
    # 7. Convert back to waveform
    # -----------------------------------------

    enhanced_audio = librosa.istft(
        enhanced_stft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann"
    )

    return enhanced_audio.astype(np.float32)
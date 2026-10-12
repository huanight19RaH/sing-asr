"""
sing-asr Advanced DSP Module: Transient-Preserving HPSS & Consonant Energy Boosting.
Directly addresses Phase VII of the Research Plan (SLP301 - AIT Laboratory).
Lead Researcher: Thai Hung Nguyen

Scientific Mechanism:
In Vietnamese singing, vowels carry the vast majority of acoustic power and are prolonged
by 200% - 500% across musical bars. Initial consonant attacks (plosives /t, k, p, b, d/,
affricates /ch, tr/, and fricatives /s, x, v/) have short durations (<80ms) and low relative energy.
Consequently, acoustic encoders and cross-attention mechanisms attend disproportionately to
the sustained harmonic energy, causing severe consonant deletion and substitution.

This module applies:
1. Harmonic-Percussive Source Separation (HPSS) via 2D median filtering on the STFT magnitude.
2. Soft spectral masking to cleanly isolate harmonic (vocal chord resonance) vs percussive (consonant transient) fields.
3. Selective amplification (boosting) of consonant transients by beta in [1.5, 2.5] (+3.5 dB to +8.0 dB).
4. Phase-coherent ISTFT reconstruction with peak normalization to ensure zero clipping.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import scipy.signal
import librosa

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dsp.audio_io import load_audio, save_audio


@dataclass
class HPSSBoostMetrics:
    original_energy_total: float
    harmonic_energy_ratio: float
    percussive_energy_ratio: float
    boost_factor: float
    transient_gain_db: float
    peak_normalized: bool


def separate_hpss_stft(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    kernel_harm: int = 31,
    kernel_perc: int = 31,
    margin_harm: float = 1.0,
    margin_perc: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separates a mono audio signal into Harmonic and Percussive components using STFT median filtering.

    Parameters:
    -----------
    audio : np.ndarray
        Mono input signal.
    sr : int
        Sample rate.
    n_fft : int
        FFT window length.
    hop_length : int
        STFT hop length.
    kernel_harm : int
        Median filter kernel length in frames (horizontal / time axis).
    kernel_perc : int
        Median filter kernel length in frequency bins (vertical / frequency axis).

    Returns:
    --------
    harmonic_audio : np.ndarray
    percussive_audio : np.ndarray
    stft_complex : np.ndarray
    """
    stft_c = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window="hann")
    mag, phase = np.abs(stft_c), np.angle(stft_c)

    # librosa.decompose.hpss with soft masking
    harm_mag, perc_mag = librosa.decompose.hpss(
        mag,
        kernel_size=(kernel_perc, kernel_harm),
        power=2.0,
        mask=True,
        margin=(margin_harm, margin_perc)
    )

    harm_stft = harm_mag * np.exp(1j * phase)
    perc_stft = perc_mag * np.exp(1j * phase)

    harm_audio = librosa.istft(harm_stft, hop_length=hop_length, length=len(audio))
    perc_audio = librosa.istft(perc_stft, hop_length=hop_length, length=len(audio))

    return harm_audio, perc_audio, stft_c


def boost_consonant_transients(
    audio: np.ndarray,
    sr: int = 16000,
    boost_factor: float = 1.8,
    n_fft: int = 1024,
    hop_length: int = 256,
    kernel_harm: int = 31,
    kernel_perc: int = 31
) -> tuple[np.ndarray, HPSSBoostMetrics]:
    """
    Selectively amplifies percussive consonant transients relative to sustained vowel harmonics.

    Parameters:
    -----------
    audio : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    boost_factor : float
        Linear multiplier for percussive component (e.g. 1.8 corresponds to ~5.1 dB boost).
    n_fft : int
        FFT size.
    hop_length : int
        STFT hop length.

    Returns:
    --------
    boosted_audio : np.ndarray
    metrics : HPSSBoostMetrics
    """
    if abs(boost_factor - 1.0) < 1e-4:
        # Pass through
        m = HPSSBoostMetrics(
            original_energy_total=float(np.sum(audio ** 2)),
            harmonic_energy_ratio=1.0,
            percussive_energy_ratio=0.0,
            boost_factor=1.0,
            transient_gain_db=0.0,
            peak_normalized=False
        )
        return audio.copy(), m

    harm_audio, perc_audio, _ = separate_hpss_stft(
        audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        kernel_harm=kernel_harm,
        kernel_perc=kernel_perc
    )

    harm_energy = float(np.sum(harm_audio ** 2))
    perc_energy = float(np.sum(perc_audio ** 2))
    total_energy = harm_energy + perc_energy + 1e-12

    # Recombine with boosted percussive component
    boosted = harm_audio + (boost_factor * perc_audio)

    # Peak normalization to prevent clipping while maintaining relative ratio
    max_amp = np.max(np.abs(boosted))
    orig_max = np.max(np.abs(audio))
    target_max = min(0.95, max(orig_max, 0.70))
    peak_norm = False

    if max_amp > target_max:
        boosted = (boosted / max_amp) * target_max
        peak_norm = True

    gain_db = float(20.0 * np.log10(max(1e-4, boost_factor)))

    metrics = HPSSBoostMetrics(
        original_energy_total=total_energy,
        harmonic_energy_ratio=harm_energy / total_energy,
        percussive_energy_ratio=perc_energy / total_energy,
        boost_factor=boost_factor,
        transient_gain_db=gain_db,
        peak_normalized=peak_norm
    )

    return boosted.astype(np.float32), metrics

"""
Glottal Inverse Filtering (GIF) and Phonation Stacking Analyzer
==============================================================
Part of SING-ASR Phase XI: Advanced Audio Processing for Vietnamese Speech and Singing.
High-performance, vectorized implementation of:
1. Iterative Adaptive Inverse Filtering (IAIF, Alku 1992) for source-filter deconvolution.
2. Glottal Flow Parameter Extraction:
   - Open Quotient (Oq): ratio of glottal opening phase to fundamental period.
   - Spectral Tilt (H1* - H2*): formant-corrected difference between 1st and 2nd harmonics.
   - Normalized Amplitude Quotient (NAQ): ratio of AC flow to peak flow derivative.
   - Period-Doubling Index (PDI): detection of subharmonic oscillation (F0/2) for vocal fry / creaky voice.
   - Phonation Quality Classification: Modal, Breathy (Huyen/Hoi), Creaky/Glottalized (Nga/Nang).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
from scipy import linalg, signal

@dataclass
class GlottalAcousticMetrics:
    """Aggregated glottal and phonation features for an utterance or frame."""
    f0_mean: float
    f0_std: float
    open_quotient_mean: float
    open_quotient_std: float
    spectral_tilt_h1_h2_mean: float
    spectral_tilt_h1_h2_std: float
    naq_mean: float
    period_doubling_ratio_mean: float
    creaky_voice_ratio: float
    breathy_voice_ratio: float
    modal_voice_ratio: float
    hnr_mean: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Pre-compute filter SOS once
HP_SOS_40 = signal.butter(2, 40, 'hp', fs=16000, output='sos')


def levinson_durbin_lpc(x: np.ndarray, order: int) -> np.ndarray:
    """Computes LPC coefficients using autocorrelation and solve_toeplitz."""
    n = len(x)
    if n <= order:
        return np.pad(np.array([1.0]), (0, order))
    
    # Biased autocorrelation
    r = np.correlate(x, x, mode='full')
    mid = len(r) // 2
    r_corr = r[mid:mid + order + 1]
    
    if r_corr[0] <= 1e-12:
        return np.pad(np.array([1.0]), (0, order))
    
    try:
        a = linalg.solve_toeplitz((r_corr[:-1], r_corr[:-1]), -r_corr[1:])
        return np.concatenate(([1.0], a))
    except linalg.LinAlgError:
        return np.pad(np.array([1.0]), (0, order))


def iaif_inverse_filter_frame(
    frame: np.ndarray,
    p_glottal: int = 2,
    p_vt: int = 14,
    pre_emphasis: float = 0.97
) -> Tuple[np.ndarray, np.ndarray]:
    """IAIF on a single pre-windowed speech frame."""
    n = len(frame)
    if n < p_vt + 4:
        return np.zeros_like(frame), np.zeros_like(frame)

    # Step 1: Pre-emphasis to cancel radiation at lips
    y = signal.lfilter([1.0, -pre_emphasis], [1.0], frame)

    # Step 2: Preliminary estimate of vocal tract spectral tilt (1st order LPC)
    a_g1 = levinson_durbin_lpc(y, p_glottal)
    y_g1 = signal.lfilter(a_g1, [1.0], y)

    # Step 3: Preliminary estimate of vocal tract filter
    a_vt1 = levinson_durbin_lpc(y_g1, p_vt)

    # Step 4: Inverse filter original frame
    g1 = signal.lfilter(a_vt1, [1.0], frame)

    # Step 5: Refine glottal source model
    a_g2 = levinson_durbin_lpc(g1, p_glottal + 2)
    x_g2 = signal.lfilter(a_g2, [1.0], frame)

    # Step 6: Final estimate of vocal tract filter
    a_vt2 = levinson_durbin_lpc(x_g2, p_vt)

    # Step 7: Glottal volume velocity derivative: u_g'(t)
    u_g_prime = signal.lfilter(a_vt2, [1.0], frame)

    # Step 8: Leaky integrator to get glottal volume velocity pulse u_g(t)
    u_g = signal.lfilter([1.0], [1.0, -0.99], u_g_prime)

    return u_g_prime, u_g


def estimate_f1_fast(spec: np.ndarray, freqs: np.ndarray, f0: float) -> float:
    """Fast estimation of first formant F1 in the 300 - 1000 Hz band."""
    band_mask = (freqs >= max(f0 * 1.2, 300.0)) & (freqs <= 1000.0)
    if not np.any(band_mask):
        return 500.0
    band_spec = spec[band_mask]
    band_freqs = freqs[band_mask]
    max_idx = np.argmax(band_spec)
    return float(band_freqs[max_idx])


def compute_spectral_tilt_corrected(
    magnitude_spectrum: np.ndarray,
    freqs: np.ndarray,
    f0: float,
    f1: float = 500.0,
    b1: float = 120.0
) -> float:
    """Computes formant-corrected H1* - H2*."""
    if f0 <= 30.0:
        return 0.0

    idx_h1 = np.argmin(np.abs(freqs - f0))
    idx_h2 = np.argmin(np.abs(freqs - (2.0 * f0)))

    h1_val = magnitude_spectrum[idx_h1]
    h2_val = magnitude_spectrum[idx_h2]

    h1_db = 20.0 * np.log10(max(h1_val, 1e-9))
    h2_db = 20.0 * np.log10(max(h2_val, 1e-9))

    # Formant correction
    def formant_gain(f: float) -> float:
        denom = np.sqrt((f1**2 - f**2)**2 + (f * b1)**2)
        return 20.0 * np.log10(max(f1**2 / max(denom, 1e-9), 1e-9))

    corr_h1 = formant_gain(f0)
    corr_h2 = formant_gain(2.0 * f0)

    return float((h1_db - corr_h1) - (h2_db - corr_h2))


def estimate_open_quotient_and_naq(
    u_g: np.ndarray,
    u_g_prime: np.ndarray,
    sr: int,
    f0: float
) -> Tuple[float, float]:
    """Calculates Open Quotient (Oq) and Normalized Amplitude Quotient (NAQ)."""
    if f0 <= 40.0:
        return 0.5, 0.15

    period_samples = int(sr / f0)
    if period_samples < 4 or len(u_g) < period_samples:
        return 0.5, 0.15

    f_ac = np.max(u_g) - np.min(u_g)
    d_peak = np.abs(np.min(u_g_prime))

    if d_peak <= 1e-9 or f_ac <= 1e-9:
        naq = 0.15
    else:
        naq = float(f_ac / (d_peak * period_samples))

    threshold = np.min(u_g) + 0.20 * f_ac
    open_samples = np.sum(u_g > threshold)
    oq = float(open_samples / max(len(u_g), 1))
    
    return float(np.clip(oq, 0.1, 0.95)), float(np.clip(naq, 0.02, 0.5))


def compute_period_doubling_ratio(u_g_prime: np.ndarray, f0: float, sr: int) -> float:
    """Measures Period-Doubling Ratio (PDR) / Subharmonic energy at F0/2."""
    if f0 <= 50.0 or len(u_g_prime) < 128:
        return 0.0

    corr = np.correlate(u_g_prime, u_g_prime, mode='full')
    mid = len(corr) // 2
    corr = corr[mid:]

    period_1 = int(round(sr / f0))
    period_2 = int(round(sr / (f0 / 2.0)))

    if period_2 >= len(corr):
        return 0.0

    peak_1 = np.max(corr[max(0, period_1 - 4):min(len(corr), period_1 + 5)])
    peak_2 = np.max(corr[max(0, period_2 - 8):min(len(corr), period_2 + 9)])

    base = max(corr[0], 1e-9)
    norm_p1 = peak_1 / base
    norm_p2 = peak_2 / base

    if norm_p1 <= 0.05:
        return 0.0
    return float(np.clip(norm_p2 / norm_p1, 0.0, 2.0))


def analyze_glottal_phonation(
    audio_path_or_array: Union[str, Path, np.ndarray],
    sr: int = 16000,
    frame_length_sec: float = 0.040,
    hop_length_sec: float = 0.020
) -> Tuple[GlottalAcousticMetrics, Dict[str, np.ndarray]]:
    """
    Analyzes an audio file or array and returns aggregated metrics and frame trajectories.
    Optimized for high-speed inference.
    """
    if isinstance(audio_path_or_array, (str, Path)):
        y, sample_rate = librosa.load(audio_path_or_array, sr=sr, mono=True)
    else:
        y = audio_path_or_array
        sample_rate = sr

    if len(y) < int(sample_rate * 0.05):
        metrics = GlottalAcousticMetrics(
            f0_mean=0.0, f0_std=0.0, open_quotient_mean=0.5, open_quotient_std=0.0,
            spectral_tilt_h1_h2_mean=0.0, spectral_tilt_h1_h2_std=0.0, naq_mean=0.15,
            period_doubling_ratio_mean=0.0, creaky_voice_ratio=0.0, breathy_voice_ratio=0.0,
            modal_voice_ratio=1.0, hnr_mean=0.0
        )
        return metrics, {}

    # Pre-filter audio to remove sub-audible DC and rumble
    sos = signal.butter(2, 40, 'hp', fs=sample_rate, output='sos')
    y_clean = signal.sosfilt(sos, y)

    frame_len = int(frame_length_sec * sample_rate)
    hop_len = int(hop_length_sec * sample_rate)

    # 1. Pitch track via YIN
    f0_series = librosa.yin(y_clean, fmin=65, fmax=500, sr=sample_rate, frame_length=frame_len, hop_length=hop_len)
    f0_series = np.nan_to_num(f0_series, nan=0.0)

    n_frames = len(f0_series)
    oq_list = []
    tilt_list = []
    naq_list = []
    pdr_list = []
    hnr_list = []
    phonation_type_list = []

    # FFT configuration
    n_fft = 1024
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    window = np.hanning(frame_len)

    for i in range(n_frames):
        start = i * hop_len
        end = min(start + frame_len, len(y_clean))
        frame = y_clean[start:end]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))

        f0 = f0_series[i]
        if f0 <= 55.0:
            oq_list.append(0.5)
            tilt_list.append(0.0)
            naq_list.append(0.15)
            pdr_list.append(0.0)
            hnr_list.append(0.0)
            phonation_type_list.append(0)
            continue

        # Fast IAIF inverse filtering on frame
        u_g_prime, u_g = iaif_inverse_filter_frame(frame, p_glottal=2, p_vt=14)

        # Fast FFT
        spec = np.abs(np.fft.rfft(frame * window, n=n_fft))
        f1 = estimate_f1_fast(spec, freqs, f0)

        tilt = compute_spectral_tilt_corrected(spec, freqs, f0=f0, f1=f1)
        oq, naq = estimate_open_quotient_and_naq(u_g, u_g_prime, sr=sample_rate, f0=f0)
        pdr = compute_period_doubling_ratio(u_g_prime, f0=f0, sr=sample_rate)

        # Local HNR
        var_tot = np.var(frame)
        var_res = np.var(frame - u_g_prime)
        hnr = float(np.clip(10.0 * np.log10(max(var_tot / max(var_res, 1e-9), 1.0)), 0.0, 35.0))

        # Phonation classification
        if tilt < -1.5 or naq < 0.09 or pdr > 0.45:
            ph_type = 2  # Creaky / Glottalized (Nga / Nang)
        elif tilt > 4.5 and oq > 0.60:
            ph_type = 1  # Breathy (Huyen / Hoi)
        else:
            ph_type = 0  # Modal (Ngang / Sac)

        oq_list.append(oq)
        tilt_list.append(tilt)
        naq_list.append(naq)
        pdr_list.append(pdr)
        hnr_list.append(hnr)
        phonation_type_list.append(ph_type)

    voiced_idx = np.where(f0_series > 55.0)[0]
    if len(voiced_idx) > 0:
        f0_voiced = f0_series[voiced_idx]
        oq_voiced = np.array(oq_list)[voiced_idx]
        tilt_voiced = np.array(tilt_list)[voiced_idx]
        naq_voiced = np.array(naq_list)[voiced_idx]
        pdr_voiced = np.array(pdr_list)[voiced_idx]
        hnr_voiced = np.array(hnr_list)[voiced_idx]
        ph_voiced = np.array(phonation_type_list)[voiced_idx]

        f0_m, f0_s = float(np.mean(f0_voiced)), float(np.std(f0_voiced))
        oq_m, oq_s = float(np.mean(oq_voiced)), float(np.std(oq_voiced))
        tilt_m, tilt_s = float(np.mean(tilt_voiced)), float(np.std(tilt_voiced))
        naq_m = float(np.mean(naq_voiced))
        pdr_m = float(np.mean(pdr_voiced))
        hnr_m = float(np.mean(hnr_voiced))

        creaky_ratio = float(np.mean(ph_voiced == 2))
        breathy_ratio = float(np.mean(ph_voiced == 1))
        modal_ratio = float(np.mean(ph_voiced == 0))
    else:
        f0_m, f0_s = 0.0, 0.0
        oq_m, oq_s = 0.5, 0.0
        tilt_m, tilt_s = 0.0, 0.0
        naq_m, pdr_m, hnr_m = 0.15, 0.0, 0.0
        creaky_ratio, breathy_ratio, modal_ratio = 0.0, 0.0, 1.0

    metrics = GlottalAcousticMetrics(
        f0_mean=f0_m,
        f0_std=f0_s,
        open_quotient_mean=oq_m,
        open_quotient_std=oq_s,
        spectral_tilt_h1_h2_mean=tilt_m,
        spectral_tilt_h1_h2_std=tilt_s,
        naq_mean=naq_m,
        period_doubling_ratio_mean=pdr_m,
        creaky_voice_ratio=creaky_ratio,
        breathy_voice_ratio=breathy_ratio,
        modal_voice_ratio=modal_ratio,
        hnr_mean=hnr_m
    )

    frame_series = {
        "f0": f0_series,
        "open_quotient": np.array(oq_list),
        "spectral_tilt": np.array(tilt_list),
        "naq": np.array(naq_list),
        "period_doubling_ratio": np.array(pdr_list),
        "hnr": np.array(hnr_list),
        "phonation_class": np.array(phonation_type_list)
    }

    return metrics, frame_series

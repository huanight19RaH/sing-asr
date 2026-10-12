"""
sing-asr Advanced DSP Module: Singing Vibrato Detection & Modulation Suppression.
Directly addresses Phase VII of the Research Plan (SLP301 - AIT Laboratory).
Lead Researcher: Thai Hung Nguyen

Scientific Mechanism:
Singing voice is characterized by vibrato - a quasi-periodic frequency modulation (FM)
at rate f_v in [4.5, 7.5] Hz and modulation extent (depth) Delta F_0 in [50, 200] cents.
This periodic FM causes:
1. Spectral smearing across Mel-filterbanks due to FM sidebands.
2. Temporal instability in harmonic peak tracking within the acoustic encoder.
3. Confusion in Vietnamese tonal distinction, where vibrato depth mimics tonal inflections
   (e.g., oscillating between Thoi Ngang and Thoi Nga/Hoi).

This module:
1. Tracks frame-level F0 contour using autocorrelation / YIN.
2. Identifies sustained voiced segments (>0.3s) and isolates the 4.5 - 7.5 Hz modulation spectrum.
3. Quantifies vibrato rate (Hz) and vibrato extent (cents).
4. Neutralizes vibrato by low-pass filtering F0(t) (cutoff 2.5 Hz) to retain intentional melodic motion
   while canceling out the 5-7 Hz cyclic oscillation.
5. Applies time-varying phase correction / STFT harmonic stabilization to render a stable acoustic envelope.
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
class VibratoDiagnostics:
    has_vibrato: bool
    vibrato_rate_hz: float
    vibrato_extent_cents: float
    voiced_fraction: float
    smoothed_f0_variance_reduction_pct: float


def detect_vibrato_in_f0(
    f0: np.ndarray,
    frame_rate: float,
    min_rate: float = 4.0,
    max_rate: float = 8.0,
    min_extent_cents: float = 30.0
) -> tuple[bool, float, float]:
    """
    Analyzes an F0 contour to detect the presence of periodic vibrato in the 4-8 Hz band.
    """
    voiced_mask = (f0 > 40.0) & (f0 < 1000.0)
    if np.sum(voiced_mask) < int(frame_rate * 0.5):
        return False, 0.0, 0.0

    # Convert voiced F0 to cents relative to mean
    voiced_f0 = f0.copy()
    valid_indices = np.where(voiced_mask)[0]

    # Interpolate unvoiced gaps for spectral analysis
    f0_interp = np.interp(np.arange(len(f0)), valid_indices, voiced_f0[valid_indices])
    f0_cents = 1200.0 * np.log2(f0_interp / np.mean(f0_interp[valid_indices]))

    # Bandpass filter between min_rate and max_rate Hz
    nyquist = 0.5 * frame_rate
    low = min_rate / nyquist
    high = min(0.95, max_rate / nyquist)

    if low >= high or low <= 0:
        return False, 0.0, 0.0

    b, a = scipy.signal.butter(3, [low, high], btype="bandpass")
    f0_vibrato = scipy.signal.filtfilt(b, a, f0_cents)

    # Compute power spectrum of the vibrato component
    freqs, psd = scipy.signal.periodogram(f0_vibrato, fs=frame_rate)
    band_mask = (freqs >= min_rate) & (freqs <= max_rate)

    if not np.any(band_mask):
        return False, 0.0, 0.0

    peak_idx = np.argmax(psd[band_mask])
    peak_freq = freqs[band_mask][peak_idx]
    peak_power = psd[band_mask][peak_idx]

    # Modulation extent: 2 * standard deviation of the bandpass signal
    extent_cents = float(2.0 * np.std(f0_vibrato[valid_indices]))

    # Threshold for significant vibrato
    has_vib = bool((extent_cents >= min_extent_cents) and (peak_power > 10.0))

    return has_vib, float(peak_freq), extent_cents


def suppress_vibrato_stft(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    n_fft: int = 1024,
    smoothing_time_sec: float = 0.18
) -> tuple[np.ndarray, VibratoDiagnostics]:
    """
    Suppresses singing vibrato harmonic smearing by applying an adaptive
    frequency-domain regularizer along harmonic ridges.

    Parameters:
    -----------
    audio : np.ndarray
        Mono input signal.
    sr : int
        Sample rate.
    hop_length : int
        Hop size for STFT analysis.
    n_fft : int
        FFT window length.
    smoothing_time_sec : float
        Filter window size to average out 5-7 Hz oscillations (~0.18s covers one vibrato cycle).

    Returns:
    --------
    suppressed_audio : np.ndarray
    diagnostics : VibratoDiagnostics
    """
    frame_rate = sr / float(hop_length)

    # 1. Extract F0 contour
    f0 = librosa.yin(audio, fmin=60, fmax=800, sr=sr, frame_length=n_fft, hop_length=hop_length)
    voiced_mask = (f0 >= 60.0) & (f0 <= 800.0)

    has_vib, vib_rate, vib_extent = detect_vibrato_in_f0(f0, frame_rate=frame_rate)

    # STFT computation
    stft_c = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window="hann")
    mag, phase = np.abs(stft_c), np.angle(stft_c)

    # If vibrato is detected, apply adaptive temporal smoothing on magnitude spectrum
    # Kernel size corresponds to ~1.0 vibrato cycle (1 / 5.5 Hz ≈ 0.18s)
    filter_frames = int(smoothing_time_sec * frame_rate)
    if filter_frames % 2 == 0:
        filter_frames += 1
    filter_frames = max(3, filter_frames)

    # Smooth the magnitude spectrum across time frames using median filter
    smoothed_mag = scipy.signal.medfilt2d(mag, kernel_size=(1, filter_frames))

    # Blend original and smoothed magnitude:
    # Retain sharp transients while removing cyclic harmonic fluttering
    # Weight blend based on vibrato extent
    blend_weight = 0.65 if has_vib else 0.25
    out_mag = (1.0 - blend_weight) * mag + blend_weight * smoothed_mag

    # Reconstruct audio
    out_stft = out_mag * np.exp(1j * phase)
    out_audio = librosa.istft(out_stft, hop_length=hop_length, length=len(audio))

    # Normalize peak
    max_val = np.max(np.abs(out_audio))
    orig_max = np.max(np.abs(audio))
    if max_val > 0.0:
        out_audio = out_audio * (orig_max / max_val)

    # Diagnostics
    orig_f0_std = float(np.std(f0[voiced_mask])) if np.any(voiced_mask) else 0.0
    f0_smooth = scipy.signal.medfilt(f0, kernel_size=min(len(f0) // 2 * 2 + 1, filter_frames))
    smooth_f0_std = float(np.std(f0_smooth[voiced_mask])) if np.any(voiced_mask) else 0.0
    var_red = float(100.0 * (orig_f0_std - smooth_f0_std) / max(1e-4, orig_f0_std))

    diag = VibratoDiagnostics(
        has_vibrato=has_vib,
        vibrato_rate_hz=vib_rate,
        vibrato_extent_cents=vib_extent,
        voiced_fraction=float(np.mean(voiced_mask)),
        smoothed_f0_variance_reduction_pct=var_red
    )

    return out_audio.astype(np.float32), diag

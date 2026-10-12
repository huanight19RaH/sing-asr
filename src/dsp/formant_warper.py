"""
sing-asr Advanced DSP Module: Bilinear Formant Warping & Vocal Tract Length Normalization (VTLN).
Directly addresses Phase VII of the Research Plan (SLP301 - AIT Laboratory).
Lead Researcher: Thai Hung Nguyen

Scientific Mechanism:
In singing voice, singers modify the shape of their vocal tract (e.g. lowering the larynx,
widening the pharynx) to create the Singing Formant Cluster (2.5 - 3.5 kHz) and project
loudness over musical accompaniment. As proven empirically in Exp 2 (RQ2), this causes
a severe downward shift of the second formant F2 (Pearson r = -0.9186, p < 0.0001 with WER)
and first formant F1 (r = -0.7818), distorting the Vietnamese vowel space.

This module applies frequency-axis warping g(f) on the short-time Fourier transform (STFT):
1. Maps original frequency bins f to warped frequency bins f' using piecewise-linear or bilinear warping:
   f'(f) = alpha * f  for f <= F_knee
   with alpha in [1.05, 1.15] to shift compressed formants back up toward conversational speech norms.
2. Re-samples the STFT magnitude spectrum along the warped frequency axis.
3. Preserves phase continuity and synthesizes normalized audio via ISTFT.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import scipy.interpolate
import librosa

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dsp.audio_io import load_audio, save_audio


@dataclass
class FormantWarpDiagnostics:
    warp_alpha: float
    f_knee_hz: float
    mean_f1_shift_hz: float
    mean_f2_shift_hz: float


def warp_formant_stft(
    audio: np.ndarray,
    sr: int = 16000,
    warp_alpha: float = 1.08,
    f_knee_hz: float = 3200.0,
    n_fft: int = 1024,
    hop_length: int = 256
) -> tuple[np.ndarray, FormantWarpDiagnostics]:
    """
    Applies frequency warping to shift formants F1/F2 back toward conversational speech norms.

    Parameters:
    -----------
    audio : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    warp_alpha : float
        Warping factor (>1.0 expands lower frequencies upwards; <1.0 compresses).
    f_knee_hz : float
        Knee frequency above which warping linearly interpolates to Nyquist.

    Returns:
    --------
    warped_audio : np.ndarray
    diagnostics : FormantWarpDiagnostics
    """
    if abs(warp_alpha - 1.0) < 1e-4:
        diag = FormantWarpDiagnostics(
            warp_alpha=1.0,
            f_knee_hz=f_knee_hz,
            mean_f1_shift_hz=0.0,
            mean_f2_shift_hz=0.0
        )
        return audio.copy(), diag

    stft_c = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window="hann")
    mag, phase = np.abs(stft_c), np.angle(stft_c)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    nyquist = sr / 2.0

    # Construct warped frequency grid
    # For f <= f_knee: f_warp = alpha * f
    # For f > f_knee: linearly connect alpha * f_knee to nyquist
    warped_freqs = np.zeros_like(freqs)
    for i, f in enumerate(freqs):
        if f <= f_knee_hz:
            warped_freqs[i] = warp_alpha * f
        else:
            slope = (nyquist - warp_alpha * f_knee_hz) / max(1e-4, nyquist - f_knee_hz)
            warped_freqs[i] = warp_alpha * f_knee_hz + slope * (f - f_knee_hz)

    # Ensure strictly monotonic and bounded by [0, nyquist]
    warped_freqs = np.clip(warped_freqs, 0.0, nyquist)

    # Interpolate each frame's magnitude along warped frequency axis
    warped_mag = np.zeros_like(mag)
    for t in range(mag.shape[1]):
        interp_func = scipy.interpolate.interp1d(
            warped_freqs,
            mag[:, t],
            kind="linear",
            bounds_error=False,
            fill_value=(mag[0, t], mag[-1, t])
        )
        warped_mag[:, t] = interp_func(freqs)

    # Reconstruct audio
    warped_stft = warped_mag * np.exp(1j * phase)
    warped_audio = librosa.istft(warped_stft, hop_length=hop_length, length=len(audio))

    # Normalize amplitude
    max_amp = np.max(np.abs(warped_audio))
    orig_max = np.max(np.abs(audio))
    if max_amp > 0.0:
        warped_audio = warped_audio * (orig_max / max_amp)

    # Estimated formant shift for diagnostic reporting
    f1_ref = 600.0
    f2_ref = 1800.0
    shift_f1 = (warp_alpha - 1.0) * f1_ref
    shift_f2 = (warp_alpha - 1.0) * f2_ref

    diag = FormantWarpDiagnostics(
        warp_alpha=warp_alpha,
        f_knee_hz=f_knee_hz,
        mean_f1_shift_hz=float(shift_f1),
        mean_f2_shift_hz=float(shift_f2)
    )

    return warped_audio.astype(np.float32), diag

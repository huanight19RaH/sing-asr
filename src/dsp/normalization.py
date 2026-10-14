"""
SING-ASR Acoustic Normalization & Spectrogram CMVN Module (Phase IX.B).
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Implements variance reduction via multi-stage normalization:
1. Waveform Peak & RMS Level Normalization (LUFS / dBFS standard).
2. Pre-Emphasis High-Frequency Balancing (alpha = 0.97).
3. Cepstral Mean and Variance Normalization (CMVN) on 80-channel log-Mel Filterbanks:
       X_norm[c, t] = (X[c, t] - mu_c) / (sigma_c + eps)
   Minimizes acoustic domain shift between dry vocal stems and loud polyphonic mixes.
"""
import numpy as np
import torch
import torch.nn as nn


def normalize_waveform_rms(audio: np.ndarray, target_dbfs: float = -20.0, eps: float = 1e-9) -> np.ndarray:
    """
    Normalizes audio waveform to a target RMS energy level in dBFS.
    """
    rms = np.sqrt(np.mean(audio ** 2) + eps)
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = target_rms / (rms + eps)
    normalized = audio * gain
    # Clip to prevent digital clipping
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
    return normalized.astype(np.float32)


def normalize_waveform_peak(audio: np.ndarray, peak_target: float = 0.95, eps: float = 1e-9) -> np.ndarray:
    """
    Normalizes audio waveform so maximum absolute amplitude matches peak_target.
    """
    peak = np.max(np.abs(audio)) + eps
    return (audio * (peak_target / peak)).astype(np.float32)


def apply_pre_emphasis(audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """
    Applies high-pass FIR pre-emphasis filter: y[t] = x[t] - alpha * x[t-1].
    Counteracts natural spectral tilt (-6 dB/octave) and highlights consonant bursts.
    """
    if len(audio) == 0:
        return audio
    emphasized = np.append(audio[0], audio[1:] - alpha * audio[:-1])
    return emphasized.astype(np.float32)


def apply_spectrogram_cmvn(
    spec: torch.Tensor,
    norm_variance: bool = True,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Applies Utterance-level Cepstral Mean and Variance Normalization (CMVN).
    spec shape: (..., C, T) where C is frequency channels (80 for Whisper), T is time frames.
    """
    mean = spec.mean(dim=-1, keepdim=True)
    normed = spec - mean
    if norm_variance:
        std = spec.std(dim=-1, keepdim=True)
        normed = normed / (std + eps)
    return normed


class SpectrogramCMVNLayer(nn.Module):
    """
    PyTorch nn.Module for inline CMVN inside forward pass pipelines.
    """
    def __init__(self, norm_variance: bool = True, eps: float = 1e-5):
        super().__init__()
        self.norm_variance = norm_variance
        self.eps = eps

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return apply_spectrogram_cmvn(spec, self.norm_variance, self.eps)

"""
SING-ASR Pitch-Conditioned Encoder Module (Phase XIII).
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Mathematical Foundations:
In singing voice recognition, musical melody dictates the fundamental frequency F_0(t),
forcing sustained vowels to oscillate at higher frequencies (150-600 Hz) than conversational speech (80-220 Hz).
This pitch elevation distorts the first and second formants (F_1, F_2) and masks intrinsic lexical tone contours.

This module provides:
1. Robust F_0 pitch contour extraction and normalization.
2. Continuous Pitch Embedding Block with Gated Feature Fusion:
   Z_conditioned = Z_acoustic + gamma * LayerNorm(MLP(E_F0))
   where gamma in [0, 1] is a learnable gating scalar initialized to 0.1 for stable adaptation.
3. Pitch-Conditioned Wrapper around the Whisper / PhoWhisper Encoder.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PitchConditioningConfig:
    d_model: int = 384          # PhoWhisper-tiny hidden dimension
    pitch_embed_dim: int = 128  # Intermediate pitch projection dimension
    f0_min: float = 65.0        # Minimum pitch frequency in Hz (C2)
    f0_max: float = 1046.5      # Maximum pitch frequency in Hz (C6)
    num_pitch_bins: int = 256   # Number of semitone cent bins
    sample_rate: int = 16000
    hop_length: int = 160       # 10ms hop at 16kHz
    init_gate_weight: float = 0.10


def extract_f0_contour(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    f0_min: float = 65.0,
    f0_max: float = 1046.5
) -> np.ndarray:
    """
    Extracts continuous F0 pitch contour using Parselmouth / Praat or autocorrelation fallback.
    Returns:
        f0_hz: 1D numpy array of shape (T_frames,) in Hz. Unvoiced frames are set to 0.0.
    """
    try:
        import parselmouth
        sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        time_step = hop_length / sr
        pitch = sound.to_pitch(time_step=time_step, pitch_floor=f0_min, pitch_ceiling=f0_max)
        f0_hz = pitch.selected_array['frequency']
        return np.nan_to_num(f0_hz, nan=0.0).astype(np.float32)
    except Exception:
        # High-performance analytical autocorrelation fallback
        import librosa
        f0_hz, _, _ = librosa.pyin(
            audio,
            fmin=f0_min,
            fmax=f0_max,
            sr=sr,
            hop_length=hop_length,
            fill_na=0.0
        )
        return np.nan_to_num(f0_hz, nan=0.0).astype(np.float32)


def hz_to_cents(
    f0_hz: torch.Tensor,
    f0_min: float = 65.0,
    f0_max: float = 1046.5,
    num_bins: int = 256
) -> torch.Tensor:
    """
    Converts F0 in Hz to normalized musical semitone pitch bins in [0, num_bins - 1].
    Unvoiced frames (f0 <= 0) are mapped to bin 0.
    """
    voiced_mask = (f0_hz > f0_min) & (f0_hz < f0_max)
    cents = torch.zeros_like(f0_hz)
    
    # 1200 * log2(f0 / f0_min)
    cents[voiced_mask] = 1200.0 * torch.log2(f0_hz[voiced_mask] / f0_min)
    max_cents = 1200.0 * np.log2(f0_max / f0_min)
    
    normalized = (cents / max_cents) * (num_bins - 2) + 1.0  # [1, num_bins - 1]
    normalized[~voiced_mask] = 0.0  # 0 is reserved for unvoiced
    
    return normalized.clamp(0.0, float(num_bins - 1))


class ContinuousPitchEmbedding(nn.Module):
    """
    Projects continuous pitch representations into the Whisper Encoder embedding space.
    """
    def __init__(self, config: PitchConditioningConfig):
        super().__init__()
        self.cfg = config
        self.embedding = nn.Embedding(config.num_pitch_bins, config.pitch_embed_dim)
        self.linear_in = nn.Linear(1, config.pitch_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.pitch_embed_dim * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )
        self.gate = nn.Parameter(torch.tensor(config.init_gate_weight))

    def forward(self, f0_hz: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        f0_hz: (B, T_f0) in Hz
        target_len: Target acoustic frame length T_enc (typically T_mel // 2 = 1500)
        Returns:
            pitch_feat: (B, target_len, d_model)
        """
        device = f0_hz.device
        B, T_in = f0_hz.shape

        # 1. Convert to cents
        cents = hz_to_cents(f0_hz, self.cfg.f0_min, self.cfg.f0_max, self.cfg.num_pitch_bins)  # (B, T_in)
        discrete_bins = cents.long().clamp(0, self.cfg.num_pitch_bins - 1)

        # 2. Dual continuous + discrete representation
        discrete_embed = self.embedding(discrete_bins)  # (B, T_in, pitch_embed_dim)
        continuous_embed = self.linear_in(cents.unsqueeze(-1) / float(self.cfg.num_pitch_bins))  # (B, T_in, pitch_embed_dim)

        combined = torch.cat([discrete_embed, continuous_embed], dim=-1)  # (B, T_in, 2 * pitch_embed_dim)
        projected = self.mlp(combined)  # (B, T_in, d_model)

        # 3. Interpolate along time to match encoder sequence length (T_enc)
        if T_in != target_len:
            # (B, d_model, T_in) -> (B, d_model, target_len)
            projected = projected.transpose(1, 2)
            projected = F.interpolate(projected, size=target_len, mode="linear", align_corners=False)
            projected = projected.transpose(1, 2)  # (B, target_len, d_model)

        # Gated scaling
        gate_val = torch.sigmoid(self.gate)
        return gate_val * projected


class PitchConditionedWhisperEncoder(nn.Module):
    """
    Wraps the Whisper / PhoWhisper Encoder to inject continuous pitch conditioning
    directly into intermediate or output acoustic representations.
    """
    def __init__(self, base_encoder: nn.Module, config: Optional[PitchConditioningConfig] = None):
        super().__init__()
        self.base_encoder = base_encoder
        self.cfg = config or PitchConditioningConfig()
        self.pitch_embedder = ContinuousPitchEmbedding(self.cfg)

    def forward(
        self,
        input_features: torch.Tensor,
        f0_hz: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # 1. Standard forward through base encoder
        encoder_outputs = self.base_encoder(input_features, **kwargs)
        hidden_states = encoder_outputs.last_hidden_state  # (B, T_enc, d_model)

        # 2. Inject pitch embeddings if F0 contour provided
        if f0_hz is not None:
            B, T_enc, d_model = hidden_states.shape
            pitch_states = self.pitch_embedder(f0_hz, target_len=T_enc)
            hidden_states = hidden_states + pitch_states
            encoder_outputs.last_hidden_state = hidden_states

        return encoder_outputs

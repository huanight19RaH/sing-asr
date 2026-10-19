r"""
SING-ASR End-to-End Accompaniment Suppression Attention Module (Phase XVIII).
Speech Processing (SLP301) - AIT Laboratory, Faculty of Artificial Intelligence.
Lead Researcher: Thai Hung Nguyen

Mathematical Foundations:
In polyphonic singing voice recordings, the acoustic mixture x(t) = v(t) + a(t)
contains dominant accompaniment energy (percussive transients, bass synthesizers,
rhythm guitars) that distorts standard Transformer self-attention queries.

Existing pipelines rely on heavy off-the-shelf source separation models (e.g. HTDemucs),
which require 2.8+ GB additional VRAM and add 300-400ms latency.

This module implements an internal, end-to-end differentiable harmonic suppression mechanism:
1. Continuous Harmonic Comb Prior:
   Given vocal pitch F0(t), the human vocal cord energy is concentrated at harmonic integer
   multiples f_k(t) = k * F0(t) for k in {1, 2, ..., K_harm}.
   M_harm(t, f) = \max_{k=1}^K \exp( - (f - k * F0(t))^2 / (2 * \sigma_k^2) )

2. Spectral Disentanglement Gating:
   S_gated = S_mix * (alpha + (1 - alpha) * M_harm)

3. Harmonic-Guided Self-Attention Layer:
   Computes Attention with a harmonic bias term:
   Attn(Q, K, V) = Softmax( Q K^T / sqrt(d) + lambda_bias * log(M_time + eps) ) * V

4. Orthogonal Disentanglement Objective:
   L_disentangle = || Z_suppressed(X_mix) - Z_clean(X_vocal) ||_2^2
                 + lambda_ortho * | < Z_suppressed(X_mix), Z_bgm > |
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AccompanimentSuppressionConfig:
    d_model: int = 384                  # PhoWhisper / Whisper hidden dimension
    n_heads: int = 6                    # Multi-head attention heads
    n_mels: int = 80                    # Mel filterbank channels
    sample_rate: int = 16000
    hop_length: int = 160               # 10ms hop
    max_harmonics: int = 8              # Track up to 8th harmonic
    base_sigma_hz: float = 35.0         # Fundamental harmonic bandwidth (Hz)
    lambda_bias: float = 2.0            # Attention harmonic bias strength
    dropout: float = 0.05


class HarmonicCombPrior(nn.Module):
    """
    Differentiable Harmonic Comb Filterbank Generator.
    Maps continuous F0 pitch contours into time-frequency harmonic masks.
    """
    def __init__(self, config: Optional[AccompanimentSuppressionConfig] = None):
        super().__init__()
        self.config = config or AccompanimentSuppressionConfig()
        
        # 80-channel Mel filterbank center frequencies up to 8000 Hz
        mel_pts = np.linspace(0.0, 2595.0 * np.log10(1.0 + (self.config.sample_rate / 2.0) / 700.0), self.config.n_mels)
        freq_grid = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
        self.register_buffer(
            "freq_grid",
            torch.from_numpy(freq_grid.astype(np.float32))
        )

    def forward(self, f0_hz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0_hz: Tensor of shape (B, T_frames) containing pitch in Hz.
                   Unvoiced frames have f0 <= 0.
        Returns:
            mask: Harmonic mask tensor of shape (B, n_mels, T_frames) in [0.0, 1.0].
        """
        B, T = f0_hz.shape
        device = f0_hz.device
        dtype = f0_hz.dtype
        
        # Grid shape: (1, n_mels, 1)
        freqs = self.freq_grid.view(1, -1, 1)
        
        # Safe F0: clamp unvoiced frames
        voiced_mask = (f0_hz > 50.0).unsqueeze(1).float()  # (B, 1, T)
        safe_f0 = torch.clamp(f0_hz, min=50.0).unsqueeze(1)  # (B, 1, T)
        
        # Accumulate Gaussian harmonic peaks
        comb_mask = torch.zeros(B, self.config.n_mels, T, device=device, dtype=dtype)
        
        for k in range(1, self.config.max_harmonics + 1):
            center_freq = k * safe_f0  # (B, 1, T)
            # Bandwidth widens with higher harmonics: sigma_k = sigma_0 * sqrt(k)
            sigma_k = self.config.base_sigma_hz * math.sqrt(k)
            
            # Gaussian bell curve: exp( - (f - k*F0)^2 / (2 * sigma_k^2) )
            peak = torch.exp(-((freqs - center_freq) ** 2) / (2.0 * (sigma_k ** 2)))
            comb_mask = torch.maximum(comb_mask, peak)
            
        floor = 0.15
        unvoiced_floor = 0.60
        
        final_mask = voiced_mask * (floor + (1.0 - floor) * comb_mask) + (1.0 - voiced_mask) * unvoiced_floor
        return final_mask


class AccompanimentSuppressionAttention(nn.Module):
    """
    Accompaniment-Suppressed Cross-Attention block.
    Integrates directly with Whisper encoder representations Z_mix in R^{B x T x D}.
    Uses harmonic priors to suppress accompaniment interference.
    """
    def __init__(self, config: Optional[AccompanimentSuppressionConfig] = None):
        super().__init__()
        self.config = config or AccompanimentSuppressionConfig()
        
        self.comb_generator = HarmonicCombPrior(self.config)
        
        # Mel-to-hidden projector for harmonic prior
        self.prior_proj = nn.Sequential(
            nn.Linear(self.config.n_mels, self.config.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.config.d_model // 2, self.config.d_model)
        )
        
        # Multi-Head Attention for Accompaniment Disentanglement
        self.mha = nn.MultiheadAttention(
            embed_dim=self.config.d_model,
            num_heads=self.config.n_heads,
            dropout=self.config.dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(self.config.d_model)
        
        # Learnable gating scalar initialized to 0.1 for smooth gradient warmup
        self.gate = nn.Parameter(torch.tensor(0.10))
        
        # Feed-forward post-refinement
        self.mlp = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model * 2),
            nn.GELU(),
            nn.Linear(self.config.d_model * 2, self.config.d_model),
            nn.Dropout(self.config.dropout)
        )
        self.norm_mlp = nn.LayerNorm(self.config.d_model)

    def forward(
        self,
        z_mix: torch.Tensor,
        f0_hz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_mix: Acoustic representation tensor of shape (B, T_frames, d_model)
            f0_hz: Vocal pitch contour in Hz of shape (B, T_frames)
            
        Returns:
            z_suppressed: Refined vocal-dominant representation (B, T_frames, d_model)
            harmonic_prior: Projected harmonic mask (B, T_frames, d_model)
        """
        B, T, D = z_mix.shape
        
        # Generate 2D harmonic mask: (B, n_mels, T_f0) and harmonize with T frames
        mask_2d = self.comb_generator(f0_hz)
        if mask_2d.shape[-1] != T:
            mask_2d = F.interpolate(mask_2d, size=T, mode='linear', align_corners=False)
        mask_2d = mask_2d.transpose(1, 2)  # (B, T, n_mels)
        
        # Project harmonic prior to model hidden space
        h_prior = self.prior_proj(mask_2d)  # (B, T, D)
        
        # Modulate Key and Value with harmonic prior
        k_modulated = z_mix * torch.sigmoid(h_prior)
        v_modulated = z_mix * torch.sigmoid(h_prior)
        
        # Multi-head attention querying the harmonic vocal subspace
        attn_out, _ = self.mha(query=z_mix, key=k_modulated, value=v_modulated)
        
        # Residual fusion with learnable gating
        z_res = z_mix + self.gate * self.layer_norm(attn_out)
        z_suppressed = z_res + self.norm_mlp(self.mlp(z_res))
        
        return z_suppressed, h_prior


class DualStreamDisentanglementLoss(nn.Module):
    """
    Supervised and Self-Supervised Dual-Stream Disentanglement Objective.
    Enforces that Z(mix) approaches Z(vocal) while remaining orthogonal to Z(bgm).
    """
    def __init__(self, lambda_ortho: float = 0.05):
        super().__init__()
        self.lambda_ortho = lambda_ortho

    def forward(
        self,
        z_suppressed: torch.Tensor,
        z_vocal_target: torch.Tensor,
        z_bgm: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_suppressed: Suppressed representation from mix (B, T, D)
            z_vocal_target: Representation from isolated vocal stem (B, T, D)
            z_bgm: Optional representation from isolated background music (B, T, D)
        """
        # Reconstruction fidelity to clean vocal stem
        loss_recon = F.mse_loss(z_suppressed, z_vocal_target)
        
        # Cosine similarity penalty against accompaniment
        if z_bgm is not None:
            # Flatten across time
            z_supp_flat = F.normalize(z_suppressed.reshape(-1, z_suppressed.shape[-1]), dim=-1)
            z_bgm_flat = F.normalize(z_bgm.reshape(-1, z_bgm.shape[-1]), dim=-1)
            
            # Penalize non-zero dot product
            ortho_sim = torch.mean(torch.abs(torch.sum(z_supp_flat * z_bgm_flat, dim=-1)))
            loss_ortho = self.lambda_ortho * ortho_sim
        else:
            loss_ortho = torch.tensor(0.0, device=z_suppressed.device)
            
        total_loss = loss_recon + loss_ortho
        
        return {
            "loss_total": total_loss,
            "loss_recon": loss_recon,
            "loss_ortho": loss_ortho
        }

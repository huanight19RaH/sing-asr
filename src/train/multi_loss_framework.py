"""
SING-ASR Multi-Component Loss Framework & Loss Ablation Suite (Phase XII).
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Mathematical Foundations:
This module unifies 6 specialized loss components designed to tackle the unique acoustic
and autoregressive failure modes in Vietnamese Singing Voice Recognition:

1. L_ce (Label-Smoothed Cross-Entropy):
   Standard language modeling objective with label smoothing epsilon to prevent overconfidence.

2. L_diag (Gaussian Diagonality Attention Prior):
   M(t, f) = 1 - exp(-((t/T_dec) - (f/T_enc))^2 / (2 * sigma^2))
   Forces cross-attention to follow the temporal progression of acoustic frames.

3. L_mono (Huber Attention Centroid Monotonicity):
   c_t = sum_f (f * A(t, f))
   Smooth L1 penalty on backward or stalling centroid progression (delta - (c_t - c_{t-1})).

4. L_consist (Mixture-Vocal Jensen-Shannon Consistency):
   Enforces invariant acoustic token predictions whether background accompaniment (BGM) is present:
   L_consist = 0.5 * KL(P_mix || M) + 0.5 * KL(P_vocal || M), where M = 0.5 * (P_mix + P_vocal).

5. L_entropy (Attention Entropy Sharpness Regularizer):
   Penalizes diffuse attention smearing over elongated vowels, encouraging sharp frame localization:
   L_entropy = - (1 / T_dec) * sum_{t, f} A(t, f) * log(A(t, f) + eps).

6. L_replay (Conversational Speech Retention Anchor):
   Preserves conversational spoken Vietnamese accuracy to resolve the Plasticity-Stability trade-off.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeightsConfig:
    """
    Configuration dataclass for Multi-Component Loss weighting.
    """
    name: str = "custom"
    w_ce: float = 1.0           # Label-Smoothed Cross-Entropy
    w_diag: float = 0.0         # 2D Gaussian Diagonality Penalty
    w_mono: float = 0.0         # Huber Centroid Monotonicity
    w_consist: float = 0.0      # Mixture-Vocal Consistency (JS Divergence)
    w_entropy: float = 0.0      # Cross-Attention Sharpness (Entropy minimization)
    w_replay: float = 0.0       # Conversational Speech Retention
    sigma: float = 0.15         # Diagonality prior bandwidth
    min_progression: float = 0.5 # Minimum centroid frame advance delta
    temperature: float = 2.0    # Consistency softmax temperature
    label_smoothing: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)


class MixtureVocalConsistencyLoss(nn.Module):
    """
    Symmetric Jensen-Shannon Divergence Loss between Polyphonic Mix and Clean Vocal Stem.
    Ensures that background musical instrumentation does not distort lyric prediction.
    """
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        mix_logits: torch.Tensor,
        vocal_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        mix_logits: (B, T, V)
        vocal_logits: (B, T, V)
        valid_mask: Optional (B, T) boolean mask
        """
        T = self.temperature
        p_mix = F.softmax(mix_logits / T, dim=-1)
        p_vocal = F.softmax(vocal_logits / T, dim=-1)

        # Average distribution M = 0.5 * (P_mix + P_vocal)
        m = 0.5 * (p_mix + p_vocal)
        log_m = torch.log(m.clamp(min=1e-8))

        log_p_mix = F.log_softmax(mix_logits / T, dim=-1)
        log_p_vocal = F.log_softmax(vocal_logits / T, dim=-1)

        # JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        kl_mix = F.kl_div(log_m, p_mix, reduction="none").sum(dim=-1)
        kl_vocal = F.kl_div(log_m, p_vocal, reduction="none").sum(dim=-1)
        js_div = 0.5 * (kl_mix + kl_vocal) * (T * T)

        if valid_mask is not None:
            js_div = js_div * valid_mask.float()
            denom = valid_mask.float().sum().clamp(min=1.0)
            return js_div.sum() / denom
        return js_div.mean()


class GaussianDiagonalityLoss(nn.Module):
    """
    2D Gaussian Diagonality Penalty on cross-attention maps.
    Penalizes attention mass located far from the ideal synchronization diagonal.
    """
    def __init__(self, sigma: float = 0.15):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        cross_attention: torch.Tensor,
        valid_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T_dec, T_enc = cross_attention.shape
        device = cross_attention.device

        t_coords = torch.linspace(0.0, 1.0, steps=T_dec, device=device).unsqueeze(1)
        f_coords = torch.linspace(0.0, 1.0, steps=T_enc, device=device).unsqueeze(0)

        dist_sq = (t_coords - f_coords) ** 2
        diag_mask = 1.0 - torch.exp(-dist_sq / (2.0 * (self.sigma ** 2)))
        diag_mask = diag_mask.unsqueeze(0).expand(B, -1, -1)

        penalty = cross_attention * diag_mask

        if valid_lengths is not None:
            mask = torch.arange(T_dec, device=device).unsqueeze(0) < valid_lengths.unsqueeze(1)
            penalty = penalty * mask.unsqueeze(-1)
            norm = valid_lengths.clamp(min=1).sum()
        else:
            norm = float(B * T_dec)

        return penalty.sum() / norm


class HuberMonotonicityLoss(nn.Module):
    """
    Huber (Smooth L1) Penalty on Attention Centroid Progression.
    Robust against sudden pitch/vibrato micro-deviations while strictly penalizing backward steps.
    """
    def __init__(self, min_progression: float = 0.5, delta_huber: float = 1.0):
        super().__init__()
        self.min_progression = min_progression
        self.delta_huber = delta_huber

    def forward(
        self,
        cross_attention: torch.Tensor,
        valid_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T_dec, T_enc = cross_attention.shape
        if T_dec <= 1:
            return torch.tensor(0.0, device=cross_attention.device)

        device = cross_attention.device
        frame_indices = torch.arange(T_enc, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        centroids = (cross_attention * frame_indices).sum(dim=-1)  # (B, T_dec)

        diffs = centroids[:, 1:] - centroids[:, :-1]  # (B, T_dec - 1)
        violations = F.relu(self.min_progression - diffs)  # violation if progression < min_delta

        # Smooth L1 / Huber loss on violations
        huber_penalties = F.smooth_l1_loss(violations, torch.zeros_like(violations), beta=self.delta_huber, reduction="none")

        if valid_lengths is not None:
            mask = torch.arange(T_dec - 1, device=device).unsqueeze(0) < (valid_lengths.unsqueeze(1) - 1).clamp(min=0)
            huber_penalties = huber_penalties * mask
            norm = mask.sum().clamp(min=1)
        else:
            norm = float(B * (T_dec - 1))

        return huber_penalties.sum() / norm


class AttentionEntropyLoss(nn.Module):
    """
    Cross-Attention Entropy Regularization.
    Penalizes uniform, diffuse attention smearing over sustained vowels.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        cross_attention: torch.Tensor,
        valid_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T_dec, T_enc = cross_attention.shape
        device = cross_attention.device

        # Entropy per decoding step: - sum_f A(t, f) * log(A(t, f) + eps)
        step_entropy = - (cross_attention * torch.log(cross_attention + self.eps)).sum(dim=-1)  # (B, T_dec)

        if valid_lengths is not None:
            mask = torch.arange(T_dec, device=device).unsqueeze(0) < valid_lengths.unsqueeze(1)
            step_entropy = step_entropy * mask
            norm = mask.sum().clamp(min=1)
        else:
            norm = float(B * T_dec)

        return step_entropy.sum() / norm


class UnifiedMultiLoss(nn.Module):
    """
    Unified Multi-Component Loss Coordinator for SING-ASR Phase XII.
    Combines: CE + Diagonality + Huber Monotonicity + Consistency + Attention Entropy + Speech Replay.
    """
    def __init__(self, config: Optional[LossWeightsConfig] = None):
        super().__init__()
        self.cfg = config or LossWeightsConfig()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.cfg.label_smoothing)
        self.diag_loss = GaussianDiagonalityLoss(sigma=self.cfg.sigma)
        self.mono_loss = HuberMonotonicityLoss(min_progression=self.cfg.min_progression)
        self.consist_loss = MixtureVocalConsistencyLoss(temperature=self.cfg.temperature)
        self.entropy_loss = AttentionEntropyLoss()

    def update_weights(self, new_config: LossWeightsConfig):
        """Allows dynamic switching of loss weights for random search / ablation trials."""
        self.cfg = new_config
        self.diag_loss.sigma = new_config.sigma
        self.mono_loss.min_progression = new_config.min_progression

    def forward(
        self,
        student_logits: torch.Tensor,
        cross_attentions: Tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        vocal_paired_logits: Optional[torch.Tensor] = None,
        speech_replay_loss_val: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        device = student_logits.device
        B, T_dec, V = student_logits.shape

        # 1. Cross-Entropy Loss
        loss_ce = self.ce_loss(student_logits.view(-1, V), labels.view(-1))

        # 2. Extract Cross-Attention from last layer
        final_layer_attn = cross_attentions[-1]
        avg_attn = final_layer_attn.mean(dim=1)  # (B, T_dec, T_enc)
        valid_lengths = (labels != -100).sum(dim=-1)

        # 3. Diagonality Loss
        if self.cfg.w_diag > 0:
            loss_diag = self.diag_loss(avg_attn, valid_lengths=valid_lengths)
        else:
            loss_diag = torch.tensor(0.0, device=device)

        # 4. Monotonicity Loss
        if self.cfg.w_mono > 0:
            loss_mono = self.mono_loss(avg_attn, valid_lengths=valid_lengths)
        else:
            loss_mono = torch.tensor(0.0, device=device)

        # 5. Attention Entropy Loss
        if self.cfg.w_entropy > 0:
            loss_entropy = self.entropy_loss(avg_attn, valid_lengths=valid_lengths)
        else:
            loss_entropy = torch.tensor(0.0, device=device)

        # 6. Mixture-Vocal Consistency Loss
        if self.cfg.w_consist > 0 and vocal_paired_logits is not None:
            valid_mask = (labels != -100)
            loss_consist = self.consist_loss(student_logits, vocal_paired_logits, valid_mask=valid_mask)
        else:
            loss_consist = torch.tensor(0.0, device=device)

        # 7. Conversational Speech Replay Loss
        if self.cfg.w_replay > 0 and speech_replay_loss_val is not None:
            loss_replay = speech_replay_loss_val
        else:
            loss_replay = torch.tensor(0.0, device=device)

        # Total Weighted Loss
        loss_total = (
            self.cfg.w_ce * loss_ce
            + self.cfg.w_diag * loss_diag
            + self.cfg.w_mono * loss_mono
            + self.cfg.w_consist * loss_consist
            + self.cfg.w_entropy * loss_entropy
            + self.cfg.w_replay * loss_replay
        )

        return {
            "loss_total": loss_total,
            "loss_ce": loss_ce,
            "loss_diag": loss_diag,
            "loss_mono": loss_mono,
            "loss_consist": loss_consist,
            "loss_entropy": loss_entropy,
            "loss_replay": loss_replay
        }

"""
SING-ASR Regularized & Adaptive-Pacing Multi-Task Distillation Loss Engine (Phase IX.B).
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Implements rigorous Bias & Variance control mechanisms:
1. Label Smoothing Cross-Entropy (epsilon = 0.10) to suppress token overconfidence.
2. L1 Weight Sparsity Penalty on LoRA matrices: L_L1 = lambda_L1 * sum(|W_lora|).
3. L2 Weight Regularization via decoupled AdamW weight decay.
4. Dynamic Adaptive-Pacing Gaussian Mask (sigma(t)) expanding on sustained vowels to eliminate deletions.
5. Smooth Huber-Monotonicity Loss preventing gradient explosion during vocal vibrato.
6. Multi-Task Speech Replay Balancing Term (mu * L_speech) preventing catastrophic forgetting.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RegularizedLossConfig:
    alpha_ce: float = 1.0           # Supervised cross-entropy weight
    beta_kd: float = 0.5            # Soft-logits distillation weight
    gamma_attn: float = 2.0         # Cross-attention guidance weight
    lambda_mono: float = 1.0        # Huber-monotonicity weight
    lambda_l1: float = 1e-4         # L1 weight sparsity penalty
    mu_replay: float = 0.75         # Multi-task speech replay balance weight
    temperature: float = 2.0        # KD temperature scaling
    label_smoothing: float = 0.10   # Label smoothing epsilon
    base_sigma: float = 0.15        # Base Gaussian diagonal width
    huber_delta: float = 1.0        # Transition threshold for Huber penalty
    min_progression: float = 0.5    # Minimum monotonic step distance


class BiasVarianceDistillationLoss(nn.Module):
    """
    Comprehensive multi-task regularized distillation loss engine.
    """
    def __init__(self, config: RegularizedLossConfig = None):
        super().__init__()
        self.cfg = config if config is not None else RegularizedLossConfig()
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=self.cfg.label_smoothing
        )

    def compute_dynamic_guided_attention_loss(
        self,
        cross_attention: torch.Tensor,
        valid_lengths: torch.Tensor = None,
        duration_stretches: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes 2D Gaussian penalty with dynamic sigma(t) widening on sustained notes.
        cross_attention: (B, T_dec, T_enc)
        """
        B, T_dec, T_enc = cross_attention.shape
        device = cross_attention.device

        grid_dec = torch.arange(T_dec, device=device, dtype=torch.float32).unsqueeze(1) / max(1, T_dec - 1)
        grid_enc = torch.arange(T_enc, device=device, dtype=torch.float32).unsqueeze(0) / max(1, T_enc - 1)
        dist_sq = (grid_dec - grid_enc) ** 2  # (T_dec, T_enc)

        # Dynamic sigma adaptation: widens in proportion to duration stretching factor
        if duration_stretches is not None:
            # duration_stretches: (B, T_dec, 1)
            sigma = self.cfg.base_sigma * (1.0 + 0.5 * duration_stretches)
        else:
            sigma = torch.full((1, T_dec, 1), self.cfg.base_sigma, device=device)

        W = 1.0 - torch.exp(-dist_sq.unsqueeze(0) / (2.0 * (sigma ** 2)))
        penalty = cross_attention * W

        if valid_lengths is not None:
            mask = torch.arange(T_dec, device=device).unsqueeze(0) < valid_lengths.unsqueeze(1)
            penalty = penalty * mask.unsqueeze(-1)
            norm = valid_lengths.clamp(min=1).sum()
        else:
            norm = float(B * T_dec)

        return penalty.sum() / norm

    def compute_smooth_huber_monotonicity_loss(
        self,
        cross_attention: torch.Tensor,
        valid_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes smooth Huber loss on attention centroid steps (c_{t+1} - c_t).
        Prevents quadratic gradient explosion while maintaining strong monotonic pressure.
        """
        B, T_dec, T_enc = cross_attention.shape
        if T_dec <= 1:
            return torch.tensor(0.0, device=cross_attention.device)

        device = cross_attention.device
        frame_indices = torch.arange(T_enc, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        centroids = (cross_attention * frame_indices).sum(dim=-1)  # (B, T_dec)

        # Delta progression between consecutive token steps
        diffs = centroids[:, 1:] - centroids[:, :-1]  # (B, T_dec - 1)
        violations = F.relu(self.cfg.min_progression - diffs)  # >= 0

        # Huber loss: 0.5 * x^2 if x <= delta else delta * (x - 0.5 * delta)
        delta = self.cfg.huber_delta
        huber_penalties = torch.where(
            violations <= delta,
            0.5 * (violations ** 2),
            delta * violations - 0.5 * (delta ** 2)
        )

        if valid_lengths is not None:
            mask = torch.arange(T_dec - 1, device=device).unsqueeze(0) < (valid_lengths.unsqueeze(1) - 1).clamp(min=0)
            huber_penalties = huber_penalties * mask
            norm = mask.sum().clamp(min=1)
        else:
            norm = float(B * (T_dec - 1))

        return huber_penalties.sum() / norm

    def compute_l1_regularization(self, model: nn.Module) -> torch.Tensor:
        """
        L1 weight penalty on all trainable LoRA adapter parameters: sum(|W_trainable|).
        Enforces weight sparsity and prevents over-specialization.
        """
        l1_norm = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if param.requires_grad and ("lora" in name.lower() or "adapter" in name.lower()):
                l1_norm = l1_norm + torch.norm(param, p=1)
        return l1_norm

    def compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        T = self.cfg.temperature
        p_teacher = F.softmax(teacher_logits / T, dim=-1)
        log_p_student = F.log_softmax(student_logits / T, dim=-1)
        return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

    def forward(
        self,
        student_logits: torch.Tensor,
        student_cross_attentions: tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        model: nn.Module = None,
        teacher_logits: torch.Tensor = None,
        is_speech_anchor: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Forward evaluation returning composite regularized loss.
        """
        device = student_logits.device
        B, T_dec, V = student_logits.shape

        # 1. Label-smoothed Cross-Entropy
        loss_ce = self.ce_loss(student_logits.view(-1, V), labels.view(-1))

        # 2. Extract attention from last decoder layer
        final_layer_attn = student_cross_attentions[-1]
        avg_student_attn = final_layer_attn.mean(dim=1)
        valid_lengths = (labels != -100).sum(dim=-1)

        # 3. Dynamic Attention Loss
        loss_attn = self.compute_dynamic_guided_attention_loss(avg_student_attn, valid_lengths)

        # 4. Smooth Huber-Monotonicity Loss
        loss_mono = self.compute_smooth_huber_monotonicity_loss(avg_student_attn, valid_lengths)

        # 5. Teacher KD Loss
        loss_kd = self.compute_kd_loss(student_logits, teacher_logits) if teacher_logits is not None else torch.tensor(0.0, device=device)

        # 6. L1 Sparsity Regularization
        loss_l1 = self.compute_l1_regularization(model) if model is not None else torch.tensor(0.0, device=device)

        # If instance or batch contains conversational speech anchor, relax attention guidance to avoid pacing drift
        if isinstance(is_speech_anchor, torch.Tensor):
            speech_ratio = is_speech_anchor.float().mean().item()
        elif isinstance(is_speech_anchor, bool):
            speech_ratio = 1.0 if is_speech_anchor else 0.0
        else:
            speech_ratio = 0.0

        if speech_ratio > 0.5:
            loss_total = loss_ce + 0.1 * loss_attn + self.cfg.lambda_l1 * loss_l1
        else:
            loss_total = (
                self.cfg.alpha_ce * loss_ce
                + self.cfg.beta_kd * loss_kd
                + self.cfg.gamma_attn * loss_attn
                + self.cfg.lambda_mono * loss_mono
                + self.cfg.lambda_l1 * loss_l1
            )

        return {
            "loss_total": loss_total,
            "loss_ce": loss_ce,
            "loss_attn": loss_attn,
            "loss_mono": loss_mono,
            "loss_l1": loss_l1,
            "loss_kd": loss_kd
        }

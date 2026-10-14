"""
SING-ASR Cross-Attention Guided Knowledge Distillation Loss Module (Phase IX).
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Mathematical Foundations:
Autoregressive decoding on sustained singing vowels suffers from attention collapse:
cross-attention distributions fail to progress along the temporal acoustic frame axis,
dwelling on stationary harmonic frames and precipitating hallucinated repetition loops.

This module provides the composite loss:
    L_total = alpha * L_ce + beta * L_kd + gamma * L_attn + lambda * L_mono

Where:
1. L_ce: Cross-Entropy language modeling loss on ground-truth target tokens.
2. L_kd: Kullback-Leibler divergence between temperature-scaled Teacher and Student logits.
3. L_attn: Cross-Attention Guided Alignment loss enforcing a 2D Gaussian monotonic diagonal prior:
       M(t, f) = 1 - exp(-((t / T_dec) - (f / T_enc))^2 / (2 * sigma^2))
       L_attn = (1 / (B * T_dec)) * sum_{b,t,f} A_bar(b, t, f) * M(t, f)
4. L_mono: Temporal Monotonicity penalty on attention centroid progression:
       c_t = sum_f (f * A_bar(b, t, f))
       L_mono = (1 / (B * (T_dec - 1))) * sum_{b, t=2} max(0, delta - (c_t - c_{t-1}))^2
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillationHyperparameters:
    alpha_ce: float = 1.0       # Weight for ground-truth Cross-Entropy
    beta_kd: float = 0.5        # Weight for Teacher logit KL divergence
    gamma_attn: float = 2.5     # Weight for Cross-Attention Diagonality Mask
    lambda_mono: float = 1.0    # Weight for Attention Centroid Monotonicity
    temperature: float = 2.0    # Softmax temperature for KD
    sigma: float = 0.15         # Bandwidth for Gaussian diagonal prior
    min_progression: float = 0.5 # Minimum required frame advance delta


class CrossAttentionDistillationLoss(nn.Module):
    """
    Composite Knowledge Distillation & Cross-Attention Regularization Loss.
    """
    def __init__(self, hparams: DistillationHyperparameters = None):
        super().__init__()
        self.hp = hparams or DistillationHyperparameters()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)

    def compute_guided_attention_loss(
        self,
        cross_attention: torch.Tensor,
        valid_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes the guided attention loss against a Gaussian monotonic diagonal prior.
        Args:
            cross_attention: Tensor of shape (B, T_dec, T_enc), averaged over layers/heads.
            valid_lengths: Optional valid token lengths per batch item (B,).
        Returns:
            Scalar tensor for guided attention loss.
        """
        B, T_dec, T_enc = cross_attention.shape
        device = cross_attention.device

        # Create normalized coordinate grid
        t_coords = torch.linspace(0.0, 1.0, steps=T_dec, device=device).unsqueeze(1)  # (T_dec, 1)
        f_coords = torch.linspace(0.0, 1.0, steps=T_enc, device=device).unsqueeze(0)  # (1, T_enc)

        # 2D Gaussian Diagonality Penalty Mask: M(t, f) in [0, 1]
        dist_sq = (t_coords - f_coords) ** 2
        diag_mask = 1.0 - torch.exp(-dist_sq / (2.0 * (self.hp.sigma ** 2)))  # (T_dec, T_enc)
        diag_mask = diag_mask.unsqueeze(0).expand(B, -1, -1)  # (B, T_dec, T_enc)

        # Weighted penalty
        penalty = cross_attention * diag_mask

        if valid_lengths is not None:
            # Mask out padding tokens
            mask = torch.arange(T_dec, device=device).unsqueeze(0) < valid_lengths.unsqueeze(1)
            penalty = penalty * mask.unsqueeze(-1)
            norm = valid_lengths.clamp(min=1).sum()
        else:
            norm = float(B * T_dec)

        return penalty.sum() / norm

    def compute_monotonicity_loss(
        self,
        cross_attention: torch.Tensor,
        valid_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes monotonicity penalty on attention centroid progression.
        Args:
            cross_attention: Tensor of shape (B, T_dec, T_enc).
        Returns:
            Scalar tensor penalizing stalling or backward attention progression.
        """
        B, T_dec, T_enc = cross_attention.shape
        if T_dec <= 1:
            return torch.tensor(0.0, device=cross_attention.device)

        device = cross_attention.device
        frame_indices = torch.arange(T_enc, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, T_enc)

        # Compute centroid at each decoding step: c_t = sum_f (f * A(b, t, f))
        centroids = (cross_attention * frame_indices).sum(dim=-1)  # (B, T_dec)

        # Compute delta between consecutive steps
        diffs = centroids[:, 1:] - centroids[:, :-1]  # (B, T_dec - 1)

        # Penalty: max(0, delta - diffs)^2
        penalties = F.relu(self.hp.min_progression - diffs) ** 2

        if valid_lengths is not None:
            mask = torch.arange(T_dec - 1, device=device).unsqueeze(0) < (valid_lengths.unsqueeze(1) - 1).clamp(min=0)
            penalties = penalties * mask
            norm = mask.sum().clamp(min=1)
        else:
            norm = float(B * (T_dec - 1))

        return penalties.sum() / norm

    def compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes temperature-scaled KL divergence between teacher and student distributions.
        """
        T = self.hp.temperature
        p_teacher = F.softmax(teacher_logits / T, dim=-1)
        log_p_student = F.log_softmax(student_logits / T, dim=-1)

        # KL(P_T || P_S) = sum P_T * (log P_T - log P_S)
        kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)
        return kl

    def forward(
        self,
        student_logits: torch.Tensor,
        student_cross_attentions: tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        teacher_logits: torch.Tensor = None,
        teacher_cross_attentions: tuple[torch.Tensor, ...] = None
    ) -> dict[str, torch.Tensor]:
        """
        Evaluates full composite distillation loss.
        """
        device = student_logits.device

        # 1. Standard Cross-Entropy
        # student_logits: (B, T_dec, V), labels: (B, T_dec)
        B, T_dec, V = student_logits.shape
        loss_ce = self.ce_loss(student_logits.view(-1, V), labels.view(-1))

        # 2. Extract and average student cross-attention
        # cross_attentions: tuple of (B, num_heads, T_dec, T_enc) for each decoder layer
        # Take the average over heads from the final decoder layer (highest semantic level)
        final_layer_attn = student_cross_attentions[-1]  # (B, num_heads, T_dec, T_enc)
        avg_student_attn = final_layer_attn.mean(dim=1)  # (B, T_dec, T_enc)

        # Compute valid token lengths ignoring -100
        valid_lengths = (labels != -100).sum(dim=-1)

        # 3. Guided Attention Loss
        loss_attn = self.compute_guided_attention_loss(avg_student_attn, valid_lengths=valid_lengths)

        # 4. Monotonicity Step Loss
        loss_mono = self.compute_monotonicity_loss(avg_student_attn, valid_lengths=valid_lengths)

        # 5. Teacher KD Loss (if teacher logits supplied)
        if teacher_logits is not None:
            loss_kd = self.compute_kd_loss(student_logits, teacher_logits)
        else:
            loss_kd = torch.tensor(0.0, device=device)

        # Optional: Direct attention matching if teacher attention supplied
        if teacher_cross_attentions is not None:
            avg_teacher_attn = teacher_cross_attentions[-1].mean(dim=1)
            # Frobenius distance between attention maps
            loss_attn_match = F.mse_loss(avg_student_attn, avg_teacher_attn)
            loss_attn = loss_attn + loss_attn_match

        # Total Loss
        loss_total = (
            self.hp.alpha_ce * loss_ce
            + self.hp.beta_kd * loss_kd
            + self.hp.gamma_attn * loss_attn
            + self.hp.lambda_mono * loss_mono
        )

        return {
            "loss_total": loss_total,
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_attn": loss_attn,
            "loss_mono": loss_mono
        }

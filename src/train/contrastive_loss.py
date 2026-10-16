"""
SING-ASR Syllable-Tone InfoNCE Contrastive Loss Module (Phase XIII).
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Mathematical Foundations:
The primary challenge of singing ASR is the domain drift between conversational speech and singing:
the same lexical token (e.g. "thương", "buồn", "mây") produces widely divergent acoustic embeddings
due to melodic fundamental frequency modulation (F0), vibrato, and vowel prolongation.

This module provides the Syllable-Tone InfoNCE Loss:
    L_InfoNCE = - (1 / B) * sum_{i=1}^B log [ exp(sim(z_sing_i, z_speech_i) / tau) / sum_{j=1}^B exp(sim(z_sing_i, z_speech_j) / tau) ]

Where:
- Positive pairs (z_sing_i, z_speech_i): Time-pooled encoder representations of identical lexical text.
- Negative pairs (z_sing_i, z_speech_j): Heterogeneous text tokens across the mini-batch.
- tau: Temperature hyperparameter (default 0.07).

Minimizing L_InfoNCE forces the encoder to strip away musical melody and isolate intrinsic phonetic identities.
"""
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContrastiveLossConfig:
    temperature: float = 0.07       # InfoNCE temperature
    projection_dim: int = 128      # Contrastive projection head dimension
    d_model: int = 384             # Input acoustic hidden dimension
    orthogonality_weight: float = 0.05 # Penalty on correlation with pitch vectors


class SyllableToneInfoNCELoss(nn.Module):
    """
    Contrastive Representation Learning Head & InfoNCE Loss for Melody Invariance.
    """
    def __init__(self, config: Optional[ContrastiveLossConfig] = None):
        super().__init__()
        self.cfg = config or ContrastiveLossConfig()
        
        # Projection head: MLP with LayerNorm
        self.projector = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.projection_dim),
            nn.GELU(),
            nn.Linear(self.cfg.projection_dim, self.cfg.projection_dim),
            nn.LayerNorm(self.cfg.projection_dim)
        )

    def forward(
        self,
        singing_features: torch.Tensor,
        speech_features: torch.Tensor,
        singing_mask: Optional[torch.Tensor] = None,
        speech_mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """
        singing_features: (B, T_sing, d_model)
        speech_features: (B, T_speech, d_model)
        Returns:
            dict containing loss_infonce, avg_positive_sim, avg_negative_sim.
        """
        device = singing_features.device
        B = singing_features.shape[0]

        # 1. Global average pooling along time dimension
        if singing_mask is not None:
            # (B, T_sing, 1)
            mask_s = singing_mask.unsqueeze(-1).float()
            z_sing_pooled = (singing_features * mask_s).sum(dim=1) / mask_s.sum(dim=1).clamp(min=1.0)
        else:
            z_sing_pooled = singing_features.mean(dim=1)  # (B, d_model)

        if speech_mask is not None:
            mask_sp = speech_mask.unsqueeze(-1).float()
            z_speech_pooled = (speech_features * mask_sp).sum(dim=1) / mask_sp.sum(dim=1).clamp(min=1.0)
        else:
            z_speech_pooled = speech_features.mean(dim=1)  # (B, d_model)

        # 2. Project into contrastive space
        p_sing = self.projector(z_sing_pooled)      # (B, projection_dim)
        p_speech = self.projector(z_speech_pooled)  # (B, projection_dim)

        # 3. L2 Normalize
        p_sing = F.normalize(p_sing, p=2, dim=-1)
        p_speech = F.normalize(p_speech, p=2, dim=-1)

        # 4. Cosine similarity matrix: (B, B)
        sim_matrix = torch.matmul(p_sing, p_speech.T) / self.cfg.temperature

        # 5. Targets: Diagonals are paired instances (0, 1, ..., B-1)
        targets = torch.arange(B, dtype=torch.long, device=device)

        # Cross-Entropy over similarity matrix (InfoNCE)
        # Symmetrical bidirectional loss
        loss_s2sp = F.cross_entropy(sim_matrix, targets)
        loss_sp2s = F.cross_entropy(sim_matrix.T, targets)
        loss_infonce = 0.5 * (loss_s2sp + loss_sp2s)

        # Metrics for monitoring
        diag_sim = torch.diagonal(sim_matrix * self.cfg.temperature).mean().item()
        
        # Off-diagonal (negative) similarity
        mask_off_diag = ~torch.eye(B, dtype=torch.bool, device=device)
        if mask_off_diag.any():
            off_diag_sim = (sim_matrix * self.cfg.temperature)[mask_off_diag].mean().item()
        else:
            off_diag_sim = 0.0

        return {
            "loss_infonce": loss_infonce,
            "avg_positive_sim": round(diag_sim, 4),
            "avg_negative_sim": round(off_diag_sim, 4),
            "alignment_gap": round(diag_sim - off_diag_sim, 4)
        }

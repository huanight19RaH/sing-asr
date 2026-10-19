"""
Cross-Lingual Pitch Adapter (Phase XX: Multilingual Tonal Generalization).
Speech Processing (SLP301) - AIT Laboratory, Faculty of Artificial Intelligence.
Lead Researcher: Thai Hung Nguyen

Enables zero-shot and few-shot cross-lingual pitch transfer across Asian tonal languages:
- Vietnamese (6 tones, creaky glottalization)
- Mandarin Chinese (4 tones + neutral, continuous contours)
- Cantonese (6 tones, high level density)
- Thai (5 tones, distinct onset inflections)

Architecture:
1. Universal Pitch Physics Core:
   F0 normalization to semitone space relative to speaker baseline F0_ref:
   Delta_st(t) = 12 * log2(F0(t) / F0_ref)
2. Language-Specific Tonal Projector:
   Lightweight bottleneck adapter (rank=16) that maps universal continuous pitch contours
   into language-specific tonal latent representations.
3. Zero-Shot vs Few-Shot Transfer Evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CrossLingualPitchConfig:
    f0_dim: int = 1                # Raw continuous F0
    hidden_dim: int = 384          # Transformer hidden dimension
    adapter_bottleneck: int = 32   # Lightweight adapter rank
    num_languages: int = 4         # Vietnamese, Mandarin, Cantonese, Thai
    dropout: float = 0.10


class LanguageTonalProjector(nn.Module):
    """
    Language-specific bottleneck adapter mapping continuous F0 features
    into latent tonal subspace.
    """
    def __init__(self, in_dim: int, bottleneck_dim: int, out_dim: int):
        super().__init__()
        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.act(self.down(x))
        out = self.up(h) + res
        return self.norm(out)


class CrossLingualPitchAdapter(nn.Module):
    """
    Cross-lingual pitch adapter for multi-tonal speech & singing ASR.
    """
    def __init__(self, config: Optional[CrossLingualPitchConfig] = None):
        super().__init__()
        self.config = config or CrossLingualPitchConfig()
        
        # 1. Universal Semitone Feature Extractor
        # Maps raw F0 and voiced probability into universal harmonic embedding
        self.universal_pitch_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim)
        )
        
        # 2. Language-specific adapters:
        # 0: Vietnamese, 1: Mandarin, 2: Cantonese, 3: Thai
        self.language_adapters = nn.ModuleList([
            LanguageTonalProjector(self.config.hidden_dim, self.config.adapter_bottleneck, self.config.hidden_dim)
            for _ in range(self.config.num_languages)
        ])
        
        self.lang_to_idx = {
            "vietnamese": 0,
            "mandarin": 1,
            "cantonese": 2,
            "thai": 3
        }

    def normalize_f0_to_semitones(self, f0: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
        """
        Converts continuous F0 to semitones relative to speaker mean.
        Args:
            f0: (B, T) in Hz
            voiced: (B, T) boolean mask
        Returns:
            st: (B, T, 1) normalized semitones
        """
        # Replace unvoiced with 0
        voiced_mask = voiced & (f0 > 30.0)
        eps = 1e-4
        
        # Speaker mean F0 across voiced frames
        f0_clamped = torch.clamp(f0, min=eps)
        if torch.any(voiced_mask):
            mean_f0 = torch.sum(f0 * voiced_mask.float()) / (torch.sum(voiced_mask.float()) + eps)
        else:
            mean_f0 = torch.tensor(180.0, device=f0.device)
            
        st = 12.0 * torch.log2(f0_clamped / (mean_f0 + eps))
        st = torch.where(voiced_mask, st, torch.zeros_like(st))
        return st.unsqueeze(-1)

    def forward(
        self,
        f0: torch.Tensor,
        voiced: torch.Tensor,
        language: str = "vietnamese",
        zero_shot: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            f0: (B, T) continuous F0 in Hz
            voiced: (B, T) boolean voiced flag
            language: 'vietnamese', 'mandarin', 'cantonese', or 'thai'
            zero_shot: If True, uses Vietnamese universal base without language adapter
        Returns:
            tonal_embeddings: (B, T, hidden_dim) aligned for Transformer conditioning
        """
        B, T = f0.shape
        st = self.normalize_f0_to_semitones(f0, voiced) # (B, T, 1)
        v_feat = voiced.float().unsqueeze(-1)          # (B, T, 1)
        
        pitch_input = torch.cat([st, v_feat], dim=-1)   # (B, T, 2)
        universal_feat = self.universal_pitch_embed(pitch_input) # (B, T, hidden_dim)
        
        if zero_shot or language.lower() not in self.lang_to_idx:
            # Zero-shot transfer directly using universal base
            tonal_embeddings = universal_feat
            adapter_applied = "zero_shot_universal"
        else:
            lang_idx = self.lang_to_idx[language.lower()]
            tonal_embeddings = self.language_adapters[lang_idx](universal_feat)
            adapter_applied = f"adapter_{language.lower()}"
            
        return {
            "tonal_embeddings": tonal_embeddings,
            "universal_features": universal_feat,
            "adapter_type": adapter_applied
        }

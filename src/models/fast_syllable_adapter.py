"""
Fast Syllable Adapter (Phase XIX: Rhythmic Syncopation & Extreme Flow Robustness).
Speech Processing (SLP301) - AIT Laboratory, Faculty of Artificial Intelligence.
Lead Researcher: Thai Hung Nguyen

Solves the Acoustic Decimation Bottleneck in Transformer ASR for Fast Rap / Hip-Hop:
In standard Whisper/PhoWhisper encoders:
- Mel input: 10ms hop (100 Hz)
- Two conv layers with stride=2 reduce resolution to 40ms (25 Hz)
- Syllables at 8.0 - 10.5 SPS have durations of 95 - 125ms with 30-50ms vowels.
- 40ms stride merges adjacent consonant-vowel transitions, causing catastrophic deletion (>40% WER).

FastSyllableAdapter implements Transient-Preserving Temporal Pooling (TPTP):
1. Transient Flux Detector:
   Measures frame-level high-frequency energy flux to detect rapid syllabic bursts.
2. Dual-Resolution Feature Pyramid:
   - Base branch: standard 4x downsampled path (40ms) for linguistic macro-context.
   - High-res branch: 2x downsampled path (20ms) preserving transient phoneme boundaries.
3. Adaptive Dynamic Gating:
   Dynamically injects high-resolution consonant onset features when syllable rate > 6.0 SPS.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FastSyllableConfig:
    in_channels: int = 80          # Log-mel spectrogram channels
    hidden_dim: int = 384          # PhoWhisper / Whisper-small encoder dim
    transient_threshold: float = 0.45
    high_res_weight: float = 0.50
    dropout: float = 0.10


class TransientPreservingPooling(nn.Module):
    """
    Computes an adaptive pooling mask preserving onset transients
    where frame-to-frame spectral flux is maximal.
    """
    def __init__(self, in_channels: int = 80):
        super().__init__()
        self.flux_conv = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.gate_proj = nn.Conv1d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, C, T) log-mel spectrogram
        Returns:
            transient_gate: (B, 1, T) soft transient probability in [0, 1]
        """
        # First-order temporal difference (spectral flux proxy)
        diff = torch.zeros_like(mel)
        diff[:, :, 1:] = torch.clamp(mel[:, :, 1:] - mel[:, :, :-1], min=0.0)
        
        feat = F.relu(self.flux_conv(diff))
        gate = torch.sigmoid(self.gate_proj(feat))
        return gate


class FastSyllableAdapter(nn.Module):
    """
    Dual-resolution temporal adapter seamlessly attaching before/inside Whisper encoder.
    """
    def __init__(self, config: Optional[FastSyllableConfig] = None):
        super().__init__()
        self.config = config or FastSyllableConfig()
        
        # 1. Transient Detector
        self.transient_detector = TransientPreservingPooling(self.config.in_channels)
        
        # 2. Standard Macro Branch: 4x downsampling (two stride-2 convs)
        self.macro_conv1 = nn.Conv1d(self.config.in_channels, self.config.hidden_dim, kernel_size=3, stride=2, padding=1)
        self.macro_conv2 = nn.Conv1d(self.config.hidden_dim, self.config.hidden_dim, kernel_size=3, stride=2, padding=1)
        
        # 3. High-Resolution Micro Branch: 2x downsampling (one stride-2, one stride-1)
        self.micro_conv1 = nn.Conv1d(self.config.in_channels, self.config.hidden_dim // 2, kernel_size=3, stride=2, padding=1)
        self.micro_conv2 = nn.Conv1d(self.config.hidden_dim // 2, self.config.hidden_dim // 2, kernel_size=3, stride=1, padding=1)
        
        # 4. Syllable Boundary Consonant Injector (projects 2x micro features into 4x macro space)
        # Using stride-2 pooling on micro features to align temporal dimensions
        self.micro_to_macro = nn.Sequential(
            nn.Conv1d(self.config.hidden_dim // 2, self.config.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.config.hidden_dim),
            nn.GELU()
        )
        
        # 5. Dynamic Fusion Layer
        self.fusion = nn.Sequential(
            nn.Conv1d(self.config.hidden_dim * 2, self.config.hidden_dim, kernel_size=1),
            nn.Dropout(self.config.dropout),
            nn.LayerNorm([self.config.hidden_dim])
        )

    def forward(
        self,
        mel: torch.Tensor,
        sps_hint: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel: (B, C, T) log-mel spectrogram at 100 Hz (10ms hop)
            sps_hint: Optional estimated SPS (syllables per second)
        Returns:
            Dictionary containing:
            - adapter_output: (B, T_out, hidden_dim) aligned for Transformer blocks
            - transient_gate: (B, 1, T) detected transient activity
            - high_res_features: (B, hidden_dim // 2, T // 2) 20ms resolution representations
        """
        B, C, T = mel.shape
        
        # Step 1: Detect onset transients
        transient_gate = self.transient_detector(mel)  # (B, 1, T)
        
        # Step 2: Base Macro Branch (standard 4x downsampling)
        x_macro = F.gelu(self.macro_conv1(mel))
        x_macro = F.gelu(self.macro_conv2(x_macro))  # (B, hidden_dim, T // 4)
        
        # Step 3: High-Resolution Micro Branch (2x downsampling, 20ms frames)
        x_micro = F.gelu(self.micro_conv1(mel))
        x_micro = F.gelu(self.micro_conv2(x_micro))  # (B, hidden_dim//2, T // 2)
        
        # Modulate micro branch by pooled transient gate
        gate_2x = F.adaptive_avg_pool1d(transient_gate, x_micro.shape[-1])
        x_micro_gated = x_micro * (1.0 + gate_2x)
        
        # Project micro branch into macro temporal resolution
        x_micro_projected = self.micro_to_macro(x_micro_gated)  # (B, hidden_dim, T // 4)
        
        # Ensure exact frame match
        min_len = min(x_macro.shape[-1], x_micro_projected.shape[-1])
        x_macro = x_macro[:, :, :min_len]
        x_micro_projected = x_micro_projected[:, :, :min_len]
        
        # Adaptive weighting based on SPS hint or mean transient activation
        mean_transient = torch.mean(transient_gate).item()
        if sps_hint is not None:
            # If SPS > 6.0, boost high-resolution weighting
            weight = min(0.85, max(0.20, (sps_hint - 4.0) / 6.0))
        else:
            weight = min(0.85, max(0.20, mean_transient * 1.5))
            
        fused = torch.cat([x_macro, x_micro_projected * weight], dim=1)  # (B, 2*hidden_dim, T_out)
        
        # Permute for LayerNorm (B, T_out, 2*hidden_dim) -> Conv1d needs (B, C, T)
        fused_proj = self.fusion[0](fused)  # (B, hidden_dim, T_out)
        fused_drop = self.fusion[1](fused_proj)
        
        # LayerNorm over hidden_dim (channel dimension last)
        out = self.fusion[2](fused_drop.transpose(1, 2))  # (B, T_out, hidden_dim)
        
        return {
            "adapter_output": out,
            "transient_gate": transient_gate,
            "high_res_features": x_micro,
            "applied_weight": torch.tensor(weight, device=mel.device)
        }

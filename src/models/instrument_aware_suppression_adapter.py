"""
src/models/instrument_aware_suppression_adapter.py
Phase XXIII: Sub-Agent 2 (Model Architect)

Instrument-Aware Sub-Band Suppression Adapter (IAS-Adapter):
Selectively suppresses specific musical instrument interference (Drums, Piano, Guitar, Synth/808)
using a tri-band decomposition architecture:
1. Low-Band (<250 Hz): Sub-Bass De-leakage Gate (conditioned on Synth/808 probability).
2. Mid-Band (250 - 3000 Hz): Harmonic Formant De-smearing Self-Attention (conditioned on Piano/Guitar).
3. High-Band (>3000 Hz): Transient Gated Soft Limiter (conditioned on Drums/Percussion).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InstrumentPresenceClassifier(nn.Module):
    """
    Lightweight classifier detecting active musical instrument interference:
    Classes: [Drums, Piano, Guitar, Synth_808]
    """
    def __init__(self, embed_dim=768, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, num_classes)
        )

    def forward(self, x):
        # x: (B, T, D)
        # Global temporal average pooling
        x_pool = torch.mean(x, dim=1)  # (B, D)
        logits = self.net(x_pool)      # (B, 4)
        probs = torch.sigmoid(logits)  # (B, 4)
        return probs

class LowBandSubBassGate(nn.Module):
    """
    Suppresses sub-bass energy floor leakage (<250 Hz) without degrading
    fundamental vocal pitch inflections.
    """
    def __init__(self, dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, z_low, p_synth):
        # z_low: (B, T, D_low)
        # p_synth: (B, 1)
        B, T, D = z_low.shape
        p_expanded = p_synth.unsqueeze(1).expand(B, T, 1)
        cond = torch.cat([z_low, p_expanded], dim=-1)
        alpha = self.gate_net(cond)
        # Suppress according to synth interference
        z_out = self.proj(z_low * (1.0 - 0.7 * alpha * p_expanded))
        return z_out

class MidBandFormantAttentionGate(nn.Module):
    """
    Un-smears vocal vowel formants (F1, F2) in the mid-band (250-3000 Hz)
    against dense polyphonic piano chords and guitar harmonic vibrato.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.cond_proj = nn.Linear(2, dim)

    def forward(self, z_mid, p_piano, p_guitar):
        # z_mid: (B, T, D_mid)
        # p_piano, p_guitar: (B, 1)
        B, T, D = z_mid.shape
        p_keys = torch.cat([p_piano, p_guitar], dim=-1) # (B, 2)
        cond_bias = self.cond_proj(p_keys).unsqueeze(1) # (B, 1, D)

        z_norm = self.norm(z_mid + cond_bias)
        attn_out, _ = self.mha(z_norm, z_norm, z_norm)
        z_mid = z_mid + attn_out
        z_mid = z_mid + self.ffn(self.norm(z_mid))
        return z_mid

class HighBandTransientLimiterGate(nn.Module):
    """
    Suppresses sharp percussive drum/snare transients (>3000 Hz) while
    preserving subtle voiceless consonant onsets (/t/, /th/, /s/, /x/).
    """
    def __init__(self, dim):
        super().__init__()
        self.transient_detector = nn.Sequential(
            nn.Linear(dim + 1, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        self.soft_limiter = nn.Tanh()
        self.proj = nn.Linear(dim, dim)

    def forward(self, z_high, p_drum):
        # z_high: (B, T, D_high)
        # p_drum: (B, 1)
        B, T, D = z_high.shape
        p_expanded = p_drum.unsqueeze(1).expand(B, T, 1)
        cond = torch.cat([z_high, p_expanded], dim=-1)
        transient_mask = self.transient_detector(cond) # (B, T, 1)

        # Attenuate when drum probability is high and transient spike detected
        attenuation = 1.0 - (0.65 * p_expanded * transient_mask)
        z_out = self.proj(self.soft_limiter(z_high) * attenuation)
        return z_out

class InstrumentAwareSuppressionAdapter(nn.Module):
    """
    Full Phase XXIII Instrument-Aware Suppression Adapter.
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_low = embed_dim // 4   # 192
        self.d_mid = embed_dim // 2   # 384
        self.d_high = embed_dim // 4  # 192

        # 1. Instrument Presence Classifier Head
        self.classifier = InstrumentPresenceClassifier(embed_dim=embed_dim, num_classes=4)

        # 2. Tri-Band Sub-Band Split Projections
        self.proj_low = nn.Linear(embed_dim, self.d_low)
        self.proj_mid = nn.Linear(embed_dim, self.d_mid)
        self.proj_high = nn.Linear(embed_dim, self.d_high)

        # 3. Sub-Band Gates
        self.low_gate = LowBandSubBassGate(self.d_low)
        self.mid_gate = MidBandFormantAttentionGate(self.d_mid, num_heads=4)
        self.high_gate = HighBandTransientLimiterGate(self.d_high)

        # 4. Fusion and Residual Recombination
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, embed_dim) - input acoustic representations
        Returns:
            dict containing:
                "output": Suppressed feature representations (B, T, embed_dim)
                "instrument_probs": Detected instrument collision probabilities (B, 4)
                "p_drum": Tensor (B, 1)
                "p_piano": Tensor (B, 1)
                "p_guitar": Tensor (B, 1)
                "p_synth": Tensor (B, 1)
        """
        # 1. Detect instrument collisions
        inst_probs = self.classifier(x) # (B, 4) -> [drum, piano, guitar, synth]
        p_drum = inst_probs[:, 0:1]
        p_piano = inst_probs[:, 1:2]
        p_guitar = inst_probs[:, 2:3]
        p_synth = inst_probs[:, 3:4]

        # 2. Tri-band projection
        z_low = self.proj_low(x)
        z_mid = self.proj_mid(x)
        z_high = self.proj_high(x)

        # 3. Gated sub-band suppression
        z_low_out = self.low_gate(z_low, p_synth)
        z_mid_out = self.mid_gate(z_mid, p_piano, p_guitar)
        z_high_out = self.high_gate(z_high, p_drum)

        # 4. Concatenation and fusion
        z_fused = torch.cat([z_low_out, z_mid_out, z_high_out], dim=-1) # (B, T, D)
        suppressed = self.fusion(z_fused)

        # 5. Residual identity connection
        out = self.out_norm(x + self.res_scale * suppressed)

        return {
            "output": out,
            "instrument_probs": inst_probs,
            "p_drum": p_drum,
            "p_piano": p_piano,
            "p_guitar": p_guitar,
            "p_synth": p_synth
        }

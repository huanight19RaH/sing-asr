import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HarmonicDisentanglementGate(nn.Module):
    """
    Computes soft harmonic masks for two concurrent speakers based on their
    estimated fundamental frequency trajectories F0_1(t) and F0_2(t).
    Projects masks into the latent space to modulate key/value representations.
    """
    def __init__(self, d_model=384, n_mels=80, num_harmonics=8):
        super().__init__()
        self.d_model = d_model
        self.n_mels = n_mels
        self.num_harmonics = num_harmonics
        
        # Linear projections from mel-scale harmonic energy to latent modulation
        self.mel_to_latent_lead = nn.Sequential(
            nn.Linear(n_mels, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        self.mel_to_latent_sec = nn.Sequential(
            nn.Linear(n_mels, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        
        # Center frequencies for 80 Mel bands (approximate 0 - 8000 Hz)
        mel_points = torch.linspace(0, 2595 * math.log10(1 + 8000 / 700), n_mels)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        self.register_buffer("mel_center_hz", hz_points)

    def forward(self, f0_lead, f0_sec, seq_len):
        """
        f0_lead: [B, T] in Hz
        f0_sec:  [B, T] in Hz
        Returns:
            mask_lead: [B, T, d_model]
            mask_sec:  [B, T, d_model]
        """
        B, T = f0_lead.shape
        f0_l = f0_lead.unsqueeze(-1).clamp(min=40.0, max=1200.0) # [B, T, 1]
        f0_s = f0_sec.unsqueeze(-1).clamp(min=40.0, max=1200.0)  # [B, T, 1]
        
        centers = self.mel_center_hz.view(1, 1, self.n_mels)     # [1, 1, M]
        
        # Multi-harmonic Gaussian comb for Lead
        lead_energy = torch.zeros(B, T, self.n_mels, device=f0_lead.device)
        sec_energy = torch.zeros(B, T, self.n_mels, device=f0_sec.device)
        
        for k in range(1, self.num_harmonics + 1):
            sigma_k = 30.0 * math.sqrt(k)
            k_f0_l = k * f0_l
            k_f0_s = k * f0_s
            lead_energy += torch.exp(-((centers - k_f0_l) ** 2) / (2 * sigma_k ** 2))
            sec_energy += torch.exp(-((centers - k_f0_s) ** 2) / (2 * sigma_k ** 2))
            
        lead_energy = lead_energy / self.num_harmonics
        sec_energy = sec_energy / self.num_harmonics
        
        # Project to latent space
        mask_l = self.mel_to_latent_lead(lead_energy) # [B, T, d_model]
        mask_s = self.mel_to_latent_sec(sec_energy)   # [B, T, d_model]
        
        # Interpolate if time dimensions differ
        if T != seq_len:
            mask_l = F.interpolate(mask_l.transpose(1, 2), size=seq_len, mode="linear", align_corners=False).transpose(1, 2)
            mask_s = F.interpolate(mask_s.transpose(1, 2), size=seq_len, mode="linear", align_corners=False).transpose(1, 2)
            
        return mask_l, mask_s


class DuetStreamBlock(nn.Module):
    """
    Quantization-safe multi-head self-attention and feedforward stream block.
    """
    def __init__(self, d_model=384, n_heads=6):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x


class DualStreamDuetAdapter(nn.Module):
    """
    Dual-Stream Attentive Disentanglement Adapter (D-SADA) for multi-speaker duet
    and overlapping vocal transcription.
    Splits mixed acoustic latents into two orthogonal latent streams: Lead & Secondary.
    """
    def __init__(self, d_model=384, n_heads=6, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # Harmonic prior gating
        self.harmonic_gate = HarmonicDisentanglementGate(d_model=d_model)
        
        # Stream 1: Lead vocal processing path
        self.lead_stream = nn.Sequential(*[DuetStreamBlock(d_model, n_heads) for _ in range(num_layers)])
        self.lead_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Stream 2: Secondary / backing vocal processing path
        self.sec_stream = nn.Sequential(*[DuetStreamBlock(d_model, n_heads) for _ in range(num_layers)])
        self.sec_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Mutual cross-attention for speaker boundary disentanglement
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)

    def forward(self, z_mixed, f0_lead=None, f0_sec=None):
        """
        z_mixed: [B, T, d_model] - Latent representations from ASR encoder
        f0_lead: [B, T_f0] - Optional pitch contour of Speaker 1 (in Hz)
        f0_sec:  [B, T_f0] - Optional pitch contour of Speaker 2 (in Hz)
        Returns:
            z_lead: [B, T, d_model] - Disentangled Lead speaker latent
            z_sec:  [B, T, d_model] - Disentangled Secondary/Backing speaker latent
            orth_penalty: scalar loss promoting orthogonality between streams
        """
        B, T, D = z_mixed.shape
        
        if f0_lead is not None and f0_sec is not None:
            mask_l, mask_s = self.harmonic_gate(f0_lead, f0_sec, seq_len=T)
            z_l_in = z_mixed * (1.0 + mask_l)
            z_s_in = z_mixed * (1.0 + mask_s)
        else:
            z_l_in = z_mixed
            z_s_in = z_mixed
            
        # Process streams
        z_l = self.lead_stream(z_l_in)
        z_l = self.lead_projection(z_l)
        
        z_s = self.sec_stream(z_s_in)
        z_s = self.sec_projection(z_s)
        
        # Mutual cross-attention: allow secondary stream to subtract lead energy
        z_s_refined, _ = self.cross_attn(query=z_s, key=z_l, value=z_l)
        z_s = self.norm_cross(z_s - 0.3 * z_s_refined)
        
        # Orthogonality loss: penalize high cosine similarity between streams
        z_l_norm = F.normalize(z_l, p=2, dim=-1)
        z_s_norm = F.normalize(z_s, p=2, dim=-1)
        cos_sim = torch.sum(z_l_norm * z_s_norm, dim=-1) # [B, T]
        orth_penalty = torch.mean(torch.abs(cos_sim))
        
        return z_l, z_s, orth_penalty


class PermutationInvariantLoss(nn.Module):
    """
    Utterance-level Permutation Invariant Training (uPIT) loss.
    Computes minimum loss over all permutations of predicted streams vs target streams.
    """
    def __init__(self, criterion=nn.MSELoss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, pred_1, pred_2, target_1, target_2):
        """
        pred_1, pred_2: Disentangled latent outputs from DualStreamDuetAdapter
        target_1, target_2: Ground-truth isolated representations of speaker 1 and speaker 2
        Returns:
            min_loss: Minimum loss across (P1->T1, P2->T2) vs (P1->T2, P2->T1)
            best_perm: (0, 1) or (1, 0)
        """
        # Permutation 0: (1->1, 2->2)
        loss_p0 = self.criterion(pred_1, target_1) + self.criterion(pred_2, target_2)
        
        # Permutation 1: (1->2, 2->1)
        loss_p1 = self.criterion(pred_1, target_2) + self.criterion(pred_2, target_1)
        
        if loss_p0 <= loss_p1:
            return loss_p0, (0, 1)
        else:
            return loss_p1, (1, 0)

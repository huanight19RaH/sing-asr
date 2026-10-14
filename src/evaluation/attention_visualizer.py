"""
sing-asr Evaluation Module: 2D Cross-Attention Heatmap Visualization Engine.
Visualizes decoder-encoder cross-attention weights (Tokens vs. Audio Frames)
comparing Baseline Zero-shot vs. Regularized Champion LoRA model.
Demonstrates monotonic diagonal alignment vs. horizontal attention stalling / looping.
Supports Speech Processing (SLP301) and academic paper figures.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Configure UTF-8 for console output
sys.stdout.reconfigure(encoding='utf-8')

class AttentionVisualizer:
    def __init__(
        self,
        base_model_id: str = "vinai/PhoWhisper-tiny",
        champion_adapter_path: str = "models/large_regularized_champion_phowhisper",
        device: str = "cpu"
    ):
        self.base_model_id = base_model_id
        self.champion_adapter_path = Path(champion_adapter_path)
        self.device = device
        self._processor = None
        self._base_model = None
        self._champion_model = None

    def _init_models(self):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from peft import PeftModel

        if self._processor is None:
            print(f"[AttentionVisualizer] Loading processor from {self.base_model_id}...")
            self._processor = WhisperProcessor.from_pretrained(self.base_model_id)

        if self._base_model is None:
            print(f"[AttentionVisualizer] Loading Baseline model ({self.base_model_id}) on {self.device}...")
            self._base_model = WhisperForConditionalGeneration.from_pretrained(
                self.base_model_id,
                attn_implementation="eager"
            )
            self._base_model.to(self.device)
            self._base_model.eval()

        if self._champion_model is None and self.champion_adapter_path.exists():
            print(f"[AttentionVisualizer] Loading Champion model with LoRA adapter from {self.champion_adapter_path}...")
            base = WhisperForConditionalGeneration.from_pretrained(
                self.base_model_id,
                attn_implementation="eager"
            )
            self._champion_model = PeftModel.from_pretrained(base, str(self.champion_adapter_path))
            self._champion_model.to(self.device)
            self._champion_model.eval()

    def extract_cross_attention(
        self,
        model: Any,
        audio_path: str,
        max_new_tokens: int = 40
    ) -> Dict[str, Any]:
        """
        Run inference with output_attentions=True and return the 2D cross-attention map (Tokens x Frames).
        """
        self._init_models()
        # Load audio
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 16000)
            audio = resampler(torch.tensor(audio, dtype=torch.float32)).numpy()

        duration_sec = len(audio) / 16000.0
        # Active encoder frames: Whisper downsamples 100fps spectrogram by 2 -> 50 frames/sec (20ms/frame)
        active_frames = min(1500, int(duration_sec * 50))

        # Preprocess features
        inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        # 1. First generate the sequence tokens
        with torch.no_grad():
            outputs = model.generate(
                input_features,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )

        token_ids = outputs[0] if isinstance(outputs, torch.Tensor) else outputs.sequences[0]
        token_ids_list = token_ids.cpu().tolist()
        tokens = [self._processor.decode([tid], skip_special_tokens=False) for tid in token_ids_list]
        full_text = self._processor.decode(token_ids_list, skip_special_tokens=True).strip()

        # 2. Extract Cross-Attentions via deterministic forward pass on decoder
        from peft import PeftModel
        if isinstance(model, PeftModel):
            whisper_core = model.get_base_model().model
        else:
            whisper_core = model.model
        
        with torch.no_grad():
            enc_out = whisper_core.encoder(input_features)
            dec_in = token_ids.unsqueeze(0)
            dec_out = whisper_core.decoder(
                input_ids=dec_in,
                encoder_hidden_states=enc_out.last_hidden_state,
                output_attentions=True
            )

        # dec_out.cross_attentions is a tuple of L layers, each (1, num_heads, T_tokens, 1500)
        if dec_out.cross_attentions is not None and len(dec_out.cross_attentions) > 0:
            last_layer_attn = dec_out.cross_attentions[-1]  # shape: (1, num_heads, T, 1500)
            avg_heads = last_layer_attn.mean(dim=1).squeeze(0).cpu().numpy()  # shape: (T, 1500)
            attn_matrix = avg_heads[:, :active_frames]  # shape: (T, active_frames)
        else:
            attn_matrix = np.zeros((len(token_ids_list), active_frames))

        # Compute Monotonicity Alignment Metric (Pearson correlation r between token idx and argmax frame)
        if attn_matrix.shape[0] > 2 and attn_matrix.shape[1] > 2:
            token_indices = np.arange(attn_matrix.shape[0])
            argmax_frames = np.argmax(attn_matrix, axis=1)
            r_val, p_val = pearsonr(token_indices, argmax_frames)
        else:
            r_val, p_val = 0.0, 1.0

        return {
            'text': full_text,
            'tokens': tokens,
            'attention_matrix': attn_matrix,
            'duration_sec': duration_sec,
            'active_frames': active_frames,
            'monotonicity_r': round(float(r_val), 4),
            'monotonicity_p': float(p_val)
        }

    def compare_and_plot(
        self,
        audio_path: str,
        output_plot_path: str = "docs/figures/cross_attention_alignment_comparison.png",
        sample_title: str = "Singing Voice Sample: 'Trời còn làm mưa bay bay...'"
    ) -> Dict[str, Any]:
        """
        Compare Baseline vs Champion model on a singing audio sample and generate side-by-side heatmaps.
        """
        self._init_models()
        print(f"[AttentionVisualizer] Extracting Baseline Cross-Attention on {audio_path}...")
        base_res = self.extract_cross_attention(self._base_model, audio_path)

        print(f"[AttentionVisualizer] Extracting Champion Cross-Attention on {audio_path}...")
        champ_res = self.extract_cross_attention(self._champion_model, audio_path)

        # Plotting
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

        duration = base_res['duration_sec']
        active_frames = base_res['active_frames']
        time_extent = [0, duration, base_res['attention_matrix'].shape[0], 0]

        # 1. Baseline Plot
        mat_base = base_res['attention_matrix']
        # Normalize per token row for visual clarity
        row_max_base = mat_base.max(axis=1, keepdims=True)
        norm_base = np.divide(mat_base, row_max_base, out=np.zeros_like(mat_base), where=row_max_base > 0)

        im0 = axes[0].imshow(norm_base, aspect='auto', cmap='magma', extent=[0, duration, norm_base.shape[0], 0])
        axes[0].set_title(
            f"Baseline Zero-Shot PhoWhisper\n"
            f"Transcription: \"{base_res['text']}\"\n"
            f"Monotonicity Alignment r = {base_res['monotonicity_r']:.3f} (Attention Stalling / Looping)",
            fontsize=10, fontweight='bold', pad=12
        )
        axes[0].set_xlabel("Audio Time (seconds)", fontsize=10, fontweight='bold')
        axes[0].set_ylabel("Generated Token Index (Decoding Step)", fontsize=10, fontweight='bold')

        # 2. Champion Plot
        mat_champ = champ_res['attention_matrix']
        row_max_champ = mat_champ.max(axis=1, keepdims=True)
        norm_champ = np.divide(mat_champ, row_max_champ, out=np.zeros_like(mat_champ), where=row_max_champ > 0)

        im1 = axes[1].imshow(norm_champ, aspect='auto', cmap='viridis', extent=[0, duration, norm_champ.shape[0], 0])
        axes[1].set_title(
            f"Regularized Champion PhoWhisper (Ours)\n"
            f"Transcription: \"{champ_res['text']}\"\n"
            f"Monotonicity Alignment r = {champ_res['monotonicity_r']:.3f} (Sharp Diagonal Guidance)",
            fontsize=10, fontweight='bold', pad=12
        )
        axes[1].set_xlabel("Audio Time (seconds)", fontsize=10, fontweight='bold')
        axes[1].set_ylabel("Generated Token Index (Decoding Step)", fontsize=10, fontweight='bold')

        fig.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.15, label="Attention Weight (Norm)")
        fig.colorbar(im1, ax=axes[1], orientation='horizontal', fraction=0.046, pad=0.15, label="Attention Weight (Norm)")

        plt.suptitle(f"Cross-Attention Alignment Comparison: {sample_title}", fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[AttentionVisualizer] Figure saved successfully to: {output_plot_path}")

        return {
            'sample_audio': audio_path,
            'duration_sec': duration,
            'baseline': {
                'text': base_res['text'],
                'monotonicity_r': base_res['monotonicity_r'],
                'monotonicity_p': base_res['monotonicity_p'],
                'num_tokens': int(norm_base.shape[0])
            },
            'champion': {
                'text': champ_res['text'],
                'monotonicity_r': champ_res['monotonicity_r'],
                'monotonicity_p': champ_res['monotonicity_p'],
                'num_tokens': int(norm_champ.shape[0])
            },
            'figure_saved_to': output_plot_path
        }

if __name__ == '__main__':
    audio_sample = "data/processed/large_benchmark_suite/singing/sng_sec_02_verse1_p2.wav"
    output_png = "docs/figures/cross_attention_alignment_comparison.png"
    print(f"Executing Attention Visualizer on {audio_sample}...")
    viz = AttentionVisualizer()
    res = viz.compare_and_plot(audio_sample, output_png)
    import json
    stats_path = "data/processed/attention_alignment_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("Attention Alignment Analysis Complete!")
    print(f"Baseline Monotonicity r: {res['baseline']['monotonicity_r']}")
    print(f"Champion Monotonicity r: {res['champion']['monotonicity_r']}")

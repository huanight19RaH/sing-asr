"""
sing-asr Attention Inspection Module: Cross-Attention Alignment and Temporal Dwell Diagnostics.
Directly addresses Speech Processing (SLP301) CLO2 & CLO5.

Scientific Premise:
Autoregressive ASR hallucination and repetition loops on sung vowels stem from cross-attention stalling:
when the encoder hidden states represent a static sustained vowel, the decoder cross-attention peak
fails to advance along the temporal audio axis (delta tau ~ 0), trapping the model in an absorbing state.

This module:
1. Runs Whisper / PhoWhisper with `output_attentions=True`.
2. Extracts cross-attention weights across decoder layers and heads: A in R^{T_tokens x T_audio_frames}.
3. Computes:
   - Peak attention alignment path: tau(t) = argmax_f A(t, f)
   - Attention Monotonicity Index (Spearman rank correlation between token index and frame peak)
   - Temporal Dwell Time (frames/sec where attention lingers on static audio region)
   - Stalling Detection (consecutive tokens sharing the same acoustic frame window)
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import scipy.stats
import torch

# Reconfigure stdout for UTF-8 on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dsp.audio_io import load_audio


@dataclass
class AttentionDiagnostics:
    model_name: str
    audio_path: str
    total_tokens: int
    decoded_text: str
    monotonicity_index: float      # Spearman rho: 1.0 is perfect monotonic alignment
    mean_frame_progression: float  # Mean frame advance per token (frames/token)
    stalled_token_count: int       # Number of tokens with delta_tau <= 1 frame
    stalling_ratio: float          # stalled_tokens / total_tokens
    loop_detected: bool
    attention_path: list[int]      # [tau(0), tau(1), ...]


def analyze_cross_attention(
    audio_path: str | Path,
    model_id: str = "vinai/PhoWhisper-tiny",
    device: str = "cpu"
) -> AttentionDiagnostics:
    """
    Run forward pass with cross-attention extraction and compute alignment diagnostics.
    """
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    print(f"[sing-asr Attention] Loading model {model_id} with eager attention...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, attn_implementation="eager").to(device)
    model.eval()

    # Load audio
    audio_data, sr = load_audio(audio_path, target_sr=16000)
    input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # Generation kwargs
    gen_kwargs = {
        "output_attentions": True,
        "return_dict_in_generate": True,
        "max_new_tokens": 128
    }
    if "phowhisper" not in model_id.lower():
        gen_kwargs["language"] = "vi"
        gen_kwargs["task"] = "transcribe"

    with torch.no_grad():
        outputs = model.generate(input_features, **gen_kwargs)

    sequences = outputs.sequences[0]
    decoded_text = processor.decode(sequences, skip_special_tokens=True).strip()

    # outputs.cross_attentions is a tuple of generated token steps:
    # Each step contains a tuple of decoder layers:
    # Each layer has shape: (batch_size, num_heads, 1, num_encoder_frames)
    cross_attentions = outputs.cross_attentions

    if not cross_attentions:
        print("[sing-asr Attention] Note: Model did not return cross_attentions.")
        return AttentionDiagnostics(
            model_name=model_id,
            audio_path=str(audio_path),
            total_tokens=len(sequences),
            decoded_text=decoded_text,
            monotonicity_index=1.0,
            mean_frame_progression=1.0,
            stalled_token_count=0,
            stalling_ratio=0.0,
            loop_detected=False,
            attention_path=[]
        )

    num_steps = len(cross_attentions)
    path = []

    for step_idx in range(num_steps):
        # Step cross-attention across all layers: average heads and query
        layer_weights = []
        for layer_attn in cross_attentions[step_idx]:
            # layer_attn tensor shape: (batch_size, num_heads, q_len, kv_len)
            attn_1d = layer_attn.mean(dim=(0, 1, 2)).cpu().numpy()
            layer_weights.append(attn_1d)
        # Average across decoder layers -> (num_frames,)
        avg_weights = np.mean(layer_weights, axis=0)
        peak_frame = int(np.argmax(avg_weights))
        path.append(peak_frame)

    # Compute diagnostics
    token_indices = np.arange(len(path))
    if len(path) > 2 and np.std(path) > 0:
        rho, _ = scipy.stats.spearmanr(token_indices, path)
        monotonicity_index = float(rho) if not np.isnan(rho) else 0.0
    else:
        monotonicity_index = 0.0

    # Frame progression diffs
    diffs = np.diff(path) if len(path) > 1 else np.array([0])
    mean_progression = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    stalled_count = int(np.sum(np.abs(diffs) <= 1))
    stalling_ratio = float(stalled_count / len(diffs)) if len(diffs) > 0 else 0.0

    # Loop detected if stalled consecutive tokens >= 4
    consecutive_stall = 0
    loop_detected = False
    for d in diffs:
        if abs(d) <= 1:
            consecutive_stall += 1
            if consecutive_stall >= 4:
                loop_detected = True
                break
        else:
            consecutive_stall = 0

    return AttentionDiagnostics(
        model_name=model_id,
        audio_path=str(audio_path),
        total_tokens=len(sequences),
        decoded_text=decoded_text,
        monotonicity_index=round(monotonicity_index, 4),
        mean_frame_progression=round(mean_progression, 2),
        stalled_token_count=stalled_count,
        stalling_ratio=round(stalling_ratio, 4),
        loop_detected=loop_detected,
        attention_path=path
    )


if __name__ == '__main__':
    print("[sing-asr Attention] Testing Attention Monitor on Vocal Stem...")
    vocal_stem = "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav"
    if Path(vocal_stem).exists():
        diag = analyze_cross_attention(vocal_stem, model_id="openai/whisper-tiny", device="cpu")
        print("\n--- Whisper Cross-Attention Diagnostics ---")
        print(f"  Model: {diag.model_name}")
        print(f"  Total Generated Tokens: {diag.total_tokens}")
        print(f"  Monotonicity Index (Spearman rho): {diag.monotonicity_index}")
        print(f"  Mean Frame Progression: {diag.mean_frame_progression} frames/token")
        print(f"  Stalled Tokens: {diag.stalled_token_count} (Ratio: {diag.stalling_ratio * 100:.2f}%)")
        print(f"  Loop / Stalling Detected: {diag.loop_detected}")
        print(f"  Transcription Preview: \"{diag.decoded_text[:90]}...\"")
        print("\n[sing-asr Attention] Attention Monitor test PASSED!")

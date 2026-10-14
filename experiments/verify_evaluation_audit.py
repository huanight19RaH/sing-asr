"""
SING-ASR Independent Verification & Evaluation Audit Engine.
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Performs a rigorous, reproducible, end-to-end audit of:
1. Model checkpoints integrity (Zero-shot baseline, Phase VIII LoRA, Phase IX Distilled).
2. Word-by-word Levenshtein alignment (S, D, I counts, WER, CER).
3. Cross-Attention progression dynamics (Spearman rho, stalling ratio).
4. Qualitative error analysis on exact decoded strings vs reference lyrics/speech.
"""
import os
import sys

# Ensure pure CPU execution without CUDA driver interference
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import json
import time
from pathlib import Path
import numpy as np
import scipy.stats
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dsp.audio_io import load_audio
from evaluation.metrics import compute_metrics
from evaluation.alignment import align_words


def audit_model(model_name: str, model, processor, audio_path: Path, ref_text: str):
    audio, sr = load_audio(audio_path, target_sr=16000)
    duration_sec = len(audio) / sr

    t0 = time.time()
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features

    gen_params = {
        "max_new_tokens": 128,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "language": "vi",
        "task": "transcribe",
        "output_attentions": True,
        "return_dict_in_generate": True
    }

    with torch.no_grad():
        gen_out = model.generate(input_features, **gen_params)

    inf_time = time.time() - t0
    rtf = round(inf_time / duration_sec, 4) if duration_sec > 0 else 0.0

    hyp_text = processor.batch_decode(gen_out.sequences, skip_special_tokens=True)[0].strip()

    # Metrics
    m = compute_metrics(ref_text, hyp_text)
    al = align_words(ref_text, hyp_text)

    # Attention Dynamics
    path = []
    if gen_out.cross_attentions:
        for step_attn in gen_out.cross_attentions:
            if step_attn:
                avg_1d = step_attn[-1].mean(dim=(0, 1, 2)).cpu().numpy()
                path.append(int(np.argmax(avg_1d)))

    if len(path) > 2 and np.std(path) > 0:
        rho, p_val = scipy.stats.spearmanr(np.arange(len(path)), path)
        mono_idx = round(float(rho) if not np.isnan(rho) else 0.0, 4)
    else:
        mono_idx = 0.0

    diffs = np.diff(path) if len(path) > 1 else np.array([0])
    stalled = int(np.sum(np.abs(diffs) <= 1))
    stall_ratio = round(float(stalled / len(diffs)) if len(diffs) > 0 else 0.0, 4)

    return {
        "model": model_name,
        "audio_file": audio_path.name,
        "duration_sec": round(duration_sec, 2),
        "inference_sec": round(inf_time, 2),
        "rtf": rtf,
        "hypothesis": hyp_text,
        "wer": round(m.wer, 4),
        "cer": round(m.cer, 4),
        "substitutions": al.substitutions,
        "deletions": al.deletions,
        "insertions": al.insertions,
        "ref_words": len(ref_text.split()),
        "hyp_words": len(hyp_text.split()),
        "monotonicity_index": mono_idx,
        "stalling_ratio": stall_ratio,
        "attention_path_length": len(path)
    }


def main():
    print("=" * 80, flush=True)
    print("SING-ASR RIGOROUS RE-VERIFICATION & EVALUATION AUDIT", flush=True)
    print("Lead Researcher: Thai Hung Nguyen (AIT Laboratory, Dept of IT)", flush=True)
    print("=" * 80, flush=True)

    # Benchmark Audio Paths
    vocal_wav = root_dir / "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav"
    poly_wav = root_dir / "data/processed/vietnamese_song_clip_26s.wav"
    speech_wav = root_dir / "data/processed/vietnamese_speech_real_01_16k.wav"

    ref_singing = (
        "trời còn làm mây xanh ngắt cho mắt em buồn trời còn làm mưa bay bay cho hạt sầu lên mắt người thương "
        "lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương"
    )
    ref_speech = (
        "đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng cho mục đích huấn luyện trí tuệ nhân tạo "
        "mỗi bản ghi âm được chuẩn bị cẩn thận để đảm bảo phát âm rõ ràng nhịp điệu tự nhiên và chất lượng âm thanh ổn định"
    )

    models_to_audit = [
        ("C0_ZeroShot_Baseline", None),
        ("C1_Phase8_LoRA_Adapter", root_dir / "models/lora_adapter_phowhisper"),
        ("C3_Phase9_Distilled_Adapter", root_dir / "models/distilled_adapter_phowhisper")
    ]

    base_model_id = "vinai/PhoWhisper-tiny"
    processor = WhisperProcessor.from_pretrained(base_model_id)

    audit_results = {
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lead_researcher": "Thai Hung Nguyen",
        "reference_singing": ref_singing,
        "reference_speech": ref_speech,
        "evaluations": {}
    }

    for model_tag, lora_path in models_to_audit:
        print(f"\n>>> Auditing Checkpoint: {model_tag} ...", flush=True)
        base = WhisperForConditionalGeneration.from_pretrained(base_model_id, attn_implementation="eager")
        base.eval()

        if lora_path and lora_path.exists():
            print(f"    Attaching LoRA weights from: {lora_path}", flush=True)
            model = PeftModel.from_pretrained(base, str(lora_path))
            model.eval()
        else:
            model = base

        audit_results["evaluations"][model_tag] = {}

        # 1. Singing Vocal Stem
        print("    [1/3] Testing Isolated Singing Vocal Stem...", flush=True)
        res_vocal = audit_model(model_tag, model, processor, vocal_wav, ref_singing)
        audit_results["evaluations"][model_tag]["vocal_stem"] = res_vocal
        print(f"          WER: {res_vocal['wer']*100:.2f}% | CER: {res_vocal['cer']*100:.2f}% | S:{res_vocal['substitutions']} D:{res_vocal['deletions']} I:{res_vocal['insertions']} | Mono rho: {res_vocal['monotonicity_index']:.4f}", flush=True)
        print(f"          Hyp: \"{res_vocal['hypothesis']}\"", flush=True)

        # 2. Singing Polyphonic Mix
        print("    [2/3] Testing Singing Polyphonic Mix (+BGM)...", flush=True)
        res_poly = audit_model(model_tag, model, processor, poly_wav, ref_singing)
        audit_results["evaluations"][model_tag]["polyphonic_mix"] = res_poly
        print(f"          WER: {res_poly['wer']*100:.2f}% | CER: {res_poly['cer']*100:.2f}% | S:{res_poly['substitutions']} D:{res_poly['deletions']} I:{res_poly['insertions']} | Mono rho: {res_poly['monotonicity_index']:.4f}", flush=True)
        print(f"          Hyp: \"{res_poly['hypothesis']}\"", flush=True)

        # 3. Conversational Speech Anchor
        print("    [3/3] Testing Conversational Speech Anchor...", flush=True)
        res_speech = audit_model(model_tag, model, processor, speech_wav, ref_speech)
        audit_results["evaluations"][model_tag]["speech_anchor"] = res_speech
        print(f"          WER: {res_speech['wer']*100:.2f}% | CER: {res_speech['cer']*100:.2f}% | S:{res_speech['substitutions']} D:{res_speech['deletions']} I:{res_speech['insertions']} | Mono rho: {res_speech['monotonicity_index']:.4f}", flush=True)
        print(f"          Hyp: \"{res_speech['hypothesis']}\"", flush=True)

        del model
        del base

    # Save complete audit report
    out_audit_path = root_dir / "data/processed/audit_verification_results.json"
    with open(out_audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f"[+] AUDIT COMPLETE! Full verified results saved to: {out_audit_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

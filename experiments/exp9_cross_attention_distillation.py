"""
SING-ASR Experiment 9: Cross-Attention Guided Knowledge Distillation Benchmark (Phase IX).
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Investigates:
1. Impact of Cross-Attention Diagonality Regularization (L_attn) on vowel dwell and looping.
2. Impact of Attention Centroid Monotonicity Regularization (L_mono) on stalling elimination.
3. Comparative Recognition Accuracy across 4 Experimental Conditions:
   - C0: Zero-Shot Foundation Baseline (PhoWhisper-tiny)
   - C1: LoRA Fine-Tuning Only (Phase VIII Champion)
   - C2: Vanilla Logit KD (L_ce + L_kd)
   - C3: Cross-Attention Guided Alignment Distillation (Proposed Phase IX: L_total)
4. Dual-Domain Evaluation:
   - Singing Voice (Vocal Stem & Polyphonic Accompaniment)
   - Conversational Spoken Anchor (Catastrophic Forgetting Diagnostic)
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from train.dataset import create_singing_paired_corpus
from train.distillation_loss import DistillationHyperparameters
from train.distillation import DistillationTrainer, get_safe_device
from asr.attention_monitor import analyze_cross_attention
from dsp.audio_io import load_audio
from evaluation.metrics import compute_metrics
from evaluation.alignment import align_words


def run_experiment_9():
    print("================================================================================")
    print("SING-ASR Experiment 9: Cross-Attention Guided Knowledge Distillation (Phase IX)")
    print("Lead Researcher: Thai Hung Nguyen (AIT Laboratory, Dept of IT)")
    print("================================================================================")

    device = get_safe_device()
    print(f"[*] Compute Execution Device: {device}")

    # Paths
    vocal_wav = root_dir / "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav"
    poly_wav = root_dir / "data/processed/vietnamese_song_clip_26s.wav"
    speech_wav = root_dir / "data/processed/vietnamese_speech_real_01_16k.wav"
    lora_phase8_dir = root_dir / "models/lora_adapter_phowhisper"

    ref_singing = (
        "Trời còn làm mây xanh ngắt cho mắt em buồn trời còn làm mưa bay bay cho hạt sầu lên mắt người thương "
        "lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương"
    )
    ref_speech = (
        "đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng cho mục đích huấn luyện trí tuệ nhân tạo "
        "mỗi bản ghi âm được chuẩn bị cẩn thận để đảm bảo phát âm rõ ràng nhịp điệu tự nhiên và chất lượng âm thanh ổn định"
    )

    corpus = create_singing_paired_corpus(root_dir)
    print(f"[*] Training Corpus Instances: {len(corpus)}")

    results = {
        "experiment": "Exp 9: Cross-Attention Guided Knowledge Distillation",
        "date": "2026-10-14",
        "lead_researcher": "Thai Hung Nguyen",
        "device": str(device),
        "conditions": {}
    }

    # -------------------------------------------------------------------------
    # Condition 0: Zero-Shot Foundation Baseline (PhoWhisper-tiny)
    # -------------------------------------------------------------------------
    print("\n--- [Condition 0] Evaluating Zero-Shot Baseline (PhoWhisper-tiny) ---")
    proc = WhisperProcessor.from_pretrained("vinai/PhoWhisper-tiny")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        "vinai/PhoWhisper-tiny", attn_implementation="eager"
    ).to(device)
    base_model.eval()

    def eval_baseline(audio_path, ref_text):
        audio, sr = load_audio(audio_path, target_sr=16000)
        t0 = time.time()
        feats = proc(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            gen_out = base_model.generate(
                feats, max_new_tokens=128, repetition_penalty=1.2,
                no_repeat_ngram_size=3, output_attentions=True, return_dict_in_generate=True
            )
        hyp = proc.batch_decode(gen_out.sequences, skip_special_tokens=True)[0].strip()
        dur = len(audio) / sr
        rtf = round((time.time() - t0) / dur, 4)
        m = compute_metrics(ref_text, hyp)
        al = align_words(ref_text, hyp)

        path = []
        if gen_out.cross_attentions:
            for step_attn in gen_out.cross_attentions:
                if step_attn:
                    avg_1d = step_attn[-1].mean(dim=(0, 1, 2)).cpu().numpy()
                    path.append(int(np.argmax(avg_1d)))
        import scipy.stats
        if len(path) > 2 and np.std(path) > 0:
            rho, _ = scipy.stats.spearmanr(np.arange(len(path)), path)
            mono_idx = round(float(rho) if not np.isnan(rho) else 0.0, 4)
        else:
            mono_idx = 0.0
        diffs = np.diff(path) if len(path) > 1 else np.array([0])
        stalled = int(np.sum(np.abs(diffs) <= 1))
        stall_ratio = round(float(stalled / len(diffs)) if len(diffs) > 0 else 0.0, 4)

        return {
            "hypothesis": hyp, "wer": round(m.wer, 4), "cer": round(m.cer, 4),
            "substitutions": al.substitutions, "deletions": al.deletions, "insertions": al.insertions,
            "rtf": rtf, "monotonicity_index": mono_idx, "stalling_ratio": stall_ratio
        }

    c0_vocal = eval_baseline(vocal_wav, ref_singing)
    c0_poly = eval_baseline(poly_wav, ref_singing)
    c0_speech = eval_baseline(speech_wav, ref_speech)
    print(f"  C0 Vocal  WER: {c0_vocal['wer']*100:.2f}% | CER: {c0_vocal['cer']*100:.2f}% | Mono rho: {c0_vocal['monotonicity_index']:.4f} | Stall: {c0_vocal['stalling_ratio']*100:.1f}%")
    print(f"  C0 Poly   WER: {c0_poly['wer']*100:.2f}% | CER: {c0_poly['cer']*100:.2f}% | Mono rho: {c0_poly['monotonicity_index']:.4f} | Stall: {c0_poly['stalling_ratio']*100:.1f}%")
    print(f"  C0 Speech WER: {c0_speech['wer']*100:.2f}% | CER: {c0_speech['cer']*100:.2f}% | Mono rho: {c0_speech['monotonicity_index']:.4f} | Stall: {c0_speech['stalling_ratio']*100:.1f}%")

    results["conditions"]["C0_ZeroShot_Baseline"] = {
        "description": "Unadapted Zero-Shot PhoWhisper-tiny",
        "vocal": c0_vocal, "poly": c0_poly, "speech": c0_speech
    }
    del base_model

    # -------------------------------------------------------------------------
    # Condition 1: Phase VIII LoRA Baseline (Supervised Fine-Tuning Only)
    # -------------------------------------------------------------------------
    print("\n--- [Condition 1] Evaluating Phase VIII LoRA Model ---")
    trainer_c1 = DistillationTrainer(
        base_model_id="vinai/PhoWhisper-tiny",
        initial_lora_dir=lora_phase8_dir if lora_phase8_dir.exists() else None,
        hparams=DistillationHyperparameters(alpha_ce=1.0, beta_kd=0.0, gamma_attn=0.0, lambda_mono=0.0),
        device=device
    )
    c1_vocal = trainer_c1.evaluate_audio_and_attention(vocal_wav, ref_singing)
    c1_poly = trainer_c1.evaluate_audio_and_attention(poly_wav, ref_singing)
    c1_speech = trainer_c1.evaluate_audio_and_attention(speech_wav, ref_speech)
    print(f"  C1 Vocal  WER: {c1_vocal['wer']*100:.2f}% | CER: {c1_vocal['cer']*100:.2f}% | Mono rho: {c1_vocal['monotonicity_index']:.4f} | Stall: {c1_vocal['stalling_ratio']*100:.1f}%")
    print(f"  C1 Poly   WER: {c1_poly['wer']*100:.2f}% | CER: {c1_poly['cer']*100:.2f}% | Mono rho: {c1_poly['monotonicity_index']:.4f} | Stall: {c1_poly['stalling_ratio']*100:.1f}%")
    print(f"  C1 Speech WER: {c1_speech['wer']*100:.2f}% | CER: {c1_speech['cer']*100:.2f}% | Mono rho: {c1_speech['monotonicity_index']:.4f} | Stall: {c1_speech['stalling_ratio']*100:.1f}%")

    results["conditions"]["C1_LoRA_FineTuned_Only"] = {
        "description": "PhoWhisper-tiny + LoRA adapted with CE loss (Phase VIII)",
        "vocal": c1_vocal, "poly": c1_poly, "speech": c1_speech
    }
    del trainer_c1

    # -------------------------------------------------------------------------
    # Condition 2: Vanilla Logit Distillation (L_ce + L_kd)
    # -------------------------------------------------------------------------
    print("\n--- [Condition 2] Training Vanilla Logit Distillation ---")
    trainer_c2 = DistillationTrainer(
        base_model_id="vinai/PhoWhisper-tiny",
        initial_lora_dir=lora_phase8_dir if lora_phase8_dir.exists() else None,
        hparams=DistillationHyperparameters(alpha_ce=1.0, beta_kd=0.5, gamma_attn=0.0, lambda_mono=0.0),
        device=device
    )
    c2_history = trainer_c2.train(corpus, epochs=3, lr=5e-4)
    c2_vocal = trainer_c2.evaluate_audio_and_attention(vocal_wav, ref_singing)
    c2_poly = trainer_c2.evaluate_audio_and_attention(poly_wav, ref_singing)
    c2_speech = trainer_c2.evaluate_audio_and_attention(speech_wav, ref_speech)
    print(f"  C2 Vocal  WER: {c2_vocal['wer']*100:.2f}% | CER: {c2_vocal['cer']*100:.2f}% | Mono rho: {c2_vocal['monotonicity_index']:.4f} | Stall: {c2_vocal['stalling_ratio']*100:.1f}%")
    print(f"  C2 Poly   WER: {c2_poly['wer']*100:.2f}% | CER: {c2_poly['cer']*100:.2f}% | Mono rho: {c2_poly['monotonicity_index']:.4f} | Stall: {c2_poly['stalling_ratio']*100:.1f}%")
    print(f"  C2 Speech WER: {c2_speech['wer']*100:.2f}% | CER: {c2_speech['cer']*100:.2f}% | Mono rho: {c2_speech['monotonicity_index']:.4f} | Stall: {c2_speech['stalling_ratio']*100:.1f}%")

    results["conditions"]["C2_Vanilla_Logit_KD"] = {
        "description": "Distillation with Cross-Entropy + Soft Logits KL Divergence",
        "train_history": c2_history,
        "vocal": c2_vocal, "poly": c2_poly, "speech": c2_speech
    }
    del trainer_c2

    # -------------------------------------------------------------------------
    # Condition 3: Full Cross-Attention Guided Alignment Distillation (Phase IX Proposed)
    # -------------------------------------------------------------------------
    print("\n--- [Condition 3] Training Full Cross-Attention Guided Distillation (Proposed) ---")
    distilled_save_dir = root_dir / "models/distilled_adapter_phowhisper"
    trainer_c3 = DistillationTrainer(
        base_model_id="vinai/PhoWhisper-tiny",
        initial_lora_dir=lora_phase8_dir if lora_phase8_dir.exists() else None,
        hparams=DistillationHyperparameters(
            alpha_ce=1.0,
            beta_kd=0.5,
            gamma_attn=2.5,     # Strong Gaussian diagonal penalty
            lambda_mono=1.5,    # Strong forward centroid progression constraint
            sigma=0.15,
            min_progression=0.5
        ),
        device=device
    )
    c3_history = trainer_c3.train(corpus, epochs=4, lr=5e-4, save_dir=distilled_save_dir)
    c3_vocal = trainer_c3.evaluate_audio_and_attention(vocal_wav, ref_singing)
    c3_poly = trainer_c3.evaluate_audio_and_attention(poly_wav, ref_singing)
    c3_speech = trainer_c3.evaluate_audio_and_attention(speech_wav, ref_speech)
    print(f"  C3 Vocal  WER: {c3_vocal['wer']*100:.2f}% | CER: {c3_vocal['cer']*100:.2f}% | Mono rho: {c3_vocal['monotonicity_index']:.4f} | Stall: {c3_vocal['stalling_ratio']*100:.1f}%")
    print(f"  C3 Poly   WER: {c3_poly['wer']*100:.2f}% | CER: {c3_poly['cer']*100:.2f}% | Mono rho: {c3_poly['monotonicity_index']:.4f} | Stall: {c3_poly['stalling_ratio']*100:.1f}%")
    print(f"  C3 Speech WER: {c3_speech['wer']*100:.2f}% | CER: {c3_speech['cer']*100:.2f}% | Mono rho: {c3_speech['monotonicity_index']:.4f} | Stall: {c3_speech['stalling_ratio']*100:.1f}%")

    results["conditions"]["C3_Full_Guided_Attention_Distillation"] = {
        "description": "Proposed Phase IX: CE + KD + Guided Diagonality + Monotonicity Regularization",
        "train_history": c3_history,
        "vocal": c3_vocal, "poly": c3_poly, "speech": c3_speech
    }
    del trainer_c3

    # Save final JSON results
    out_file = root_dir / "data/processed/exp9_distillation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Experiment 9 complete. Benchmark saved to: {out_file}")


if __name__ == "__main__":
    run_experiment_9()

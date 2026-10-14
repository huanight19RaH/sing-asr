"""
SING-ASR Experiment 9.B: Deep Bias-Variance Regularization & Early Stopping Sweep.
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Benchmarking empirical performance across:
1. C0: PhoWhisper Zero-Shot Baseline
2. C1: Phase VIII LoRA Baseline (r=8, Attention-only, unregularized)
3. C3: Phase IX Cross-Attention Distillation (Static sigma=0.15)
4. C4: Phase IX.B Regularized Champion:
   - Utterance-level Spectrogram CMVN + Waveform RMS Normalization
   - All-Linear LoRA (Attn + Decoder MLPs fc1, fc2, r=16, alpha=32)
   - Neural Toggling via LoRA Dropout (p = 0.10)
   - Decoupled L2 Weight Decay (lambda = 0.05) + L1 Weight Sparsity Penalty
   - Dynamic Adaptive-Pacing sigma(t) + Smooth Huber-Monotonicity Loss
   - Multi-Task Speech Replay Balancing Term (mu = 0.75)
   - Multi-Domain Pareto Early Stopping (Poly 40%, Vocal 30%, Speech 30%)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Safe CPU execution for local environment

import json
from pathlib import Path
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import time
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from train.dataset import create_singing_paired_corpus
from train.regularized_trainer import RegularizedTrainer, RegularizedTrainerConfig
from train.regularized_loss import RegularizedLossConfig
from train.early_stopping import EarlyStoppingConfig
from evaluation.metrics import compute_metrics
from evaluation.alignment import align_words


def main():
    print("=" * 80)
    print("SING-ASR PHASE IX.B: BIAS-VARIANCE REGULARIZATION & PARETO BENCHMARK")
    print("Researcher: Thai Hung Nguyen | AIT Laboratory")
    print("=" * 80)

    # 1. Define Benchmark Assets
    poly_wav = root_dir / "data/processed/vietnamese_song_clip_26s.wav"
    vocal_wav = root_dir / "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav"
    speech_wav = root_dir / "data/processed/vietnamese_speech_real_01_16k.wav"

    ref_lyrics = (
        "Trời còn làm mây xanh ngắt cho mắt em buồn trời còn làm mưa bay bay cho hạt sầu "
        "lên mắt người thương lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương"
    )
    ref_speech = (
        "đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng cho mục đích huấn luyện trí tuệ nhân tạo "
        "mỗi bản ghi âm được chuẩn bị cẩn thận để đảm bảo phát âm rõ ràng nhịp điệu tự nhiên và chất lượng âm thanh ổn định"
    )

    val_audio_dict = {
        "polyphonic_mix": (poly_wav, ref_lyrics),
        "vocal_stem": (vocal_wav, ref_lyrics),
        "spoken_speech": (speech_wav, ref_speech)
    }

    # 2. Build Paired Training Corpus
    print("\n[Phase IX.B] Building paired singing & speech replay corpus...")
    corpus = create_singing_paired_corpus(root_dir)
    print(f"[Phase IX.B] Created corpus with {len(corpus)} instances (singing bars + spoken anchor).")

    # 3. Configure Phase IX.B Regularized Champion Trainer
    loss_cfg = RegularizedLossConfig(
        alpha_ce=1.0,
        beta_kd=0.5,
        gamma_attn=1.5,
        lambda_mono=1.0,
        lambda_l1=1e-4,
        mu_replay=0.75,
        label_smoothing=0.10,
        base_sigma=0.20,
        huber_delta=1.0
    )

    es_cfg = EarlyStoppingConfig(
        patience=4,
        min_delta=0.005,
        w_poly=0.40,
        w_vocal=0.30,
        w_speech=0.30,
        restore_best_weights=True
    )

    trainer_cfg = RegularizedTrainerConfig(
        base_model_id="vinai/PhoWhisper-tiny",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.10,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lr=3e-4,
        weight_decay=0.05,
        epochs=8,
        warmup_ratio=0.10,
        normalize_cmvn=True,
        normalize_rms=True,
        loss_config=loss_cfg,
        early_stopping_config=es_cfg
    )

    champion_save_dir = root_dir / "models/regularized_champion_phowhisper"

    trainer = RegularizedTrainer(
        config=trainer_cfg,
        device=torch.device("cpu")
    )

    # 4. Train with Early Stopping
    print("\n[Phase IX.B] Launching Multi-Domain Pareto Training...")
    training_summary = trainer.train_with_early_stopping(
        corpus=corpus,
        val_audio_dict=val_audio_dict,
        epochs=8,
        batch_size=4,
        save_dir=champion_save_dir
    )

    # 5. Evaluate Champion on all 3 domains
    print("\n[Phase IX.B] Running Final Benchmark Evaluation on Champion Model...", flush=True)
    champ_results = {}
    for dom_name, (audio_p, ref_text) in val_audio_dict.items():
        eval_res = trainer.evaluate_audio_and_attention(audio_p, ref_text, apply_cmvn=True)
        champ_results[dom_name] = eval_res
        m = eval_res["metrics"]
        print(f"  [{dom_name.upper()}] WER: {m['wer']*100:.2f}% | CER: {m['cer']*100:.2f}% | S: {m['substitutions']} | D: {m['deletions']} | I: {m['insertions']} | Mono rho: {eval_res['monotonicity_rho']} | Stalling: {eval_res['stalling_ratio']:.2%}", flush=True)
        print(f"    Hyp: '{eval_res['hypothesis'][:80]}...'", flush=True)

    # 6. Load C0, C1, C3 from verified audit to create Master Comparative Table
    audit_file = root_dir / "data/processed/audit_verification_results.json"
    audit_data = {}
    if audit_file.exists():
        with open(audit_file, "r", encoding="utf-8") as f:
            audit_data = json.load(f)

    c4_pareto = (
        es_cfg.w_poly * (champ_results["polyphonic_mix"]["metrics"]["wer"] * 100.0)
        + es_cfg.w_vocal * (champ_results["vocal_stem"]["metrics"]["wer"] * 100.0)
        + es_cfg.w_speech * (champ_results["spoken_speech"]["metrics"]["wer"] * 100.0)
    )

    full_comparison = {
        "metadata": {
            "experiment": "Phase IX.B - Deep Bias-Variance Regularization & Early Stopping",
            "lead_researcher": "Thai Hung Nguyen",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": "CPU (Intel / Safe Environment)",
            "regularization_mechanisms": [
                "Utterance-level Spectrogram CMVN",
                "Waveform RMS Level Normalization (-20 dBFS)",
                "LoRA All-Linear Targets (q, k, v, out, fc1, fc2)",
                "LoRA Structured Dropout (p = 0.10)",
                "Decoupled L2 Weight Decay (0.05)",
                "L1 Weight Sparsity Penalty (lambda = 1e-4)",
                "Label Smoothing Cross-Entropy (0.10)",
                "Dynamic Adaptive-Pacing Gaussian Mask sigma(t)",
                "Smooth Huber-Monotonicity Penalty (delta = 1.0)",
                "Multi-Task Conversational Speech Replay Term (mu = 0.75)",
                "Multi-Domain Pareto Early Stopping (patience = 4)"
            ]
        },
        "training_history": training_summary["history"],
        "best_epoch": training_summary["best_epoch"],
        "champion_pareto_score": round(c4_pareto, 4),
        "champion_metrics": champ_results,
        "historical_benchmarks": audit_data.get("models", {})
    }

    out_file = root_dir / "data/processed/exp9b_regularization_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_comparison, f, indent=2, ensure_ascii=False)

    print(f"\n[Phase IX.B] Results saved to: {out_file}")
    print("=" * 80)
    print("PHASE IX.B BENCHMARK COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()

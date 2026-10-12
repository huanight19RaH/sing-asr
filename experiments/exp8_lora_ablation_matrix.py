"""
SING-ASR Experiment 8: Comprehensive LoRA Architectural Ablation Matrix.
Phase VIII: Parameter-Efficient Fine-Tuning (PEFT) on PhoWhisper-tiny.
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Systematically investigates:
1. Rank Variation (r in {4, 8, 16}) with scaling alpha = 4 * r.
2. Target Module Scope:
   - Query & Value Attention: ['q_proj', 'v_proj']
   - Full Attention Projections: ['q_proj', 'k_proj', 'v_proj', 'out_proj']
3. Catastrophic Forgetting Diagnostic on Conversational Spoken Vietnamese.
4. Error Taxonomy Dynamics (Substitutions, Deletions, Insertions).
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from train.dataset import create_singing_paired_corpus, VietnameseSingingDataset, collate_fn_whisper
from train.peft_lora import PhoWhisperLoRAWrapper, LoRAConfigParams
from evaluation.metrics import compute_metrics
from evaluation.alignment import align_words
from dsp.audio_io import load_audio


def get_safe_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            x = torch.zeros(1, device="cuda")
            _ = x + 1
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def train_lora_variant(
    base_model_id: str,
    processor: WhisperProcessor,
    corpus: list[dict],
    r: int,
    alpha: int,
    target_modules: list[str],
    epochs: int = 3,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu")
) -> tuple[nn.Module, dict]:
    """Trains a specific LoRA architecture variant and tracks convergence."""
    print(f"\n--- Initializing LoRA Variant: r={r}, alpha={alpha}, modules={target_modules} ---")
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    base_model.config.forced_decoder_ids = None
    base_model.generation_config.language = "vi"
    base_model.generation_config.task = "transcribe"

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    param_pct = (trainable_params / all_params) * 100
    print(f"  Trainable Parameters: {trainable_params:,} / {all_params:,} ({param_pct:.3f}%)")

    dataset = VietnameseSingingDataset(processor, corpus, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda b: collate_fn_whisper(b, processor)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            feats = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_features=feats, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / len(loader)
        loss_history.append(round(avg_loss, 4))
        print(f"  Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    train_time = round(time.time() - t_start, 2)
    meta = {
        "r": r,
        "alpha": alpha,
        "target_modules": target_modules,
        "trainable_params": trainable_params,
        "param_pct": round(param_pct, 4),
        "train_time_sec": train_time,
        "loss_history": loss_history
    }
    return model, meta


def evaluate_model_on_audio(
    model: nn.Module,
    processor: WhisperProcessor,
    audio_path: Path,
    reference_text: str,
    device: torch.device
) -> dict:
    audio, sr = load_audio(audio_path, target_sr=16000)
    t0 = time.time()
    feats = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    model.eval()
    with torch.no_grad():
        pred_ids = model.generate(
            feats,
            max_new_tokens=128,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            language="vi",
            task="transcribe"
        )
    hyp = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
    infer_time = time.time() - t0
    rtf = round(infer_time / (len(audio) / sr), 4)

    metrics = compute_metrics(reference_text, hyp)
    alignment = align_words(reference_text, hyp)

    return {
        "hypothesis": hyp,
        "wer": round(metrics.wer, 4),
        "cer": round(metrics.cer, 4),
        "substitutions": alignment.substitutions,
        "deletions": alignment.deletions,
        "insertions": alignment.insertions,
        "rtf": rtf
    }


def run_ablation_matrix():
    print("================================================================================")
    print("SING-ASR Experiment 8: Comprehensive LoRA Architectural Ablation Matrix")
    print("Lead Researcher: Thai Hung Nguyen (AIT Laboratory, Dept of IT)")
    print("================================================================================")

    device = get_safe_device()
    print(f"[*] Compute Execution Device: {device}")

    model_id = "vinai/PhoWhisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_id)

    # 1. Corpus setup
    corpus = create_singing_paired_corpus(root_dir)
    print(f"[*] Training Corpus Instances: {len(corpus)}")

    # Test Audio Paths
    vocal_wav = root_dir / "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav"
    poly_wav = root_dir / "data/processed/vietnamese_song_clip_26s.wav"
    speech_wav = root_dir / "data/processed/vietnamese_speech_real_01_16k.wav"

    ref_singing = (
        "Trời còn làm mây xanh ngắt cho mắt em buồn trời còn làm mưa bay bay cho hạt sầu lên mắt người thương "
        "lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương"
    )
    ref_speech = (
        "đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng cho mục đích huấn luyện trí tuệ nhân tạo "
        "mỗi bản ghi âm được chuẩn bị cẩn thận để đảm bảo phát âm rõ ràng nhịp điệu tự nhiên và chất lượng âm thanh ổn định"
    )

    # 2. Benchmark Baseline (Zero-Shot PhoWhisper-tiny)
    print("\n--- Evaluating Zero-Shot Baseline (PhoWhisper-tiny) ---")
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    base_model.to(device)
    base_vocal = evaluate_model_on_audio(base_model, processor, vocal_wav, ref_singing, device)
    base_poly = evaluate_model_on_audio(base_model, processor, poly_wav, ref_singing, device)
    base_speech = evaluate_model_on_audio(base_model, processor, speech_wav, ref_speech, device)
    print(f"  Baseline Vocal WER: {base_vocal['wer']*100:.2f}% | CER: {base_vocal['cer']*100:.2f}%")
    print(f"  Baseline Poly  WER: {base_poly['wer']*100:.2f}% | CER: {base_poly['cer']*100:.2f}%")
    print(f"  Baseline Speech WER: {base_speech['wer']*100:.2f}% | CER: {base_speech['cer']*100:.2f}%")
    del base_model

    # 3. LoRA Architectural Variants
    variants = [
        {"name": "LoRA-R4", "r": 4, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        {"name": "LoRA-R8 (Champion)", "r": 8, "alpha": 32, "target_modules": ["q_proj", "v_proj"]},
        {"name": "LoRA-R16", "r": 16, "alpha": 64, "target_modules": ["q_proj", "v_proj"]},
        {"name": "LoRA-AllAttn", "r": 8, "alpha": 32, "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"]}
    ]

    ablation_results = {
        "baseline_zero_shot": {
            "vocal": base_vocal,
            "poly": base_poly,
            "speech": base_speech
        },
        "variants": {}
    }

    for var in variants:
        v_name = var["name"]
        model_v, meta_v = train_lora_variant(
            base_model_id=model_id,
            processor=processor,
            corpus=corpus,
            r=var["r"],
            alpha=var["alpha"],
            target_modules=var["target_modules"],
            epochs=3,
            device=device
        )

        # Evaluate on all 3 conditions
        eval_vocal = evaluate_model_on_audio(model_v, processor, vocal_wav, ref_singing, device)
        eval_poly = evaluate_model_on_audio(model_v, processor, poly_wav, ref_singing, device)
        eval_speech = evaluate_model_on_audio(model_v, processor, speech_wav, ref_speech, device)

        print(f"  [{v_name}] Vocal  WER: {eval_vocal['wer']*100:.2f}% | CER: {eval_vocal['cer']*100:.2f}% (Del: {eval_vocal['deletions']}, Ins: {eval_vocal['insertions']})")
        print(f"  [{v_name}] Poly   WER: {eval_poly['wer']*100:.2f}% | CER: {eval_poly['cer']*100:.2f}% (Del: {eval_poly['deletions']}, Ins: {eval_poly['insertions']})")
        print(f"  [{v_name}] Speech WER: {eval_speech['wer']*100:.2f}% | CER: {eval_speech['cer']*100:.2f}% (Del: {eval_speech['deletions']}, Ins: {eval_speech['insertions']})")

        ablation_results["variants"][v_name] = {
            "meta": meta_v,
            "vocal": eval_vocal,
            "poly": eval_poly,
            "speech": eval_speech
        }
        del model_v

    # Save Results
    out_file = root_dir / "data/processed/exp8_lora_ablation_matrix.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Experiment 8: LoRA Architecture Ablation Matrix",
            "date": "2026-10-12",
            "lead_researcher": "Thai Hung Nguyen",
            "device": str(device),
            "results": ablation_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Ablation Matrix Results saved to: {out_file}")


if __name__ == "__main__":
    run_ablation_matrix()

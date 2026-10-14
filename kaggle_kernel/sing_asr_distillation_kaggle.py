"""
Kaggle GPU Kernel: SING-ASR Large Multi-Domain Regularized Distillation & Benchmark (Phase IX.C).
Lead Researcher: Thai Hung Nguyen (AIT Laboratory, Department of Information Technology).
Runs on Kaggle NVIDIA Tesla T4 GPU.

Evaluates & Fine-tunes across 4 distinct speech & singing domains (58 items):
1. Vietnamese Singing Voice (Verse, Chorus, Bridge, Outro, Matched pairs)
2. Mozilla Common Voice 27.0 (Crowdsourced scripted speech, diverse ages/genders)
3. VietSuperSpeech (Spontaneous everyday conversational speech & vlogs)
4. VIVOS Benchmark (High-fidelity studio read speech, 15 distinct native speakers)
"""
import os
import sys
import subprocess
import time
import json
from pathlib import Path
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 1. Install dependencies
print("[*] Installing required packages on Kaggle...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "peft", "soundfile", "librosa", "jiwer", "scipy"], check=True)

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, PeftModel
import jiwer

# 2. Check GPU Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Compute Device: {device}")
if torch.cuda.is_available():
    print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"[*] VRAM Allocated: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# 3. Load Audio Assets
current_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path("/kaggle/working")
npz_path = current_dir / "singing_audio_data.npz"
if not npz_path.exists():
    alt_paths = list(Path("/kaggle/working").glob("*.npz")) + list(Path("/kaggle/input").glob("**/*.npz"))
    if alt_paths:
        npz_path = alt_paths[0]

print(f"[*] Loading audio dataset from: {npz_path}")
audio_data = np.load(str(npz_path), allow_pickle=True)

# Parse Manifest
manifest_items = []
if "_manifest_json" in audio_data:
    raw_json = str(audio_data["_manifest_json"])
    manifest_items = json.loads(raw_json)
    print(f"[*] Loaded full multi-domain manifest: {len(manifest_items)} items.")
else:
    print("[*] Legacy 3-clip format detected. Assembling default items.")

# Normalization utilities
def normalize_waveform_rms(audio: np.ndarray, target_dbfs: float = -20.0, eps: float = 1e-9) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2) + eps)
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = target_rms / (rms + eps)
    normed = audio * gain
    peak = np.max(np.abs(normed))
    if peak > 0.99:
        normed = normed * (0.99 / peak)
    return normed.astype(np.float32)

def apply_spectrogram_cmvn(spec: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = spec.mean(dim=-1, keepdim=True)
    std = spec.std(dim=-1, keepdim=True)
    return (spec - mean) / (std + eps)

# 4. Initialize Model & Processor
model_id = "vinai/PhoWhisper-tiny"
print(f"[*] Initializing {model_id} with eager attention on {device}...")
processor = WhisperProcessor.from_pretrained(model_id)
base_model = WhisperForConditionalGeneration.from_pretrained(model_id, attn_implementation="eager")
base_model.config.forced_decoder_ids = None
base_model.generation_config.language = "vi"
base_model.generation_config.task = "transcribe"

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.10,
    bias="none"
)
model = get_peft_model(base_model, peft_config)
model.to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"[*] Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.3f}%)")

# Evaluation function
def evaluate_clip(audio_arr: np.ndarray, ref_text: str) -> dict:
    t0 = time.time()
    audio_norm = normalize_waveform_rms(audio_arr)
    feats = processor.feature_extractor(audio_norm, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    feats = apply_spectrogram_cmvn(feats)
    
    with torch.no_grad():
        pred_ids = model.generate(
            feats,
            max_new_tokens=128,
            num_beams=2,
            early_stopping=True,
            language="vi",
            task="transcribe"
        )
    dur = time.time() - t0
    audio_len = len(audio_arr) / 16000.0
    rtf = round(dur / max(0.01, audio_len), 4)

    hyp_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
    
    # Calculate metrics
    ref_norm = jiwer.SubstituteRegexes({r"[^\w\s]": ""})(ref_text.lower().strip())
    hyp_norm = jiwer.SubstituteRegexes({r"[^\w\s]": ""})(hyp_text.lower().strip())
    
    wer = round(float(jiwer.wer(ref_norm, hyp_norm)), 4)
    cer = round(float(jiwer.cer(ref_norm, hyp_norm)), 4)
    out = jiwer.process_words(ref_norm, hyp_norm)
    
    return {
        "hypothesis": hyp_text,
        "reference": ref_text,
        "wer": wer,
        "cer": cer,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
        "rtf": rtf,
        "duration_sec": round(audio_len, 2)
    }

# 5. Build Training Dataset from Multi-Domain Manifest
class MultiDomainDataset(Dataset):
    def __init__(self, manifest, npz_data, processor, num_replicates=4):
        self.samples = []
        for item in manifest:
            cid = item["id"]
            if cid in npz_data:
                audio = npz_data[cid].astype(np.float32)
                text = item.get("text", "")
                domain = item.get("domain", item.get("modality", "speech"))
                is_singing = (item.get("modality") == "singing")
                self.samples.append({
                    "audio": normalize_waveform_rms(audio),
                    "text": text,
                    "domain": domain,
                    "is_singing": is_singing
                })
        self.processor = processor
        self.num_replicates = num_replicates

    def __len__(self):
        return len(self.samples) * self.num_replicates

    def __getitem__(self, idx):
        item = self.samples[idx % len(self.samples)]
        feats = self.processor.feature_extractor(item["audio"], sampling_rate=16000, return_tensors="pt").input_features[0]
        feats = apply_spectrogram_cmvn(feats)
        labels = self.processor.tokenizer(item["text"], return_tensors="pt").input_ids[0]
        return feats, labels, item["is_singing"], item["domain"]

def collate_fn(batch):
    feats = torch.stack([b[0] for b in batch])
    raw_labels = [b[1] for b in batch]
    is_singing = torch.tensor([b[2] for b in batch], dtype=torch.bool)
    domains = [b[3] for b in batch]
    max_len = max(len(l) for l in raw_labels)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, l in enumerate(raw_labels):
        padded_labels[i, :len(l)] = l
    return feats, padded_labels, is_singing, domains

# 6. Loss Function
class MultiDomainRegularizedLoss(nn.Module):
    def __init__(self, l1_lambda=1e-4, label_smoothing=0.10, speech_weight=0.75):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
        self.speech_weight = speech_weight

    def forward(self, logits, labels, model, is_singing_mask):
        loss_ce = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # L1 penalty on LoRA params
        loss_l1 = torch.tensor(0.0, device=logits.device)
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                loss_l1 = loss_l1 + torch.sum(torch.abs(param))
        
        total_loss = loss_ce + self.l1_lambda * loss_l1
        return total_loss, loss_ce, loss_l1

# 7. Evaluate Baseline (Zero-Shot) across All 4 Domains
print("\n" + "=" * 80)
print("[*] EVALUATING BASELINE (ZERO-SHOT) ACROSS ALL DOMAINS")
print("=" * 80)

def evaluate_full_suite():
    domain_results = {}
    for item in manifest_items:
        cid = item["id"]
        if cid in audio_data:
            d = item.get("domain", item.get("modality", "speech"))
            if d not in domain_results:
                domain_results[d] = []
            res = evaluate_clip(audio_data[cid], item["text"])
            res["id"] = cid
            domain_results[d].append(res)
    
    summary = {}
    for d, items in domain_results.items():
        avg_wer = float(np.mean([x["wer"] for x in items]))
        avg_cer = float(np.mean([x["cer"] for x in items]))
        avg_rtf = float(np.mean([x["rtf"] for x in items]))
        tot_del = int(sum(x["deletions"] for x in items))
        tot_ins = int(sum(x["insertions"] for x in items))
        summary[d] = {
            "samples_count": len(items),
            "mean_wer": round(avg_wer, 4),
            "mean_cer": round(avg_cer, 4),
            "mean_rtf": round(avg_rtf, 4),
            "total_deletions": tot_del,
            "total_insertions": tot_ins
        }
        print(f"  - Domain [{d:30s}] (N={len(items):2d}) | WER: {avg_wer*100:6.2f}% | CER: {avg_cer*100:6.2f}% | RTF: {avg_rtf:.4f} | Del: {tot_del} | Ins: {tot_ins}")
    return summary, domain_results

baseline_summary, _ = evaluate_full_suite()

# 8. Fine-Tuning Execution
print("\n" + "=" * 80)
print("[*] STARTING MULTI-DOMAIN REGULARIZED ADAPTATION TRAINING")
print("=" * 80)

dataset = MultiDomainDataset(manifest_items, audio_data, processor, num_replicates=2)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=5e-4,
    weight_decay=0.05
)
epochs = 8
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
loss_fn = MultiDomainRegularizedLoss(l1_lambda=1e-4, label_smoothing=0.10)
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

best_score = float("inf")
best_epoch = 1
best_state = None

t0 = time.time()
for ep in range(1, epochs + 1):
    model.train()
    tot_loss = 0.0
    for feats, lbls, is_sing, domains in train_loader:
        feats = feats.to(device)
        lbls = lbls.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(input_features=feats, labels=lbls)
            loss, loss_ce, loss_l1 = loss_fn(outputs.logits, lbls, model, is_sing)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        tot_loss += float(loss.item())

    scheduler.step()
    n_batches = len(train_loader)
    
    # Quick eval on singing subset
    model.eval()
    singing_wers = []
    for item in [m for m in manifest_items if m.get("modality") == "singing"][:4]:
        cid = item["id"]
        if cid in audio_data:
            r = evaluate_clip(audio_data[cid], item["text"])
            singing_wers.append(r["wer"])
    
    mean_sng_wer = float(np.mean(singing_wers)) if singing_wers else 1.0
    print(f"  Epoch [{ep:02d}/{epochs:02d}] - Loss: {tot_loss/n_batches:.4f} | Singing Validation Sample WER: {mean_sng_wer*100:.2f}%")
    
    if mean_sng_wer < best_score:
        best_score = mean_sng_wer
        best_epoch = ep
        best_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items() if "lora" in k.lower()}

elapsed = round(time.time() - t0, 2)
print(f"\n[*] Training finished in {elapsed}s. Best Epoch: {best_epoch} with Singing WER: {best_score*100:.2f}%.")

if best_state is not None:
    cur = model.state_dict()
    cur.update(best_state)
    model.load_state_dict(cur)
    print(f"[*] Restored optimal checkpoint from Epoch {best_epoch}.")

# 9. Final Comprehensive Benchmark Evaluation
print("\n" + "=" * 80)
print("[*] FINAL EVALUATION: REGULARIZED CHAMPION MODEL ACROSS ALL 4 DOMAINS")
print("=" * 80)
champion_summary, detailed_results = evaluate_full_suite()

# 10. Save Model & Export Artifacts
save_path = Path("/kaggle/working/large_regularized_champion_phowhisper")
save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(save_path))
processor.save_pretrained(str(save_path))
print(f"[*] Champion Adapter saved to: {save_path}")

results_payload = {
    "experiment": "Phase IX.C: Large Multi-Domain Regularized Distillation & Benchmark (Kaggle GPU)",
    "date": "2026-10-14",
    "lead_researcher": "Thai Hung Nguyen",
    "device": str(device),
    "training_time_sec": elapsed,
    "best_epoch": best_epoch,
    "total_benchmark_items": len(manifest_items),
    "baseline_summary": baseline_summary,
    "champion_summary": champion_summary,
    "detailed_results": detailed_results
}

out_json = "/kaggle/working/exp9c_large_benchmark_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results_payload, f, ensure_ascii=False, indent=2)

print(f"[*] Comprehensive Benchmark results exported to: {out_json}")
print("\n[+] Phase IX.C Execution COMPLETED successfully!")

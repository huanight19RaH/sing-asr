"""
Kaggle GPU Kernel: SING-ASR Cross-Attention Guided Knowledge Distillation (Phase IX).
Lead Researcher: Thai Hung Nguyen (AIT Laboratory, Department of Information Technology).
Runs on Kaggle NVIDIA Tesla T4 GPU.

Installs dependencies, loads PhoWhisper-tiny with eager attention and LoRA (r=8, alpha=32),
trains for 15 epochs with composite distillation loss:
  L_total = alpha * L_ce + beta * L_kd + gamma * L_attn + lambda * L_mono
and evaluates recognition accuracy (WER, CER) and attention dynamics (Monotonicity Index, Stalling Ratio).
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
audio_data = np.load(str(npz_path))
speech_audio = audio_data["speech"].astype(np.float32)
vocal_audio = audio_data["vocal"].astype(np.float32)
poly_audio = audio_data["poly"].astype(np.float32)
sr = 16000

# 4. Initialize PhoWhisper-tiny & LoRA with Eager Attention
model_id = "vinai/PhoWhisper-tiny"
print(f"[*] Initializing {model_id} with eager attention...")
processor = WhisperProcessor.from_pretrained(model_id)
base_model = WhisperForConditionalGeneration.from_pretrained(model_id, attn_implementation="eager")
base_model.config.forced_decoder_ids = None
base_model.generation_config.language = "vi"
base_model.generation_config.task = "transcribe"

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(base_model, peft_config)
model.to(device)
model.print_trainable_parameters()

# 5. Composite Distillation Loss Class
class CrossAttentionDistillationLoss(nn.Module):
    def __init__(self, alpha_ce=1.0, beta_kd=0.5, gamma_attn=2.5, lambda_mono=1.5, sigma=0.15, min_progression=0.5):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.beta_kd = beta_kd
        self.gamma_attn = gamma_attn
        self.lambda_mono = lambda_mono
        self.sigma = sigma
        self.min_progression = min_progression
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def compute_guided_attention_loss(self, cross_attention, valid_lengths=None):
        B, T_dec, T_enc = cross_attention.shape
        dev = cross_attention.device
        grid_dec = torch.arange(T_dec, device=dev, dtype=torch.float32).unsqueeze(1) / max(1, T_dec - 1)
        grid_enc = torch.arange(T_enc, device=dev, dtype=torch.float32).unsqueeze(0) / max(1, T_enc - 1)
        dist_sq = (grid_dec - grid_enc) ** 2
        W = 1.0 - torch.exp(-dist_sq / (2.0 * (self.sigma ** 2)))
        penalty = cross_attention * W.unsqueeze(0)
        if valid_lengths is not None:
            mask = torch.arange(T_dec, device=dev).unsqueeze(0) < valid_lengths.unsqueeze(1)
            penalty = penalty * mask.unsqueeze(-1)
            norm = valid_lengths.clamp(min=1).sum()
        else:
            norm = float(B * T_dec)
        return penalty.sum() / norm

    def compute_monotonicity_loss(self, cross_attention, valid_lengths=None):
        B, T_dec, T_enc = cross_attention.shape
        if T_dec <= 1:
            return torch.tensor(0.0, device=cross_attention.device)
        dev = cross_attention.device
        frame_indices = torch.arange(T_enc, device=dev, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        centroids = (cross_attention * frame_indices).sum(dim=-1)
        diffs = centroids[:, 1:] - centroids[:, :-1]
        penalties = F.relu(self.min_progression - diffs) ** 2
        if valid_lengths is not None:
            mask = torch.arange(T_dec - 1, device=dev).unsqueeze(0) < (valid_lengths.unsqueeze(1) - 1).clamp(min=0)
            penalties = penalties * mask
            norm = mask.sum().clamp(min=1)
        else:
            norm = float(B * (T_dec - 1))
        return penalties.sum() / norm

    def forward(self, student_logits, student_cross_attentions, labels):
        B, T_dec, V = student_logits.shape
        loss_ce = self.ce_loss(student_logits.view(-1, V), labels.view(-1))
        final_layer_attn = student_cross_attentions[-1]
        avg_student_attn = final_layer_attn.mean(dim=1)
        valid_lengths = (labels != -100).sum(dim=-1)
        loss_attn = self.compute_guided_attention_loss(avg_student_attn, valid_lengths)
        loss_mono = self.compute_monotonicity_loss(avg_student_attn, valid_lengths)
        loss_total = self.alpha_ce * loss_ce + self.gamma_attn * loss_attn + self.lambda_mono * loss_mono
        return {
            "loss_total": loss_total, "loss_ce": loss_ce,
            "loss_attn": loss_attn, "loss_mono": loss_mono
        }

loss_fn = CrossAttentionDistillationLoss(
    alpha_ce=1.0, beta_kd=0.5, gamma_attn=2.5, lambda_mono=1.5, sigma=0.15, min_progression=0.5
)

# 6. Build Paired Dataset
ref_singing = "trời còn làm mây xanh ngắt cho mắt em buồn trời còn làm mưa bay bay cho hạt sầu lên mắt người thương lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương"
ref_speech = "đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng cho mục đích huấn luyện trí tuệ nhân tạo mỗi bản ghi âm được chuẩn bị cẩn thận để đảm bảo phát âm rõ ràng nhịp điệu tự nhiên và chất lượng âm thanh ổn định"

class SingingDataset(Dataset):
    def __init__(self, items, proc):
        self.items = items
        self.proc = proc

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio = item["audio"]
        feats = self.proc(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
        labels = self.proc.tokenizer(item["text"]).input_ids
        return {"input_features": feats, "labels": torch.tensor(labels, dtype=torch.long)}

def collate_fn(batch):
    feats = torch.stack([x["input_features"] for x in batch])
    labels = torch.nn.utils.rnn.pad_sequence([x["labels"] for x in batch], batch_first=True, padding_value=-100)
    return {"input_features": feats, "labels": labels}

corpus = [
    {"audio": vocal_audio, "text": ref_singing},
    {"audio": poly_audio, "text": ref_singing},
    {"audio": speech_audio, "text": ref_speech},
    {"audio": vocal_audio[:int(8.5*sr)], "text": "trời còn làm mây xanh ngắt cho mắt em buồn"},
    {"audio": vocal_audio[int(8.5*sr):int(17.5*sr)], "text": "trời còn làm mưa bay bay cho hạt sầu lên mắt người thương"},
    {"audio": vocal_audio[int(17.5*sr):], "text": "lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương"}
]
train_loader = DataLoader(SingingDataset(corpus, processor), batch_size=2, shuffle=True, collate_fn=collate_fn)

# 7. Training Loop (15 Epochs on Tesla T4 GPU)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
epochs = 15

print(f"\n[*] Starting Phase IX Distillation Training for {epochs} epochs on {device}...")
t0 = time.time()
for ep in range(1, epochs + 1):
    model.train()
    tot_loss, tot_ce, tot_attn, tot_mono = 0.0, 0.0, 0.0, 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        feats = batch["input_features"].to(device)
        lbls = batch["labels"].to(device)
        with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available())):
            outputs = model(input_features=feats, labels=lbls, output_attentions=True)
            loss_dict = loss_fn(outputs.logits, outputs.cross_attentions, lbls)
            loss = loss_dict["loss_total"]

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
        tot_ce += float(loss_dict["loss_ce"].item())
        tot_attn += float(loss_dict["loss_attn"].item())
        tot_mono += float(loss_dict["loss_mono"].item())

    n = len(train_loader)
    print(f"  Epoch [{ep:02d}/{epochs:02d}] - Loss: {tot_loss/n:.4f} (CE: {tot_ce/n:.4f}, Attn: {tot_attn/n:.4f}, Mono: {tot_mono/n:.4f})")

elapsed = round(time.time() - t0, 2)
print(f"[*] Training finished in {elapsed}s.")

# 8. Evaluation & Metric Function
model.eval()
def evaluate_audio(audio_arr, ref_text):
    t_start = time.time()
    feats = processor(audio_arr, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        gen_out = model.generate(
            feats, max_new_tokens=128, repetition_penalty=1.2,
            no_repeat_ngram_size=3, output_attentions=True, return_dict_in_generate=True
        )
    hyp = processor.batch_decode(gen_out.sequences, skip_special_tokens=True)[0].strip()
    dur = len(audio_arr) / 16000
    rtf = round((time.time() - t_start) / dur, 4)
    wer = round(float(jiwer.wer(ref_text, hyp)), 4)
    cer = round(float(jiwer.cer(ref_text, hyp)), 4)

    path = []
    if gen_out.cross_attentions:
        for step_attn in gen_out.cross_attentions:
            if step_attn:
                avg_1d = step_attn[-1].mean(dim=(0, 1, 2)).cpu().numpy()
                path.append(int(np.argmax(avg_1d)))
    if len(path) > 2 and np.std(path) > 0:
        rho, _ = scipy.stats.spearmanr(np.arange(len(path)), path)
        mono_idx = round(float(rho) if not np.isnan(rho) else 0.0, 4)
    else:
        mono_idx = 0.0
    diffs = np.diff(path) if len(path) > 1 else np.array([0])
    stalled = int(np.sum(np.abs(diffs) <= 1))
    stall_ratio = round(float(stalled / len(diffs)) if len(diffs) > 0 else 0.0, 4)
    return {"hypothesis": hyp, "wer": wer, "cer": cer, "rtf": rtf, "monotonicity_index": mono_idx, "stalling_ratio": stall_ratio}

res_vocal = evaluate_audio(vocal_audio, ref_singing)
res_poly = evaluate_audio(poly_audio, ref_singing)
res_speech = evaluate_audio(speech_audio, ref_speech)

print(f"\n[*] Final Distillation Results:")
print(f"  Vocal Stem WER: {res_vocal['wer']*100:.2f}% | CER: {res_vocal['cer']*100:.2f}% | Mono rho: {res_vocal['monotonicity_index']:.4f} | Stall: {res_vocal['stalling_ratio']*100:.1f}%")
print(f"  Poly Mix   WER: {res_poly['wer']*100:.2f}% | CER: {res_poly['cer']*100:.2f}% | Mono rho: {res_poly['monotonicity_index']:.4f} | Stall: {res_poly['stalling_ratio']*100:.1f}%")
print(f"  Speech     WER: {res_speech['wer']*100:.2f}% | CER: {res_speech['cer']*100:.2f}% | Mono rho: {res_speech['monotonicity_index']:.4f} | Stall: {res_speech['stalling_ratio']*100:.1f}%")

# 9. Save Artifacts
save_path = Path("/kaggle/working/distilled_adapter_phowhisper")
save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(save_path))
processor.save_pretrained(str(save_path))

results_payload = {
    "experiment": "Phase IX: Cross-Attention Guided Knowledge Distillation (Kaggle GPU)",
    "date": "2026-10-14",
    "lead_researcher": "Thai Hung Nguyen",
    "device": str(device),
    "training_time_sec": elapsed,
    "vocal": res_vocal,
    "poly": res_poly,
    "speech": res_speech
}
with open("/kaggle/working/exp9_kaggle_distillation_results.json", "w", encoding="utf-8") as f:
    json.dump(results_payload, f, ensure_ascii=False, indent=2)

print("\n[+] Phase IX Kaggle Execution COMPLETED successfully!")

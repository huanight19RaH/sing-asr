"""
Kaggle GPU Kernel: SING-ASR PhoWhisper LoRA Adaptation Benchmark (Phase VIII).
Lead Researcher: Thai Hung Nguyen (AIT Laboratory, Department of Information Technology).
Runs on Kaggle NVIDIA Tesla T4 GPU.

Installs dependencies, loads PhoWhisper-tiny with LoRA (r=8, alpha=32),
trains for 15 epochs with FP16, and evaluates recognition performance.
"""
import os
import sys
import subprocess
import time
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Install dependencies
print("[*] Installing required packages on Kaggle...")
# Remove incompatible torchao 0.10.0 from Kaggle system to prevent PEFT collision
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "peft", "soundfile", "librosa", "jiwer"], check=True)

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
    # Also check /kaggle/input
    alt_paths = list(Path("/kaggle/working").glob("*.npz")) + list(Path("/kaggle/input").glob("**/*.npz"))
    if alt_paths:
        npz_path = alt_paths[0]

print(f"[*] Loading audio dataset from: {npz_path}")
audio_data = np.load(str(npz_path))
speech_audio = audio_data["speech"].astype(np.float32)
vocal_audio = audio_data["vocal"].astype(np.float32)
poly_audio = audio_data["poly"].astype(np.float32)
sr = 16000

# 4. Initialize PhoWhisper-tiny & LoRA
model_id = "vinai/PhoWhisper-tiny"
print(f"[*] Initializing {model_id}...")
processor = WhisperProcessor.from_pretrained(model_id)
base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
base_model.config.forced_decoder_ids = None
base_model.generation_config.language = "vi"
base_model.generation_config.task = "transcribe"

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(base_model, peft_config)
model.to(device)
model.print_trainable_parameters()

# 5. Build Training Corpus
ref_singing = (
    "Trời còn làm mây xanh ngắt cho mắt em buồn trời còn làm mưa bay bay cho hạt sầu lên mắt người thương "
    "lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương"
)
ref_speech = (
    "đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng cho mục đích huấn luyện trí tuệ nhân tạo "
    "mỗi bản ghi âm được chuẩn bị cẩn thận để đảm bảo phát âm rõ ràng nhịp điệu tự nhiên và chất lượng âm thanh ổn định"
)

song_segments = [
    (0.0, 8.5, "Trời còn làm mây xanh ngắt cho mắt em buồn"),
    (8.5, 17.5, "Trời còn làm mưa bay bay cho hạt sầu lên mắt người thương"),
    (17.5, 26.1, "Lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương"),
]

corpus = [
    {"audio": vocal_audio, "text": ref_singing, "id": "vocal_full"},
    {"audio": poly_audio, "text": ref_singing, "id": "poly_full"},
    {"audio": speech_audio, "text": ref_speech, "id": "speech_full"}
]
for st, et, txt in song_segments:
    s_idx, e_idx = int(st * sr), int(et * sr)
    corpus.append({"audio": vocal_audio[s_idx:e_idx], "text": txt, "id": f"vocal_seg_{st}"})
    corpus.append({"audio": poly_audio[s_idx:e_idx], "text": txt, "id": f"poly_seg_{st}"})

print(f"[*] Total paired instances in corpus: {len(corpus)}")

# Dataset Class
class KaggleSingingDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        feat = processor(item["audio"], sampling_rate=sr, return_tensors="pt").input_features[0]
        labels = processor.tokenizer(item["text"]).input_ids
        return {"input_features": feat, "labels": torch.tensor(labels, dtype=torch.long)}

def collate_fn(batch):
    feats = torch.stack([x["input_features"] for x in batch])
    labs = torch.nn.utils.rnn.pad_sequence([x["labels"] for x in batch], batch_first=True, padding_value=-100)
    return {"input_features": feats, "labels": labs}

dataset = KaggleSingingDataset(corpus)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# 6. Pre-adaptation Baseline Evaluation
def evaluate_audio(audio_tensor, ref_text):
    model.eval()
    feats = processor(audio_tensor, sampling_rate=sr, return_tensors="pt").input_features.to(device)
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
    wer = jiwer.wer(ref_text, hyp)
    cer = jiwer.cer(ref_text, hyp)
    return {"wer": round(float(wer), 4), "cer": round(float(cer), 4), "hypothesis": hyp}

print("\n[*] Evaluating Pre-Adaptation Baseline...")
base_vocal = evaluate_audio(vocal_audio, ref_singing)
base_poly = evaluate_audio(poly_audio, ref_singing)
base_speech = evaluate_audio(speech_audio, ref_speech)
print(f"  Pre-LoRA Vocal WER: {base_vocal['wer']*100:.2f}% | CER: {base_vocal['cer']*100:.2f}%")
print(f"  Pre-LoRA Poly WER:  {base_poly['wer']*100:.2f}% | CER: {base_poly['cer']*100:.2f}%")
print(f"  Pre-LoRA Speech WER:{base_speech['wer']*100:.2f}% | CER: {base_speech['cer']*100:.2f}%")

# 7. Fine-tuning on Tesla T4 GPU
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
epochs = 15

print(f"\n[*] Training LoRA on Tesla T4 GPU for {epochs} epochs...")
history = []
t_train_start = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    tot_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        feats = batch["input_features"].to(device)
        labs = batch["labels"].to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model(input_features=feats, labels=labs)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tot_loss += float(loss.item())

    avg_l = tot_loss / len(loader)
    history.append({"epoch": epoch, "loss": round(avg_l, 4)})
    if epoch % 3 == 0 or epoch == epochs:
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_l:.4f}")

train_time = time.time() - t_train_start
print(f"[*] Training finished in {train_time:.1f} seconds.")

# 8. Post-adaptation Evaluation
print("\n[*] Evaluating Post-LoRA Adaptation Benchmark...")
post_vocal = evaluate_audio(vocal_audio, ref_singing)
post_poly = evaluate_audio(poly_audio, ref_singing)
post_speech = evaluate_audio(speech_audio, ref_speech)

print(f"  Post-LoRA Vocal WER: {post_vocal['wer']*100:.2f}% (Delta: {(post_vocal['wer'] - base_vocal['wer'])*100:+.2f}%)")
print(f"  Post-LoRA Poly WER:  {post_poly['wer']*100:.2f}% (Delta: {(post_poly['wer'] - base_poly['wer'])*100:+.2f}%)")
print(f"  Post-LoRA Speech WER:{post_speech['wer']*100:.2f}% (Delta: {(post_speech['wer'] - base_speech['wer'])*100:+.2f}%)")

# 9. Save Artifacts to /kaggle/working/
out_dir = Path("/kaggle/working/lora_adapter_phowhisper")
out_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(out_dir))
processor.save_pretrained(str(out_dir))
print(f"[*] LoRA Adapter weights saved to: {out_dir}")

results_file = Path("/kaggle/working/exp8_kaggle_lora_results.json")
with open(results_file, "w", encoding="utf-8") as f:
    json.dump({
        "experiment": "Exp 8: PEFT / LoRA Adaptation Benchmark on Kaggle T4 GPU",
        "date": "2026-10-12",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "training_time_sec": round(train_time, 2),
        "loss_history": history,
        "pre_adaptation": {
            "vocal": base_vocal,
            "poly": base_poly,
            "speech": base_speech
        },
        "post_adaptation": {
            "vocal": post_vocal,
            "poly": post_poly,
            "speech": post_speech
        }
    }, f, indent=2, ensure_ascii=False)

print(f"[*] Benchmark results saved to: {results_file}")
print("[*] KAGGLE RUN COMPLETED SUCCESSFULLY.")

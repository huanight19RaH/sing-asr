"""
SING-ASR Cross-Attention Guided Knowledge Distillation Engine (Phase IX).
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Orchestrates:
1. Loading Student PhoWhisper with LoRA adapters (Phase VIII foundation).
2. Forward pass with eager Cross-Attention extraction.
3. Optimization against composite multi-task distillation loss:
       L_total = alpha * L_ce + beta * L_kd + gamma * L_attn + lambda * L_mono
4. Metric evaluation tracking both recognition accuracy (WER, CER) and
   attention dynamics (Monotonicity Index rho, Stalling Ratio).
"""
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, PeftModel

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from train.dataset import create_singing_paired_corpus, VietnameseSingingDataset, collate_fn_whisper
from train.distillation_loss import CrossAttentionDistillationLoss, DistillationHyperparameters
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


class DistillationTrainer:
    """
    Manages the Knowledge Distillation training and evaluation workflow.
    """
    def __init__(
        self,
        base_model_id: str = "vinai/PhoWhisper-tiny",
        initial_lora_dir: str | Path = None,
        hparams: DistillationHyperparameters = None,
        device: torch.device = None
    ):
        self.base_model_id = base_model_id
        self.device = device or get_safe_device()
        self.hparams = hparams or DistillationHyperparameters()
        self.loss_fn = CrossAttentionDistillationLoss(self.hparams)

        print(f"[sing-asr Distillation] Initializing processor for {base_model_id}...")
        self.processor = WhisperProcessor.from_pretrained(base_model_id)

        print(f"[sing-asr Distillation] Loading Student model with eager attention...")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_id,
            attn_implementation="eager"
        )
        base_model.config.forced_decoder_ids = None
        base_model.generation_config.language = "vi"
        base_model.generation_config.task = "transcribe"

        if initial_lora_dir and Path(initial_lora_dir).exists():
            print(f"[sing-asr Distillation] Loading pre-trained LoRA weights from {initial_lora_dir}...")
            self.model = PeftModel.from_pretrained(base_model, str(initial_lora_dir), is_trainable=True)
        else:
            print(f"[sing-asr Distillation] Initializing new LoRA adapter (All Attention Projections)...")
            peft_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                lora_dropout=0.05,
                bias="none"
            )
            self.model = get_peft_model(base_model, peft_config)

        self.model.to(self.device)
        self.model.print_trainable_parameters()

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> dict[str, float]:
        """
        Runs one epoch of Cross-Attention Guided Knowledge Distillation.
        """
        self.model.train()
        tot_loss = 0.0
        tot_ce = 0.0
        tot_attn = 0.0
        tot_mono = 0.0

        for batch in loader:
            optimizer.zero_grad()
            feats = batch["input_features"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with attention extraction
            outputs = self.model(
                input_features=feats,
                labels=labels,
                output_attentions=True
            )

            loss_dict = self.loss_fn(
                student_logits=outputs.logits,
                student_cross_attentions=outputs.cross_attentions,
                labels=labels
            )

            loss = loss_dict["loss_total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            tot_loss += float(loss.item())
            tot_ce += float(loss_dict["loss_ce"].item())
            tot_attn += float(loss_dict["loss_attn"].item())
            tot_mono += float(loss_dict["loss_mono"].item())

        n = len(loader)
        return {
            "loss_total": round(tot_loss / n, 4),
            "loss_ce": round(tot_ce / n, 4),
            "loss_attn": round(tot_attn / n, 4),
            "loss_mono": round(tot_mono / n, 4)
        }

    def train(
        self,
        corpus: list[dict],
        epochs: int = 5,
        lr: float = 5e-4,
        save_dir: str | Path = None
    ) -> list[dict]:
        dataset = VietnameseSingingDataset(self.processor, corpus, augment=True)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=lambda b: collate_fn_whisper(b, self.processor)
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        history = []
        t0 = time.time()

        print(f"\n[sing-asr Distillation] Starting Training for {epochs} epochs...", flush=True)
        for ep in range(1, epochs + 1):
            metrics = self.train_epoch(loader, optimizer)
            metrics["epoch"] = ep
            history.append(metrics)
            print(f"  Epoch [{ep:02d}/{epochs:02d}] - Total Loss: {metrics['loss_total']:.4f} "
                  f"(CE: {metrics['loss_ce']:.4f}, Attn: {metrics['loss_attn']:.4f}, Mono: {metrics['loss_mono']:.4f})", flush=True)

        elapsed = round(time.time() - t0, 2)
        print(f"[sing-asr Distillation] Training finished in {elapsed}s.", flush=True)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(save_path))
            self.processor.save_pretrained(str(save_path))
            print(f"[sing-asr Distillation] Adapter checkpoint saved to: {save_path}", flush=True)

        return history

    def evaluate_audio_and_attention(
        self,
        audio_path: str | Path,
        reference_text: str
    ) -> dict:
        """
        Evaluates transcription metrics and extracts cross-attention dynamics.
        """
        self.model.eval()
        audio, sr = load_audio(audio_path, target_sr=16000)
        t_start = time.time()
        feats = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(self.device)

        with torch.no_grad():
            gen_out = self.model.generate(
                feats,
                max_new_tokens=128,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                language="vi",
                task="transcribe",
                output_attentions=True,
                return_dict_in_generate=True
            )

        infer_time = time.time() - t_start
        rtf = round(infer_time / (len(audio) / sr), 4)

        sequences = gen_out.sequences[0]
        hyp = self.processor.batch_decode(gen_out.sequences, skip_special_tokens=True)[0].strip()

        metrics = compute_metrics(reference_text, hyp)
        alignment = align_words(reference_text, hyp)

        # Cross-Attention dynamics computation
        cross_attns = gen_out.cross_attentions
        path = []
        if cross_attns:
            for step_idx in range(len(cross_attns)):
                layer_attns = cross_attns[step_idx]
                if layer_attns:
                    # Final layer cross-attention averaged over heads -> (kv_len,)
                    final_layer = layer_attns[-1]
                    avg_1d = final_layer.mean(dim=(0, 1, 2)).cpu().numpy()
                    path.append(int(np.argmax(avg_1d)))

        # Monotonicity Index & Stalling calculation
        if len(path) > 2 and np.std(path) > 0:
            rho, _ = scipy.stats.spearmanr(np.arange(len(path)), path)
            monotonicity_index = round(float(rho) if not np.isnan(rho) else 0.0, 4)
        else:
            monotonicity_index = 0.0

        diffs = np.diff(path) if len(path) > 1 else np.array([0])
        stalled_count = int(np.sum(np.abs(diffs) <= 1))
        stalling_ratio = round(float(stalled_count / len(diffs)) if len(diffs) > 0 else 0.0, 4)

        return {
            "hypothesis": hyp,
            "wer": round(metrics.wer, 4),
            "cer": round(metrics.cer, 4),
            "substitutions": alignment.substitutions,
            "deletions": alignment.deletions,
            "insertions": alignment.insertions,
            "rtf": rtf,
            "monotonicity_index": monotonicity_index,
            "stalling_ratio": stalling_ratio,
            "attention_path_length": len(path)
        }

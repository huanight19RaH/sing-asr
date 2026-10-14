"""
SING-ASR Multi-Domain Pareto Early Stopping Engine (Phase IX.B).
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Implements multi-objective early stopping to balance bias and variance across:
- Polyphonic Mix WER (Domain 1: Music + Vocals)
- Isolated Vocal Stem WER (Domain 2: Acapella / Dry Singing)
- Conversational Speech Anchor WER (Domain 3: Spoken Vietnamese Fluency)

Prevents catastrophic overfitting or drift on any single acoustic domain by computing
a weighted Pareto Validation Score:
    S_val = w_poly * WER_poly + w_vocal * WER_vocal + w_speech * WER_speech
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import torch
import torch.nn as nn


@dataclass
class EarlyStoppingConfig:
    patience: int = 5                  # Number of epochs with no improvement before stopping
    min_delta: float = 0.005           # Minimum change in score to qualify as an improvement
    w_poly: float = 0.40               # Weight for Polyphonic Mix WER
    w_vocal: float = 0.30              # Weight for Vocal Stem WER
    w_speech: float = 0.30             # Weight for Spoken Anchor WER
    mode: str = "min"                  # Minimize composite error
    restore_best_weights: bool = True  # Automatically restore best state on stop


class MultiDomainEarlyStopping:
    """
    Monitors joint validation performance across singing and conversational speech domains.
    Halts training when Pareto score plateaus or degrades, restoring optimal checkpoint.
    """
    def __init__(self, config: EarlyStoppingConfig = None):
        self.cfg = config if config is not None else EarlyStoppingConfig()
        self.best_score = float("inf") if self.cfg.mode == "min" else float("-inf")
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_state_dict = None
        self.history = []

    def compute_pareto_score(
        self,
        wer_poly: float,
        wer_vocal: float,
        wer_speech: float
    ) -> float:
        """
        Computes composite Pareto score:
        S = w_poly * wer_poly + w_vocal * wer_vocal + w_speech * wer_speech
        """
        score = (
            self.cfg.w_poly * wer_poly
            + self.cfg.w_vocal * wer_vocal
            + self.cfg.w_speech * wer_speech
        )
        return round(float(score), 4)

    def step(
        self,
        epoch: int,
        wer_poly: float,
        wer_vocal: float,
        wer_speech: float,
        model: nn.Module = None,
        save_path: Path | str = None
    ) -> bool:
        """
        Evaluates current epoch metrics. Returns True if training should stop immediately.
        """
        score = self.compute_pareto_score(wer_poly, wer_vocal, wer_speech)
        entry = {
            "epoch": epoch,
            "wer_poly": wer_poly,
            "wer_vocal": wer_vocal,
            "wer_speech": wer_speech,
            "pareto_score": score
        }
        self.history.append(entry)

        is_better = (
            (self.best_score - score) > self.cfg.min_delta
            if self.cfg.mode == "min"
            else (score - self.best_score) > self.cfg.min_delta
        )

        if is_better:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if model is not None:
                # Save trainable adapter weights in memory
                trainable_dict = {
                    k: v.cpu().detach().clone()
                    for k, v in model.state_dict().items()
                    if any(t in k.lower() for t in ["lora", "adapter"])
                }
                self.best_state_dict = trainable_dict

                if save_path:
                    sp = Path(save_path)
                    sp.mkdir(parents=True, exist_ok=True)
                    if hasattr(model, "save_pretrained"):
                        model.save_pretrained(str(sp))
                    else:
                        torch.save(trainable_dict, sp / "best_adapter.pt")
        else:
            self.counter += 1
            if self.counter >= self.cfg.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best(self, model: nn.Module) -> None:
        """
        Restores the model state to the best recorded checkpoint.
        """
        if self.best_state_dict is not None and self.cfg.restore_best_weights:
            current_dict = model.state_dict()
            current_dict.update(self.best_state_dict)
            model.load_state_dict(current_dict)
            print(f"[EarlyStopping] Restored best adapter weights from Epoch {self.best_epoch} (Pareto Score: {self.best_score:.4f})")

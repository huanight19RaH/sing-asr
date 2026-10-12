"""
sing-asr PEFT / LoRA Adaptation Module for PhoWhisper-tiny (Phase VIII).
Directly addresses Phase VIII of the Research Plan (SLP301 - AIT Laboratory).
Lead Researcher: Thai Hung Nguyen

Scientific Mechanism:
Full fine-tuning of Foundation ASR models (e.g. PhoWhisper, Whisper) on singing voice
incurs high computational cost and risks catastrophic forgetting of conversational Vietnamese priors.
LoRA (Low-Rank Adaptation) freezes the pre-trained weights W_0 in R^{d x k} and injects trainable
rank-decomposition matrices:
    W = W_0 + (alpha / r) * (B * A)
where B in R^{d x r} is zero-initialized, A in R^{r x k} is Gaussian-initialized, with rank r=8 << min(d, k).

By targeting attention projection matrices (q_proj, v_proj), we adapt only:
    147,456 parameters out of 37,908,096 (< 0.39% trainable parameters).
This lightweight adaptation allows rapid fine-tuning on consumer hardware or Kaggle T4 GPU
to specialize cross-attention alignment for sustained singing vowels and musical dynamics.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, PeftModel

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dsp.audio_io import load_audio


@dataclass
class LoRAConfigParams:
    r: int = 8
    lora_alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lora_dropout: float = 0.05
    bias: str = "none"


class PhoWhisperLoRAWrapper:
    """
    Manages LoRA injection, training configuration, saving, and inference for PhoWhisper.
    """
    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-tiny",
        device: str = None,
        lora_params: LoRAConfigParams = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_id = model_id
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.base_model = WhisperForConditionalGeneration.from_pretrained(model_id)

        # Ensure decoder config flags are set for Vietnamese
        self.base_model.config.forced_decoder_ids = None
        self.base_model.generation_config.language = "vi"
        self.base_model.generation_config.task = "transcribe"

        if lora_params is None:
            lora_params = LoRAConfigParams()

        self.peft_config = LoraConfig(
            r=lora_params.r,
            lora_alpha=lora_params.lora_alpha,
            target_modules=list(lora_params.target_modules),
            lora_dropout=lora_params.lora_dropout,
            bias=lora_params.bias
        )

        self.model = get_peft_model(self.base_model, self.peft_config)
        self.model.to(self.device)

    def print_parameters(self):
        """Prints total and trainable parameter counts."""
        self.model.print_trainable_parameters()

    def get_trainable_param_stats(self) -> dict:
        """Returns quantitative dictionary of parameter footprints."""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        return {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_pct": round(100.0 * trainable_params / all_params, 4)
        }

    def prepare_input_features(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """Extracts 80-channel log-Mel filterbank tensor from 16kHz audio."""
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        )
        return inputs.input_features.to(self.device)

    def prepare_labels(self, text: str) -> torch.Tensor:
        """Encodes reference text into target token IDs with -100 padding."""
        labels = self.processor.tokenizer(text).input_ids
        # Replace padding with -100 to ignore in cross-entropy loss
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]
        return torch.tensor([labels], dtype=torch.long, device=self.device)

    def compute_loss(self, input_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes autoregressive cross-entropy loss with teacher forcing."""
        outputs = self.model(
            input_features=input_features,
            labels=labels
        )
        return outputs.loss

    def save_adapter(self, output_dir: str | Path):
        """Saves LoRA adapter weights and processor configuration."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out_path))
        self.processor.save_pretrained(str(out_path))
        print(f"[sing-asr LoRA] Adapter weights saved to: {out_path}")

    def load_adapter(self, adapter_dir: str | Path):
        """Loads fine-tuned LoRA adapter into base model."""
        self.model = PeftModel.from_pretrained(self.base_model, str(adapter_dir))
        self.model.to(self.device)
        self.model.eval()
        print(f"[sing-asr LoRA] Adapter weights loaded from: {adapter_dir}")

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        generate_kwargs: dict = None
    ) -> str:
        """Generates transcript using fine-tuned LoRA model."""
        self.model.eval()
        input_features = self.prepare_input_features(audio, sr=sr)

        kwargs = {
            "max_new_tokens": 128,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "language": "vi",
            "task": "transcribe"
        }
        if generate_kwargs:
            kwargs.update(generate_kwargs)

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, **kwargs)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()

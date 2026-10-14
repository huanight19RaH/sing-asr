"""
sing-asr ASR Module: Modern ASR Inference Engine using Whisper and PhoWhisper architectures.
Evaluates ASR recognition accuracy and failure modes on real Vietnamese audio for RQ1 & RQ3.
Supports Speech Processing (SLP301) CLO2.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import soundfile as sf
import numpy as np

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

@dataclass
class ASRTranscriptionResult:
    audio_path: str
    model_name: str
    transcription: str
    duration_sec: float
    inference_time_sec: float
    real_time_factor: float  # RTF = inference_time / duration

class WhisperEngine:
    def __init__(self, model_id: str = "openai/whisper-tiny", device: str = "cpu", lora_dir: str | Path = None):
        """
        Initialize Whisper or PhoWhisper inference wrapper with optional PEFT LoRA adapter.
        """
        self.model_id = model_id
        self.device = device
        self.lora_dir = Path(lora_dir) if lora_dir else None
        if not self.lora_dir:
            large_champ_dir = Path(src_dir).parent / "models" / "large_regularized_champion_phowhisper"
            champ_dir = Path(src_dir).parent / "models" / "regularized_champion_phowhisper"
            dist_dir = Path(src_dir).parent / "models" / "distilled_adapter_phowhisper"
            lora_dir_base = Path(src_dir).parent / "models" / "lora_adapter_phowhisper"
            if any(k in model_id.lower() for k in ["large", "phase ix.c", "multi-domain"]) and large_champ_dir.exists():
                self.lora_dir = large_champ_dir
            elif any(k in model_id.lower() for k in ["champion", "phase ix.b", "regular"]) and champ_dir.exists():
                self.lora_dir = champ_dir
            elif any(k in model_id.lower() for k in ["distill", "phase ix"]) and dist_dir.exists():
                self.lora_dir = dist_dir
            elif "lora" in model_id.lower() and lora_dir_base.exists():
                self.lora_dir = lora_dir_base
        self._pipe = None
        self._lora_model = None
        self._processor = None

    def _load_model(self):
        if self.lora_dir and self.lora_dir.exists():
            if self._lora_model is None:
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                from peft import PeftModel
                print(f"[sing-asr ASR] Loading PhoWhisper with LoRA adapter from: {self.lora_dir} on {self.device}...")
                base_id = "vinai/PhoWhisper-tiny"
                self._processor = WhisperProcessor.from_pretrained(str(self.lora_dir))
                base = WhisperForConditionalGeneration.from_pretrained(base_id)
                self._lora_model = PeftModel.from_pretrained(base, str(self.lora_dir))
                self._lora_model.to(self.device)
                self._lora_model.eval()
                print(f"[sing-asr ASR] LoRA adapter model loaded successfully.")
        elif self._pipe is None:
            from transformers import pipeline
            print(f"[sing-asr ASR] Loading model: {self.model_id} on {self.device}...")
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=self.device,
            )
            print(f"[sing-asr ASR] Model loaded successfully.")

    def transcribe(
        self,
        audio_path: str | Path,
        generate_kwargs: dict = None
    ) -> ASRTranscriptionResult:
        """
        Transcribe a real audio file in Vietnamese with optional decoding constraints.
        """
        self._load_model()
        from dsp.audio_io import load_audio

        # Load audio into 16kHz mono float32 numpy array via soundfile
        audio_data, sr = load_audio(audio_path, target_sr=16000)
        duration_sec = float(len(audio_data) / sr)

        # Build generation arguments
        gen_kwargs = {}
        if "phowhisper" not in self.model_id.lower():
            # Multilingual Whisper requires language & task specification
            gen_kwargs["language"] = "vi"
            gen_kwargs["task"] = "transcribe"
        
        if generate_kwargs:
            gen_kwargs.update(generate_kwargs)

        if "initial_prompt" in gen_kwargs:
            prompt_text = gen_kwargs.pop("initial_prompt")
            try:
                import torch
                p_ids = self._pipe.tokenizer.get_prompt_ids(prompt_text)
                if not isinstance(p_ids, torch.Tensor):
                    p_ids = torch.tensor(p_ids, dtype=torch.long)
                gen_kwargs["prompt_ids"] = p_ids
            except Exception as e:
                print(f"[WhisperEngine] Prompt conversion warning: {e}")

        t0 = time.time()
        if self._lora_model is not None:
            import torch
            inputs = self._processor(audio_data, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            gen_params = {
                "max_new_tokens": 128,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "language": "vi",
                "task": "transcribe"
            }
            if gen_kwargs:
                gen_params.update(gen_kwargs)
            with torch.no_grad():
                pred_ids = self._lora_model.generate(input_features, **gen_params)
            raw_text = self._processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        else:
            # Feed dictionary and kwargs to pipeline
            result = self._pipe({"raw": audio_data, "sampling_rate": sr}, generate_kwargs=gen_kwargs)
            raw_text = result.get("text", "").strip()
        inf_time = time.time() - t0
        rtf = inf_time / duration_sec if duration_sec > 0 else 0.0

        return ASRTranscriptionResult(
            audio_path=str(audio_path),
            model_name=self.model_id,
            transcription=raw_text,
            duration_sec=round(duration_sec, 2),
            inference_time_sec=round(inf_time, 2),
            real_time_factor=round(rtf, 3)
        )

if __name__ == '__main__':
    print("[sing-asr ASR] Testing WhisperEngine on REAL Audio Data...")
    real_speech = "data/processed/vietnamese_speech_real_01_16k.wav"
    if Path(real_speech).exists():
        engine = WhisperEngine(model_id="openai/whisper-tiny", device="cpu")
        res = engine.transcribe(real_speech)
        print(f"\n--- Real ASR Output ---")
        print(f"  Model: {res.model_name}")
        print(f"  Audio Duration: {res.duration_sec}s | Inference Time: {res.inference_time_sec}s (RTF: {res.real_time_factor})")
        print(f"  Transcription:\n  \"{res.transcription}\"")
        print("\n[sing-asr ASR] Real Whisper inference PASSED!")

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
    def __init__(self, model_id: str = "openai/whisper-tiny", device: str = "cpu"):
        """
        Initialize Whisper inference wrapper.
        """
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _load_model(self):
        if self._pipe is None:
            from transformers import pipeline
            print(f"[sing-asr ASR] Loading model: {self.model_id} on {self.device}...")
            # Default pipeline creation
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
        # Feed dictionary and kwargs to pipeline
        result = self._pipe({"raw": audio_data, "sampling_rate": sr}, generate_kwargs=gen_kwargs)
        inf_time = time.time() - t0

        raw_text = result.get("text", "").strip()
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

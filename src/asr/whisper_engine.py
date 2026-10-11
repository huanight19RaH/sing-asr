"""
sing-asr ASR Module: Modern ASR Inference Engine using Whisper and PhoWhisper architectures.
Evaluates ASR recognition accuracy and failure modes on real Vietnamese audio for RQ1 & RQ3.
Supports Speech Processing (SLP301) CLO2.
"""
from dataclasses import dataclass
from pathlib import Path
import time
import soundfile as sf
import numpy as np

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
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=self.device,
                generate_kwargs={"language": "vi", "task": "transcribe"}
            )
            print(f"[sing-asr ASR] Model loaded successfully.")

    def transcribe(self, audio_path: str | Path) -> ASRTranscriptionResult:
        """
        Transcribe a real audio file in Vietnamese.
        """
        self._load_model()
        from dsp.audio_io import load_audio
        
        # Load audio into 16kHz mono float32 numpy array via soundfile (no ffmpeg needed!)
        audio_data, sr = load_audio(audio_path, target_sr=16000)
        duration_sec = float(len(audio_data) / sr)
        
        t0 = time.time()
        # Feed dictionary to pipeline
        result = self._pipe({"raw": audio_data, "sampling_rate": sr})
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

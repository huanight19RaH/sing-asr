"""
Vocal-Accompaniment Separation Module for RQ3 (BGM Ablation).
Uses torchaudio's built-in Hybrid Demucs (HDEMUCS_HIGH_MUSDB) model to separate
polyphonic music into isolated vocal stems and instrumental accompaniment (BGM).
"""
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf

from dsp.audio_io import load_audio, save_audio

class VocalSeparator:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._bundle = None
        self._model = None
        
    def _load_model(self):
        if self._model is None:
            print(f"[sing-asr Separator] Loading HDEMUCS_HIGH_MUSDB on {self.device}...")
            self._bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB
            self._model = self._bundle.get_model().to(self.device)
            self._model.eval()
            print("[sing-asr Separator] Demucs model loaded successfully.")
            
    def separate(
        self,
        audio_path: str | Path,
        output_dir: str | Path = "data/processed/separated"
    ) -> dict[str, Path]:
        """
        Separates vocal and accompaniment from a music file.
        
        Returns:
        --------
        dict mapping stem name ('vocals', 'accompaniment') -> Path to saved wav file.
        """
        self._load_model()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem_name = Path(audio_path).stem
        
        # Load audio using soundfile at native or target sr
        # Demucs expects sample rate = bundle.sample_rate (44100 Hz), stereo (2 channels)
        bundle_sr = self._bundle.sample_rate
        audio_data, in_sr = load_audio(audio_path, target_sr=bundle_sr)
        
        # Shape: (channels, num_samples). Make stereo
        tensor = torch.from_numpy(audio_data).unsqueeze(0).repeat(2, 1).to(self.device)
        tensor = tensor.unsqueeze(0) # batch dimension (1, 2, num_samples)
        
        print(f"[sing-asr Separator] Running source separation on {Path(audio_path).name} ({len(audio_data)/bundle_sr:.1f}s)...")
        with torch.no_grad():
            # sources shape: (batch, num_sources, channels, time)
            # HDEMUCS_HIGH_MUSDB sources: drums (0), bass (1), other (2), vocals (3)
            sources = self._model(tensor)
            
        sources = sources.squeeze(0).cpu().numpy()
        # Source mapping:
        # sources[3] is vocals (shape: 2, time)
        # sources[0] + sources[1] + sources[2] is accompaniment (BGM)
        vocals_mono = np.mean(sources[3], axis=0) # convert stereo to mono
        bgm_mono = np.mean(sources[0] + sources[1] + sources[2], axis=0)
        
        # Resample back to standard 16kHz for ASR
        from scipy.signal import resample_poly
        # Resample from 44100 to 16000: gcd(160, 441)
        up, down = 160, 441
        vocals_16k = resample_poly(vocals_mono, up, down).astype(np.float32)
        bgm_16k = resample_poly(bgm_mono, up, down).astype(np.float32)
        
        # Normalize
        for arr in [vocals_16k, bgm_16k]:
            m = np.max(np.abs(arr))
            if m > 1.0:
                arr /= m
                
        vocals_path = out_dir / f"{stem_name}_vocals_16k.wav"
        bgm_path = out_dir / f"{stem_name}_bgm_16k.wav"
        
        save_audio(vocals_16k, 16000, vocals_path)
        save_audio(bgm_16k, 16000, bgm_path)
        
        print(f"[sing-asr Separator] Extracted vocals -> {vocals_path}")
        print(f"[sing-asr Separator] Extracted BGM    -> {bgm_path}")
        
        return {
            "vocals": vocals_path,
            "accompaniment": bgm_path
        }

if __name__ == '__main__':
    print("[sing-asr Separator] Demucs Vocal Separation Module Initialized.")

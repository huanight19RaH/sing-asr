"""
sing-asr DSP module: Audio I/O and Resampling utility for Real Audio Data.
Ensures all audio loaded or processed is standardized to 16 kHz mono PCM.
"""
from pathlib import Path
import numpy as np
import scipy.signal
import soundfile as sf
import math

def load_audio(
    file_path: str | Path,
    target_sr: int = 16000,
    mono: bool = True
) -> tuple[np.ndarray, int]:
    """
    Load real audio file (WAV, MP3, FLAC, OGG) and convert to mono 16 kHz.
    
    Args:
        file_path: Path to audio file.
        target_sr: Target sampling rate (default 16000 Hz).
        mono: Whether to convert stereo to mono by averaging channels.
    Returns:
        audio: 1D float32 array in range [-1.0, 1.0].
        target_sr: Sampling rate.
    """
    data, sr = sf.read(str(file_path), dtype='float32')
    
    # 1. Convert to mono if multi-channel
    if data.ndim > 1:
        if mono:
            data = np.mean(data, axis=1)
        else:
            data = data[:, 0]  # Take first channel
            
    # 2. Resample if necessary
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
        sr = target_sr
        
    # 3. Peak normalization to [-0.95, 0.95] if non-silent
    max_val = np.max(np.abs(data))
    if max_val > 1e-4:
        data = (data / max_val) * 0.95
        
    return data, sr

def save_audio(
    file_path: str | Path | np.ndarray,
    audio: np.ndarray | str | Path | int = None,
    sr: int | str | Path = 16000
) -> None:
    """
    Save 1D audio array to 16-bit PCM WAV.
    Flexible signature: accepts either (file_path, audio, sr) or (audio, sr, file_path).
    """
    if isinstance(file_path, np.ndarray):
        # Called as save_audio(audio, sr, file_path) or save_audio(audio, file_path, sr)
        arr = file_path
        if isinstance(audio, int):
            actual_sr = audio
            actual_path = sr
        else:
            actual_path = audio
            actual_sr = sr if isinstance(sr, int) else 16000
    else:
        # Called as save_audio(file_path, audio, sr)
        actual_path = file_path
        arr = audio
        actual_sr = sr if isinstance(sr, int) else 16000
        
    Path(actual_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(actual_path), arr, actual_sr, subtype='PCM_16')

if __name__ == '__main__':
    import sys
    print("[sing-asr Audio I/O] Testing real audio loading and resampling...")
    speech_path = "data/external/vietnamese_speech_real_01.mp3"
    song_path = "data/external/vietnamese_song_real_01.mp3"
    
    if Path(speech_path).exists():
        speech_sig, sr = load_audio(speech_path, target_sr=16000)
        out_wav = "data/processed/vietnamese_speech_real_01_16k.wav"
        save_audio(out_wav, speech_sig, sr=sr)
        print(f"  Real Speech loaded: duration={len(speech_sig)/sr:.2f}s, sr={sr}, saved={out_wav}")
        
    if Path(song_path).exists():
        song_sig, sr = load_audio(song_path, target_sr=16000)
        out_wav = "data/processed/vietnamese_song_real_01_16k.wav"
        save_audio(out_wav, song_sig, sr=sr)
        print(f"  Real Song loaded: duration={len(song_sig)/sr:.2f}s, sr={sr}, saved={out_wav}")
        
    print("[sing-asr Audio I/O] Real audio processing PASSED!")

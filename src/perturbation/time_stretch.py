"""
Time Stretching and Duration Normalization Module for Acoustic Perturbation Experiments (RQ4).
Implements Phase Vocoder-based time stretching to test whether compressing prolonged singing vowels
(re-scaling speech rate / tempo) mitigates ASR word deletion and hallucination on Vietnamese audio.
"""
from pathlib import Path
import numpy as np
import scipy.signal
import soundfile as sf

from dsp.audio_io import load_audio, save_audio

def phase_vocoder_time_stretch(
    audio: np.ndarray,
    rate: float = 1.0,
    n_fft: int = 1024,
    hop_length: int = 256
) -> np.ndarray:
    """
    Time-stretches an audio signal by a factor `rate` using a classic phase vocoder.
    - rate > 1.0: speeds up audio (compresses duration, e.g. 1.25x or 1.5x speed).
    - rate < 1.0: slows down audio (expands duration).
    Pitch is preserved.
    
    Parameters:
    -----------
    audio : np.ndarray
        Mono audio signal (float32).
    rate : float
        Stretch factor (playback speed multiplier).
    n_fft : int
        FFT window size (default: 1024).
    hop_length : int
        Analysis hop size (default: 256).
        
    Returns:
    --------
    np.ndarray
        Time-stretched audio signal.
    """
    if abs(rate - 1.0) < 1e-4:
        return audio.copy()
        
    # 1. Compute STFT
    window = scipy.signal.windows.hann(n_fft, sym=False)
    _, _, stft = scipy.signal.stft(
        audio,
        fs=16000,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary='zeros'
    )
    
    # stft shape: (freq_bins, num_frames)
    num_bins, num_frames = stft.shape
    
    # 2. Time progression in output
    # Analysis hop is hop_length, synthesis hop is hop_length * rate
    time_steps = np.arange(0, num_frames - 1, rate, dtype=float)
    stretched_stft = np.zeros((num_bins, len(time_steps)), dtype=complex)
    
    # Expected phase advance per hop
    omega = 2.0 * np.pi * hop_length * np.arange(num_bins) / n_fft
    phase_accum = np.angle(stft[:, 0])
    stretched_stft[:, 0] = stft[:, 0]
    
    for t_idx, t in enumerate(time_steps[1:], start=1):
        # Linear interpolation between adjacent frames
        left_idx = int(np.floor(t))
        right_idx = min(left_idx + 1, num_frames - 1)
        alpha = t - left_idx
        
        mag = (1.0 - alpha) * np.abs(stft[:, left_idx]) + alpha * np.abs(stft[:, right_idx])
        
        # Phase difference
        dphase = np.angle(stft[:, right_idx]) - np.angle(stft[:, left_idx]) - omega
        # Unwrap phase to [-pi, pi]
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        
        # True frequency deviation
        true_freq = omega + dphase
        
        # Accumulate phase
        phase_accum += true_freq * rate
        stretched_stft[:, t_idx] = mag * np.exp(1j * phase_accum)
        
    # 3. Inverse STFT
    _, stretched_audio = scipy.signal.istft(
        stretched_stft,
        fs=16000,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary='zeros'
    )
    
    # Peak normalization
    max_val = np.max(np.abs(stretched_audio))
    if max_val > 1.0:
        stretched_audio = stretched_audio / max_val
        
    return stretched_audio.astype(np.float32)

def generate_duration_variants(
    input_wav_path: str | Path,
    output_dir: str | Path = "data/perturbations/duration",
    speed_rates: list[float] = [0.8, 1.25, 1.5]
) -> dict[float, Path]:
    """
    Generates tempo/duration variants for an audio file.
    
    Returns:
    --------
    dict mapping speed rate -> Path to saved perturbed audio file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    audio, sr = load_audio(input_wav_path, target_sr=16000)
    stem = Path(input_wav_path).stem
    
    results = {}
    for rate in speed_rates:
        stretched = phase_vocoder_time_stretch(audio, rate=rate)
        fname = f"{stem}_speed_{rate:.2f}x.wav"
        save_file = out_path / fname
        save_audio(stretched, sr, save_file)
        results[rate] = save_file
        print(f"[Duration Perturbation] Generated rate {rate:.2f}x (dur: {len(stretched)/sr:.2f}s) -> {save_file}")
        
    return results

if __name__ == '__main__':
    print("[sing-asr Perturbation] Testing Time Stretching...")
    test_file = "data/processed/vietnamese_speech_real_01_16k.wav"
    if Path(test_file).exists():
        variants = generate_duration_variants(test_file, speed_rates=[1.25, 1.5])
        print(f"Generated variants: {variants}")

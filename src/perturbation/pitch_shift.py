"""
Pitch Modification and Flattening Module for Acoustic Perturbation Experiments (RQ4).
Enables testing whether neutralizing singing pitch excursions, vibrato, or high-pitch registers
improves ASR transcription accuracy on Vietnamese vocal audio.
"""
from pathlib import Path
import numpy as np
import torch
import torchaudio.functional as F_audio
import soundfile as sf

from dsp.audio_io import load_audio, save_audio

def pitch_shift_audio(
    audio: np.ndarray,
    sr: int = 16000,
    n_steps: float = 0.0,
    bins_per_octave: int = 12
) -> np.ndarray:
    """
    Shifts the pitch of a 1D audio waveform by n_steps semitones without modifying its duration.
    
    Parameters:
    -----------
    audio : np.ndarray
        Mono audio signal (float32, normalized between -1.0 and 1.0).
    sr : int
        Sampling frequency in Hz (default: 16000).
    n_steps : float
        Pitch shift amount in semitones (+: shift up, -: shift down).
    bins_per_octave : int
        Number of steps per octave (12 for semitones).
        
    Returns:
    --------
    np.ndarray
        Pitch-shifted audio waveform of the same length and sampling rate.
    """
    if abs(n_steps) < 1e-4:
        return audio.copy()
        
    # Convert numpy to torch tensor: shape (1, num_samples)
    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    
    # torchaudio pitch_shift
    # Uses phase vocoder internally
    shifted_tensor = F_audio.pitch_shift(
        waveform,
        sample_rate=sr,
        n_steps=n_steps,
        bins_per_octave=bins_per_octave,
        n_fft=512,
        win_length=512,
        hop_length=128
    )
    
    shifted = shifted_tensor.squeeze(0).cpu().numpy()
    
    # Match exact length if slightly different due to STFT windowing
    if len(shifted) > len(audio):
        shifted = shifted[:len(audio)]
    elif len(shifted) < len(audio):
        shifted = np.pad(shifted, (0, len(audio) - len(shifted)), mode='constant')
        
    # Peak normalization to prevent clipping
    max_val = np.max(np.abs(shifted))
    if max_val > 1.0:
        shifted = shifted / max_val
        
    return shifted

def generate_pitch_variants(
    input_wav_path: str | Path,
    output_dir: str | Path = "data/perturbations/pitch",
    semitone_shifts: list[float] = [-4.0, -2.0, 2.0, 4.0]
) -> dict[float, Path]:
    """
    Generates a set of pitch-shifted variants for an audio file.
    
    Returns:
    --------
    dict mapping semitone shift -> Path to saved perturbed audio file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    audio, sr = load_audio(input_wav_path, target_sr=16000)
    stem = Path(input_wav_path).stem
    
    results = {}
    for shift in semitone_shifts:
        shifted = pitch_shift_audio(audio, sr=sr, n_steps=shift)
        sign = "plus" if shift > 0 else "minus"
        abs_shift = abs(int(shift)) if shift.is_integer() else abs(shift)
        fname = f"{stem}_pitch_{sign}{abs_shift}st.wav"
        save_file = out_path / fname
        save_audio(shifted, sr, save_file)
        results[shift] = save_file
        print(f"[Pitch Perturbation] Generated {shift:+.1f} semitones -> {save_file}")
        
    return results

if __name__ == '__main__':
    print("[sing-asr Perturbation] Testing Pitch Modification...")
    test_file = "data/processed/vietnamese_speech_real_01_16k.wav"
    if Path(test_file).exists():
        variants = generate_pitch_variants(test_file, semitone_shifts=[-3.0, 3.0])
        print(f"Generated variants: {variants}")

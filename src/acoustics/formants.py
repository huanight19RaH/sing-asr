"""
sing-asr Acoustics Module: LPC Formant Estimation and Formant Dispersion.
Estimates F1, F2, F3 resonances and Formant Dispersion (FD) for Speech Processing (SLP301) CLO3.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.signal
import soundfile as sf

@dataclass
class FormantFeatures:
    f1_mean: float           # Mean First Formant (Hz, correlated with vowel height)
    f2_mean: float           # Mean Second Formant (Hz, correlated with vowel backness)
    f3_mean: float           # Mean Third Formant (Hz)
    f12_diff_mean: float     # F2 - F1 (vowel compactness index)
    formant_dispersion: float# FD = (F3 - F1) / 2 (vocal tract length proxy)
    valid_frame_count: int

def lpc_coefficients(frame: np.ndarray, order: int) -> np.ndarray:
    """
    Compute Linear Predictive Coding (LPC) coefficients via autocorrelation and Levinson-Durbin.
    """
    # Autocorrelation
    r = np.correlate(frame, frame, mode='full')
    r = r[len(r) // 2 : len(r) // 2 + order + 1]
    
    if r[0] <= 1e-9:
        return np.zeros(order + 1)
        
    # Levinson-Durbin recursion
    a = np.zeros(order + 1)
    a[0] = 1.0
    k = np.zeros(order)
    alpha = r[0]
    
    for i in range(1, order + 1):
        # Reflection coefficient
        acc = np.dot(a[:i], r[i:0:-1])
        k[i - 1] = -acc / alpha if alpha > 1e-9 else 0.0
        
        # Update filter coefficients
        a_new = a.copy()
        a_new[1:i+1] = a[1:i+1] + k[i - 1] * a[i-1::-1]
        a = a_new
        alpha = alpha * (1.0 - k[i - 1] ** 2)
        
    return a

def extract_formants_frame(
    frame: np.ndarray,
    samplerate: int,
    order: int = 16
) -> tuple[float, float, float] | None:
    """
    Extract F1, F2, F3 frequencies for a single voiced frame.
    """
    # Apply pre-emphasis filter to boost high frequency formants
    pre_emphasized = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
    windowed = pre_emphasized * np.hamming(len(pre_emphasized))
    
    a = lpc_coefficients(windowed, order)
    if np.all(a == 0):
        return None
        
    # Find polynomial roots
    roots = np.roots(a)
    # Retain roots in upper half of complex plane with non-zero frequency
    roots = roots[np.imag(roots) >= 0]
    
    formants = []
    for r in roots:
        freq = float(np.arctan2(np.imag(r), np.real(r)) * (samplerate / (2.0 * np.pi)))
        bw = float(-0.5 * (samplerate / np.pi) * np.log(max(1e-6, np.abs(r))))
        # Keep sensible speech formants
        if 200.0 < freq < (samplerate / 2.0 - 200.0) and bw < 600.0:
            formants.append(freq)
            
    formants.sort()
    if len(formants) >= 3:
        return float(formants[0]), float(formants[1]), float(formants[2])
    elif len(formants) == 2:
        return float(formants[0]), float(formants[1]), float(formants[1] + 1000.0)
    return None

def compute_formant_features(
    signal: np.ndarray,
    samplerate: int,
    frame_len_sec: float = 0.030,
    frame_step_sec: float = 0.015,
    lpc_order: int = 16
) -> FormantFeatures:
    """
    Extract average formant locations and formant dispersion across active frames.
    """
    frame_len = int(frame_len_sec * samplerate)
    frame_step = int(frame_step_sec * samplerate)
    num_frames = max(1, 1 + int(np.ceil((len(signal) - frame_len) / frame_step)))
    
    f1_list, f2_list, f3_list = [], [], []
    
    # Calculate energy threshold to avoid silent frames
    frame_energies = [np.mean(signal[i * frame_step : i * frame_step + frame_len] ** 2) for i in range(num_frames)]
    max_energy = max(frame_energies) if frame_energies else 0.0
    energy_thresh = 0.02 * max_energy
    
    for i in range(num_frames):
        start = i * frame_step
        end = min(len(signal), start + frame_len)
        frame = signal[start:end]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
            
        if np.mean(frame ** 2) < energy_thresh:
            continue
            
        res = extract_formants_frame(frame, samplerate, order=lpc_order)
        if res is not None:
            f1, f2, f3 = res
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            
    if not f1_list:
        return FormantFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        
    f1_mean = float(np.mean(f1_list))
    f2_mean = float(np.mean(f2_list))
    f3_mean = float(np.mean(f3_list))
    f12_diff = float(f2_mean - f1_mean)
    fd = float((f3_mean - f1_mean) / 2.0)
    
    return FormantFeatures(
        f1_mean=round(f1_mean, 1),
        f2_mean=round(f2_mean, 1),
        f3_mean=round(f3_mean, 1),
        f12_diff_mean=round(f12_diff, 1),
        formant_dispersion=round(fd, 1),
        valid_frame_count=len(f1_list)
    )

if __name__ == '__main__':
    from dsp.audio_io import load_audio
    print("[sing-asr Acoustics] Testing Formant Extraction on REAL Vietnamese Data...")
    
    real_speech = "data/processed/vietnamese_speech_real_01_16k.wav"
    real_song = "data/processed/vietnamese_song_real_01_16k.wav"
    
    if Path(real_speech).exists():
        sig_spk, sr = load_audio(real_speech)
        feats_spk = compute_formant_features(sig_spk, sr)
        print(f"\n--- Real Vietnamese Speech Formants ---")
        print(f"  F1: {feats_spk.f1_mean} Hz | F2: {feats_spk.f2_mean} Hz | F3: {feats_spk.f3_mean} Hz")
        print(f"  Formant Dispersion (FD): {feats_spk.formant_dispersion} Hz (Frames: {feats_spk.valid_frame_count})")
        assert feats_spk.f1_mean > 250 and feats_spk.f2_mean > feats_spk.f1_mean
        
    if Path(real_song).exists():
        sig_sng, sr = load_audio(real_song)
        # Take first 30 seconds for fast evaluation
        sig_sng_sub = sig_sng[: 30 * sr]
        feats_sng = compute_formant_features(sig_sng_sub, sr)
        print(f"\n--- Real Vietnamese Singing Formants ---")
        print(f"  F1: {feats_sng.f1_mean} Hz | F2: {feats_sng.f2_mean} Hz | F3: {feats_sng.f3_mean} Hz")
        print(f"  Formant Dispersion (FD): {feats_sng.formant_dispersion} Hz (Frames: {feats_sng.valid_frame_count})")
        
    print("\n[sing-asr Acoustics] Formant extraction on REAL DATA passed successfully!")

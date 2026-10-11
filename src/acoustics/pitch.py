"""
sing-asr Acoustics Module: Fundamental Frequency (F0) Tracking, Pitch Dynamics, and Vibrato Extraction.
Core implementation for RQ2 and Speech Processing (SLP301) CLO3.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class PitchFeatures:
    f0_mean: float           # Mean F0 across voiced frames (Hz)
    f0_std: float            # Standard deviation of F0 (Hz)
    f0_cv: float             # Coefficient of Variation: sigma / mu (Dimensionless)
    f0_min: float            # Min voiced F0 (Hz)
    f0_max: float            # Max voiced F0 (Hz)
    f0_range_semitones: float# Range in musical semitones: 12 * log2(max/min)
    voiced_fraction: float   # Ratio of voiced frames to total frames
    vibrato_rate_hz: float   # Estimated vibrato modulation rate (Hz, typ. 4-8 Hz in singing)

def autocorr_pitch(
    frame: np.ndarray,
    samplerate: int,
    fmin: float = 65.0,
    fmax: float = 800.0,
    threshold: float = 0.3
) -> float:
    """
    Estimate fundamental frequency F0 of a single frame using normalized autocorrelation.
    Returns 0.0 if unvoiced.
    """
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    
    # Compute autocorrelation via pure numpy
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr) // 2:]
    
    # Normalize autocorrelation
    if corr[0] <= 0:
        return 0.0
    corr_norm = corr / corr[0]
    
    min_lag = int(samplerate / fmax)
    max_lag = int(samplerate / fmin)
    if max_lag >= len(corr_norm):
        max_lag = len(corr_norm) - 1
        
    search_region = corr_norm[min_lag:max_lag]
    if len(search_region) == 0:
        return 0.0
        
    peak_idx = np.argmax(search_region) + min_lag
    peak_val = corr_norm[peak_idx]
    
    if peak_val > threshold:
        return float(samplerate / peak_idx)
    return 0.0

def extract_f0_contour(
    signal: np.ndarray,
    samplerate: int,
    frame_len_sec: float = 0.040,
    frame_step_sec: float = 0.010,
    fmin: float = 65.0,
    fmax: float = 800.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract continuous F0 contour and timestamps across frames.
    """
    frame_len = int(frame_len_sec * samplerate)
    frame_step = int(frame_step_sec * samplerate)
    num_frames = max(1, 1 + int(np.ceil((len(signal) - frame_len) / frame_step)))
    
    f0_contour = np.zeros(num_frames)
    times = np.arange(num_frames) * frame_step_sec
    
    for i in range(num_frames):
        start = i * frame_step
        end = min(len(signal), start + frame_len)
        frame = signal[start:end]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        # Window with Hann
        frame = frame * np.hanning(len(frame))
        f0_contour[i] = autocorr_pitch(frame, samplerate, fmin=fmin, fmax=fmax)
        
    return times, f0_contour

def compute_pitch_features(
    f0_contour: np.ndarray,
    frame_step_sec: float = 0.010
) -> PitchFeatures:
    """
    Compute comprehensive pitch dynamics from F0 contour for correlation with ASR WER.
    """
    voiced = f0_contour[f0_contour > 0]
    total_frames = len(f0_contour)
    voiced_fraction = float(len(voiced) / total_frames) if total_frames > 0 else 0.0
    
    if len(voiced) < 5:
        return PitchFeatures(
            f0_mean=0.0, f0_std=0.0, f0_cv=0.0,
            f0_min=0.0, f0_max=0.0, f0_range_semitones=0.0,
            voiced_fraction=voiced_fraction, vibrato_rate_hz=0.0
        )
        
    f0_mean = float(np.mean(voiced))
    f0_std = float(np.std(voiced))
    f0_cv = float(f0_std / f0_mean) if f0_mean > 0 else 0.0
    f0_min = float(np.min(voiced))
    f0_max = float(np.max(voiced))
    
    # Semitone range
    if f0_min > 0 and f0_max > f0_min:
        f0_range_semitones = float(12.0 * np.log2(f0_max / f0_min))
    else:
        f0_range_semitones = 0.0
        
    # Estimate vibrato modulation rate via FFT of voiced F0 contour detrended
    voiced_detrend = voiced - np.mean(voiced)
    if len(voiced_detrend) > 32:
        fft_vibrato = np.abs(np.fft.rfft(voiced_detrend))
        freqs = np.fft.rfftfreq(len(voiced_detrend), d=frame_step_sec)
        # Search in typical vibrato frequency range (4 Hz to 8 Hz)
        mask = (freqs >= 3.5) & (freqs <= 9.0)
        if np.any(mask) and np.max(fft_vibrato[mask]) > 0:
            vibrato_rate = float(freqs[mask][np.argmax(fft_vibrato[mask])])
        else:
            vibrato_rate = 0.0
    else:
        vibrato_rate = 0.0
        
    return PitchFeatures(
        f0_mean=round(f0_mean, 2),
        f0_std=round(f0_std, 2),
        f0_cv=round(f0_cv, 4),
        f0_min=round(f0_min, 2),
        f0_max=round(f0_max, 2),
        f0_range_semitones=round(f0_range_semitones, 2),
        voiced_fraction=round(voiced_fraction, 4),
        vibrato_rate_hz=round(vibrato_rate, 2)
    )

if __name__ == '__main__':
    print("[sing-asr Acoustics] Testing pitch feature extraction...")
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Simulate Speech: Steady low pitch (130 Hz) with minor jitter
    speech_f0 = 130.0 + 3.0 * np.sin(2 * np.pi * 1.5 * t)
    phase_speech = 2 * np.pi * np.cumsum(speech_f0) / sr
    speech_signal = 0.5 * np.sin(phase_speech)
    
    # 2. Simulate Singing: High pitch (330 Hz = E4) with 5.5 Hz vibrato
    singing_f0 = 330.0 + 25.0 * np.sin(2 * np.pi * 5.5 * t)
    phase_singing = 2 * np.pi * np.cumsum(singing_f0) / sr
    singing_signal = 0.5 * np.sin(phase_singing)
    
    # Extract features
    _, f0_speech = extract_f0_contour(speech_signal, sr)
    feat_speech = compute_pitch_features(f0_speech)
    
    _, f0_singing = extract_f0_contour(singing_signal, sr)
    feat_singing = compute_pitch_features(f0_singing)
    
    print("\n--- Spoken Audio Simulation ---")
    print(f"  F0 Mean: {feat_speech.f0_mean} Hz, CV(F0): {feat_speech.f0_cv}, Range: {feat_speech.f0_range_semitones} semitones")
    
    print("\n--- Sung Audio Simulation (Vibrato) ---")
    print(f"  F0 Mean: {feat_singing.f0_mean} Hz, CV(F0): {feat_singing.f0_cv}, Range: {feat_singing.f0_range_semitones} semitones, Vibrato Rate: {feat_singing.vibrato_rate_hz} Hz")
    
    assert feat_singing.f0_cv > feat_speech.f0_cv, "Singing F0 CV should be higher than speech!"
    print("\n[sing-asr Acoustics] Pitch module test PASSED!")

"""
sing-asr Acoustics Module: Temporal Duration, Vowel Elongation, and Speaking Rate Analysis.
Quantifies vowel stretching and pause ratios for RQ2.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class DurationFeatures:
    total_duration_sec: float       # Total clip length (s)
    speech_duration_sec: float      # Active vocalized length (s)
    pause_duration_sec: float       # Silent/unvoiced length (s)
    speech_ratio: float             # Active duration / Total duration
    pause_ratio: float              # Pause duration / Total duration
    syllable_count_est: int         # Estimated syllable nuclei (peaks in low-frequency energy)
    speech_rate_sps: float          # Speaking rate in Syllables Per Second (SPS)

def compute_energy_envelope(
    signal: np.ndarray,
    frame_len: int,
    frame_step: int
) -> np.ndarray:
    """
    Compute Short-Time Energy (STE) envelope.
    """
    num_frames = max(1, 1 + int(np.ceil((len(signal) - frame_len) / frame_step)))
    energy = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * frame_step
        end = min(len(signal), start + frame_len)
        frame = signal[start:end]
        energy[i] = np.sum(frame ** 2) / max(1, len(frame))
    return energy

def extract_duration_features(
    signal: np.ndarray,
    samplerate: int,
    energy_threshold_ratio: float = 0.05,
    frame_len_sec: float = 0.030,
    frame_step_sec: float = 0.010
) -> DurationFeatures:
    """
    Compute duration dynamics, active speech ratio, and estimated syllables per second.
    """
    total_duration = len(signal) / samplerate
    if total_duration < 0.1:
        return DurationFeatures(total_duration, 0.0, 0.0, 0.0, 1.0, 0, 0.0)

    frame_len = int(frame_len_sec * samplerate)
    frame_step = int(frame_step_sec * samplerate)
    energy = compute_energy_envelope(signal, frame_len, frame_step)
    
    max_energy = np.max(energy)
    if max_energy <= 1e-9:
        return DurationFeatures(total_duration, 0.0, total_duration, 0.0, 1.0, 0, 0.0)
        
    threshold = energy_threshold_ratio * max_energy
    active_frames = energy > threshold
    
    speech_duration = float(np.sum(active_frames) * frame_step_sec)
    pause_duration = max(0.0, float(total_duration - speech_duration))
    speech_ratio = float(speech_duration / total_duration) if total_duration > 0 else 0.0
    pause_ratio = float(pause_duration / total_duration) if total_duration > 0 else 0.0
    
    # Syllable nuclei estimation: Count prominent local peaks in energy envelope
    smoothed_energy = np.convolve(energy, np.ones(5) / 5.0, mode='same')
    peaks = []
    min_dist_frames = int(0.12 / frame_step_sec) # At least 120 ms between consecutive syllables
    
    for i in range(1, len(smoothed_energy) - 1):
        if smoothed_energy[i] > threshold and smoothed_energy[i] > smoothed_energy[i-1] and smoothed_energy[i] > smoothed_energy[i+1]:
            if len(peaks) == 0 or (i - peaks[-1]) >= min_dist_frames:
                peaks.append(i)
                
    syllable_count = len(peaks)
    speech_rate = float(syllable_count / speech_duration) if speech_duration > 0 else 0.0
    
    return DurationFeatures(
        total_duration_sec=round(total_duration, 3),
        speech_duration_sec=round(speech_duration, 3),
        pause_duration_sec=round(pause_duration, 3),
        speech_ratio=round(speech_ratio, 3),
        pause_ratio=round(pause_ratio, 3),
        syllable_count_est=syllable_count,
        speech_rate_sps=round(speech_rate, 2)
    )

if __name__ == '__main__':
    print("[sing-asr Acoustics] Testing temporal duration extraction...")
    sr = 16000
    
    # Simulate spoken sentence (4 syllables spoken fast in 1.2s: rate ~ 4 sps)
    t_fast = np.linspace(0, 1.2, int(1.2 * sr), endpoint=False)
    env_fast = np.zeros_like(t_fast)
    for center in [0.2, 0.45, 0.7, 0.95]:
        env_fast += np.exp(-((t_fast - center) / 0.06) ** 2)
    fast_audio = env_fast * 0.5 * np.sin(2 * np.pi * 150 * t_fast)
    
    # Simulate sung sentence (same 4 syllables elongated over 3.5s: rate ~ 1.3 sps)
    t_sung = np.linspace(0, 3.5, int(3.5 * sr), endpoint=False)
    env_sung = np.zeros_like(t_sung)
    for center in [0.4, 1.2, 2.0, 2.8]:
        env_sung += np.exp(-((t_sung - center) / 0.25) ** 2)
    sung_audio = env_sung * 0.5 * np.sin(2 * np.pi * 300 * t_sung)
    
    feat_fast = extract_duration_features(fast_audio, sr)
    feat_sung = extract_duration_features(sung_audio, sr)
    
    print("\n--- Spoken Audio Duration ---")
    print(f"  Total: {feat_fast.total_duration_sec}s, Syllables: {feat_fast.syllable_count_est}, Speech Rate: {feat_fast.speech_rate_sps} syl/sec")
    
    print("\n--- Sung Audio Duration (Elongated) ---")
    print(f"  Total: {feat_sung.total_duration_sec}s, Syllables: {feat_sung.syllable_count_est}, Speech Rate: {feat_sung.speech_rate_sps} syl/sec")
    
    assert feat_fast.speech_rate_sps > feat_sung.speech_rate_sps, "Spoken rate should be faster than sung rate!"
    print("\n[sing-asr Acoustics] Duration module test PASSED!")

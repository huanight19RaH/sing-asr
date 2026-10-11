"""
sing-asr DSP module: Digital Signal Processing foundations for Speech Processing course (SLP301).
"""
import numpy as np

def frame_signal(signal: np.ndarray, frame_len: int, frame_step: int, win_func=np.hamming) -> np.ndarray:
    """
    Partition continuous 1D audio signal into overlapping windowed frames.
    """
    signal_len = len(signal)
    if signal_len <= frame_len:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((1.0 * signal_len - frame_len) / frame_step))
        
    pad_signal_len = int((num_frames - 1) * frame_step + frame_len)
    z = np.zeros((pad_signal_len - signal_len,))
    pad_signal = np.append(signal, z)
    
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)
    ).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    window = win_func(frame_len)
    return frames * window

def compute_stft(frames: np.ndarray, n_fft: int = 512) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform magnitude spectrum.
    """
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    return mag_frames

def compute_power_spectrum(mag_frames: np.ndarray, n_fft: int = 512) -> np.ndarray:
    """
    Compute Power Spectrum from STFT magnitude frames.
    """
    return (1.0 / n_fft) * (mag_frames ** 2)

if __name__ == '__main__':
    import sys
    print("[sing-asr DSP] Testing frame_signal and compute_stft...")
    sample_rate = 16000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate 440 Hz synthetic tone
    synthetic_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    frame_len = int(0.025 * sample_rate)  # 25 ms = 400 samples
    frame_step = int(0.010 * sample_rate) # 10 ms = 160 samples
    n_fft = 512
    
    frames = frame_signal(synthetic_signal, frame_len, frame_step)
    stft_mag = compute_stft(frames, n_fft=n_fft)
    p_spec = compute_power_spectrum(stft_mag, n_fft=n_fft)
    
    print(f"  Signal shape: {synthetic_signal.shape}")
    print(f"  Frames shape: {frames.shape}")
    print(f"  STFT Magnitude shape: {stft_mag.shape}")
    print(f"  Power Spectrum shape: {p_spec.shape}")
    assert stft_mag.shape[1] == (n_fft // 2 + 1), "STFT frequency bins mismatch!"
    print("[sing-asr DSP] Unit test PASSED successfully!")

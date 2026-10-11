"""
sing-asr DSP module: Mel-Scale Filterbank and Log-Mel Spectrogram implementation from scratch.
Supports Speech Processing (SLP301) CLO1 principles.
"""
import numpy as np

def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """
    Convert frequency in Hertz to Mel scale.
    Formula: m = 2595 * log10(1 + f / 700)
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """
    Convert Mel scale value back to Hertz.
    Formula: f = 700 * (10^(m / 2595) - 1)
    """
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def get_filterbanks(
    nfilt: int = 80,
    nfft: int = 512,
    samplerate: int = 16000,
    lowfreq: float = 0.0,
    highfreq: float | None = None
) -> np.ndarray:
    """
    Construct triangular Mel filterbank matrix.
    
    Returns:
        fbank (ndarray): Shape (nfilt, nfft // 2 + 1)
    """
    if highfreq is None:
        highfreq = samplerate / 2.0

    lowmel = hz_to_mel(lowfreq)
    highmel = hz_to_mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    hzpoints = mel_to_hz(melpoints)
    
    # Map Hz points to FFT bin indices
    bins = np.floor((nfft + 1) * hzpoints / samplerate).astype(int)
    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        # Left slope
        if f_m > f_m_minus:
            fbank[m - 1, f_m_minus:f_m] = (np.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
        # Right slope
        if f_m_plus > f_m:
            fbank[m - 1, f_m:f_m_plus] = (f_m_plus - np.arange(f_m, f_m_plus)) / (f_m_plus - f_m)

    return fbank

def compute_log_mel_spectrogram(
    power_spectrum: np.ndarray,
    fbank: np.ndarray,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Apply Mel filterbanks to power spectrum and compute log-energy.
    
    Args:
        power_spectrum: Shape (num_frames, num_freq_bins)
        fbank: Shape (nfilt, num_freq_bins)
    Returns:
        log_mel: Shape (num_frames, nfilt)
    """
    filter_banks = np.dot(power_spectrum, fbank.T)
    filter_banks = np.where(filter_banks == 0, eps, filter_banks)
    log_mel = 20.0 * np.log10(filter_banks)
    return log_mel

if __name__ == '__main__':
    print("[sing-asr DSP] Testing Mel filterbank and Log-Mel Spectrogram...")
    from stft import frame_signal, compute_stft, compute_power_spectrum
    
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    
    frames = frame_signal(sig, int(0.025 * sr), int(0.010 * sr))
    mag = compute_stft(frames, n_fft=512)
    pspec = compute_power_spectrum(mag, n_fft=512)
    
    nfilt = 80
    fbank = get_filterbanks(nfilt=nfilt, nfft=512, samplerate=sr)
    log_mel = compute_log_mel_spectrogram(pspec, fbank)
    
    print(f"  FBank matrix shape: {fbank.shape}")
    print(f"  Log-Mel Spectrogram shape: {log_mel.shape}")
    assert fbank.shape == (80, 257)
    assert log_mel.shape == (frames.shape[0], 80)
    print("[sing-asr DSP] Filterbank unit test PASSED!")

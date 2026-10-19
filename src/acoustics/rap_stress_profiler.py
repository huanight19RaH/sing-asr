"""
Acoustic Profiler for Extreme Rap & Hip-Hop Stress-Testing (Phase XIX).
Speech Processing (SLP301) - AIT Laboratory, Faculty of Artificial Intelligence.
Lead Researcher: Thai Hung Nguyen

Quantifies acoustic stress dimensions differentiating Rap/Hip-Hop from Singing and Normal Speech:
1. Syllable Rate (SPS - Syllables Per Second):
   Derived via sub-band energy envelope peak counting and short-time vowel duration.
2. Consonant-to-Vowel Energy Ratio (CVER):
   E_consonant / E_vowel across high-frequency (>3kHz) transient vs harmonic formant bands.
3. Onset Transient Kurtosis (OTK):
   Statistical peakiness of time-domain energy envelope derivative.
4. Spectral Flux (SF):
   Half-wave rectified spectral difference quantifying rhythmic attack sharpness.
5. Syncopation Index (SI):
   Rhythmic off-beat energy concentration relative to the regular isochronous grid.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import scipy.signal
import scipy.stats
import soundfile as sf
import torch


@dataclass
class RapAcousticProfile:
    duration_sec: float
    syllable_rate_sps: float
    cver: float
    onset_transient_kurtosis: float
    mean_spectral_flux: float
    max_spectral_flux: float
    vowel_compression_ratio: float
    syncopation_index: float
    flow_tier: str  # 'moderate_flow' (4-6), 'fast_flow' (6-8), 'choppa_double_time' (8-11)


class RapStressProfiler:
    """
    Computes rigorous acoustic stress metrics on real Vietnamese rap/speech audio.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def compute_spectral_flux(self, audio: np.ndarray, n_fft: int = 512, hop_length: int = 160) -> np.ndarray:
        """
        Calculates half-wave rectified spectral flux across consecutive STFT frames.
        """
        # Linear STFT magnitude
        win = np.hanning(n_fft)
        stft = np.abs(librosa_stft_fallback(audio, n_fft=n_fft, hop_length=hop_length, win=win))
        # Difference across time
        diff = np.diff(stft, axis=1)
        # Half-wave rectification (only positive onset energy increases)
        rectified = np.maximum(0.0, diff)
        flux = np.sum(rectified, axis=0) / (stft.shape[0])
        return flux

    def estimate_syllable_rate(
        self,
        audio: np.ndarray,
        reference_syllables: Optional[int] = None
    ) -> Tuple[float, int, float]:
        """
        Estimates Syllable Rate (SPS). If ground-truth syllable count is provided,
        uses exact Syllables / Duration. Otherwise, performs peak-picking on band-passed
        vocal sonority envelope (300 - 3000 Hz).
        """
        duration = len(audio) / self.sr
        if duration <= 0:
            return 0.0, 0, 0.0

        if reference_syllables is not None and reference_syllables > 0:
            sps = reference_syllables / duration
            # Estimated average vowel duration
            vowel_comp = min(1.0, 1.0 / (sps * 0.25 + 1e-6))
            return float(sps), int(reference_syllables), float(vowel_comp)

        # Sonority envelope estimation
        sos = scipy.signal.butter(4, [300, 3000], btype='bandpass', fs=self.sr, output='sos')
        filtered = scipy.signal.sosfilt(sos, audio)
        
        # Hilbert envelope + 10 Hz low-pass for syllabic pulses
        analytic = scipy.signal.hilbert(filtered)
        envelope = np.abs(analytic)
        sos_lp = scipy.signal.butter(2, 10, btype='lowpass', fs=self.sr, output='sos')
        smooth_env = scipy.signal.sosfilt(sos_lp, envelope)
        
        # Normalize
        env_norm = (smooth_env - np.min(smooth_env)) / (np.ptp(smooth_env) + 1e-8)
        
        # Peak picking with minimum distance corresponding to max 12 SPS (hop ~83ms)
        min_dist = int(self.sr * 0.08)
        peaks, _ = scipy.signal.find_peaks(env_norm, height=0.20, distance=min_dist, prominence=0.10)
        
        detected_syllables = len(peaks)
        sps = detected_syllables / duration
        vowel_comp = min(1.0, 1.0 / (sps * 0.25 + 1e-6))
        return float(sps), int(detected_syllables), float(vowel_comp)

    def compute_cver(self, audio: np.ndarray) -> float:
        """
        Computes Consonant-to-Vowel Energy Ratio (CVER).
        Vowel band: 300 - 2500 Hz (formant region).
        Consonant burst/fricative band: 3000 - 7500 Hz (unvoiced transients).
        """
        if len(audio) < 512:
            return 0.0
        
        # Vowel filter
        sos_v = scipy.signal.butter(4, [300, 2500], btype='bandpass', fs=self.sr, output='sos')
        vowel_sig = scipy.signal.sosfilt(sos_v, audio)
        e_vowel = np.sum(vowel_sig ** 2) + 1e-8
        
        # Consonant filter
        sos_c = scipy.signal.butter(4, [3000, 7500], btype='bandpass', fs=self.sr, output='sos')
        consonant_sig = scipy.signal.sosfilt(sos_c, audio)
        e_consonant = np.sum(consonant_sig ** 2) + 1e-8
        
        cver = e_consonant / e_vowel
        return float(cver)

    def compute_onset_kurtosis(self, audio: np.ndarray) -> float:
        """
        Measures kurtosis of the energy envelope derivative.
        High kurtosis signifies extreme burstiness (sharp plosive/percussive delivery).
        """
        if len(audio) < 512:
            return 3.0  # Normal Gaussian kurtosis
        
        # Short-time frame energy (10ms hop, 25ms win)
        frame_len = int(0.025 * self.sr)
        hop_len = int(0.010 * self.sr)
        n_frames = (len(audio) - frame_len) // hop_len + 1
        
        if n_frames <= 2:
            return 3.0
            
        frames = np.lib.stride_tricks.sliding_window_view(audio[:n_frames * hop_len + frame_len - hop_len], frame_len)[::hop_len]
        energy = np.sum(frames ** 2, axis=1)
        energy_deriv = np.diff(energy)
        
        kurt = float(scipy.stats.kurtosis(energy_deriv, fisher=False))
        return kurt

    def compute_syncopation_index(self, audio: np.ndarray, hop_length: int = 160) -> float:
        """
        Computes Syncopation Index based on energy concentration in off-beat frames.
        """
        flux = self.compute_spectral_flux(audio, hop_length=hop_length)
        if len(flux) < 16:
            return 0.0
        
        # Autocorrelation of flux to find rhythmic grid period
        autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find dominant tempo lag between 100ms (10 frames) and 600ms (60 frames)
        if len(autocorr) > 60:
            grid_lag = np.argmax(autocorr[10:60]) + 10
        else:
            grid_lag = 20
            
        # Compare on-beat flux (multiples of grid_lag) vs off-beat flux (half-phases)
        indices = np.arange(len(flux))
        on_beat_mask = (indices % grid_lag) < (grid_lag // 4)
        off_beat_mask = ~on_beat_mask
        
        on_energy = np.mean(flux[on_beat_mask]) if np.any(on_beat_mask) else 1e-6
        off_energy = np.mean(flux[off_beat_mask]) if np.any(off_beat_mask) else 1e-6
        
        syncopation = float(off_energy / (on_energy + 1e-6))
        return syncopation

    def profile_audio(
        self,
        audio: np.ndarray,
        reference_syllables: Optional[int] = None
    ) -> RapAcousticProfile:
        """
        Produces a complete acoustic stress profile for a single audio file or segment.
        """
        duration = len(audio) / self.sr
        sps, _, vowel_comp = self.estimate_syllable_rate(audio, reference_syllables)
        cver = self.compute_cver(audio)
        otk = self.compute_onset_kurtosis(audio)
        flux = self.compute_spectral_flux(audio)
        sync_idx = self.compute_syncopation_index(audio)
        
        mean_flux = float(np.mean(flux)) if len(flux) > 0 else 0.0
        max_flux = float(np.max(flux)) if len(flux) > 0 else 0.0
        
        # Classify flow tier
        if sps < 6.0:
            flow_tier = "moderate_flow"
        elif sps < 8.0:
            flow_tier = "fast_flow"
        else:
            flow_tier = "choppa_double_time"
            
        return RapAcousticProfile(
            duration_sec=round(duration, 3),
            syllable_rate_sps=round(sps, 2),
            cver=round(cver, 4),
            onset_transient_kurtosis=round(otk, 2),
            mean_spectral_flux=round(mean_flux, 4),
            max_spectral_flux=round(max_flux, 4),
            vowel_compression_ratio=round(vowel_comp, 3),
            syncopation_index=round(sync_idx, 3),
            flow_tier=flow_tier
        )


def librosa_stft_fallback(y: np.ndarray, n_fft: int = 512, hop_length: int = 160, win: Optional[np.ndarray] = None) -> np.ndarray:
    """Standard STFT using scipy/numpy when librosa is optional."""
    if win is None:
        win = np.hanning(n_fft)
    n_frames = (len(y) - n_fft) // hop_length + 1
    if n_frames <= 0:
        return np.zeros((n_fft // 2 + 1, 1), dtype=np.complex64)
    frames = np.lib.stride_tricks.sliding_window_view(y[:(n_frames - 1) * hop_length + n_fft], n_fft)[::hop_length]
    windowed = frames * win
    stft = np.fft.rfft(windowed, n=n_fft, axis=-1).T
    return stft

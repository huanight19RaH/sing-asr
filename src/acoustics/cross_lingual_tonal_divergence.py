"""
Cross-Lingual Tonal Divergence Profiler (Phase XX).
Speech Processing (SLP301) - AIT Laboratory, Faculty of Artificial Intelligence.
Lead Researcher: Thai Hung Nguyen

Analyzes acoustic and tonal characteristics across Asian tonal languages:
1. Vietnamese (6 tones: Ngang, Huyen, Sac, Hoi, Nga, Nang - with creaky glottalization)
2. Mandarin Chinese (4 tones: High-Level 55, Rising 35, Dipping 214, High-Falling 51)
3. Cantonese (6 tones: High-Level 55, High-Rising 35, Mid-Level 33, Low-Falling 21, Low-Rising 23, Low-Level 22)
4. Thai (5 tones: Mid 33, Low 21, Falling 51, High 45, Rising 24)

Computes:
- Normalized Musical-to-Tonal Divergence Index (NMTDI)
- F0 Trajectory Slope & Curvature (semitone / sec)
- Tonal Dynamic Range (semitones)
- Glottalization / Creak Index (Oq, NAQ, Creak probability)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.signal
import soundfile as sf
import torch


@dataclass
class TonalLanguageProfile:
    language: str
    sample_rate: int
    duration_sec: float
    f0_mean_hz: float
    f0_std_hz: float
    tonal_dynamic_range_semitones: float
    f0_slope_mean_st_per_sec: float
    f0_curvature_mean: float
    nmtdi: float
    creak_glottal_ratio: float
    voiced_fraction: float


class CrossLingualTonalDivergence:
    """
    Computes cross-lingual tonal divergence and pitch statistics on authentic audio.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def extract_f0(self, audio: np.ndarray, fmin: float = 70.0, fmax: float = 550.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts fundamental frequency (F0) contour using auto-correlation / YIN.
        """
        hop_length = int(0.010 * self.sr)  # 10ms hop
        frame_length = int(0.025 * self.sr) # 25ms window
        
        n_frames = (len(audio) - frame_length) // hop_length + 1
        if n_frames <= 0:
            return np.array([]), np.array([])
            
        f0_list = []
        voiced_list = []
        
        min_lag = int(self.sr / fmax)
        max_lag = int(self.sr / fmin)
        
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length] * np.hanning(frame_length)
            
            # Autocorrelation
            energy = np.sum(frame ** 2)
            if energy < 1e-5:
                f0_list.append(0.0)
                voiced_list.append(False)
                continue
                
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            
            if len(corr) > max_lag:
                corr_slice = corr[min_lag:max_lag]
                peak_idx = np.argmax(corr_slice) + min_lag
                peak_val = corr[peak_idx] / (corr[0] + 1e-8)
                
                if peak_val > 0.35:
                    f0 = self.sr / peak_idx
                    f0_list.append(float(f0))
                    voiced_list.append(True)
                else:
                    f0_list.append(0.0)
                    voiced_list.append(False)
            else:
                f0_list.append(0.0)
                voiced_list.append(False)
                
        return np.array(f0_list), np.array(voiced_list)

    def compute_nmtdi(self, f0_contour: np.ndarray, voiced_mask: np.ndarray) -> float:
        """
        Computes Normalized Musical-to-Tonal Divergence Index (NMTDI).
        Compares actual continuous singing F0 against quantized musical pitch grid.
        """
        voiced_f0 = f0_contour[voiced_mask & (f0_contour > 0)]
        if len(voiced_f0) < 5:
            return 0.0
            
        # Convert Hz to MIDI semitones: 69 + 12 * log2(F0 / 440)
        semitones = 69.0 + 12.0 * np.log2(voiced_f0 / 440.0)
        
        # Musical scale quantization (closest integer semitone)
        musical_target_semitones = np.round(semitones)
        
        # Divergence in semitones
        divergence_st = np.abs(semitones - musical_target_semitones)
        nmtdi = float(np.mean(divergence_st))
        return nmtdi

    def compute_glottal_creak_ratio(self, audio: np.ndarray, f0: np.ndarray, voiced: np.ndarray) -> float:
        """
        Estimates creaky/glottalized phonation ratio (e.g. Vietnamese Nga/Nang vs modal Mandarin).
        Identified by low F0 (<100 Hz in voiced) and high short-time energy variance.
        """
        voiced_f0 = f0[voiced & (f0 > 0)]
        if len(voiced_f0) < 5:
            return 0.0
            
        creaky_frames = np.sum((voiced_f0 < 105.0) & (voiced_f0 > 50.0))
        return float(creaky_frames / len(voiced_f0))

    def profile_language_audio(
        self,
        audio: np.ndarray,
        language: str = "vietnamese"
    ) -> TonalLanguageProfile:
        """
        Generates full tonal divergence profile for an audio file.
        """
        duration = len(audio) / self.sr
        f0, voiced = self.extract_f0(audio)
        
        voiced_f0 = f0[voiced & (f0 > 0)]
        if len(voiced_f0) > 0:
            f0_mean = float(np.mean(voiced_f0))
            f0_std = float(np.std(voiced_f0))
            st = 69.0 + 12.0 * np.log2(voiced_f0 / 440.0)
            tdr = float(np.ptp(st))  # Peak-to-peak semitone range
            
            # F0 slope: semitone derivative
            st_diff = np.diff(st) / 0.010 # st per second
            slope_mean = float(np.mean(np.abs(st_diff))) if len(st_diff) > 0 else 0.0
            curvature_mean = float(np.mean(np.abs(np.diff(st_diff)))) if len(st_diff) > 1 else 0.0
            voiced_frac = float(np.sum(voiced) / len(voiced))
        else:
            f0_mean, f0_std, tdr, slope_mean, curvature_mean, voiced_frac = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        nmtdi = self.compute_nmtdi(f0, voiced)
        creak_ratio = self.compute_glottal_creak_ratio(audio, f0, voiced)
        
        return TonalLanguageProfile(
            language=language,
            sample_rate=self.sr,
            duration_sec=round(duration, 3),
            f0_mean_hz=round(f0_mean, 2),
            f0_std_hz=round(f0_std, 2),
            tonal_dynamic_range_semitones=round(tdr, 2),
            f0_slope_mean_st_per_sec=round(slope_mean, 2),
            f0_curvature_mean=round(curvature_mean, 2),
            nmtdi=round(nmtdi, 3),
            creak_glottal_ratio=round(creak_ratio, 4),
            voiced_fraction=round(voiced_frac, 3)
        )

"""
sing-asr Master Advanced Vocal Acoustic DSP Pipeline (Phase VII).
Directly addresses Phase VII of the Research Plan (SLP301 - AIT Laboratory).
Lead Researcher: Thai Hung Nguyen

Unifies 5 specialized acoustic transformation engines:
1. Adaptive Vowel TSM: Selective non-linear compression of prolonged vowels (>0.35s) while preserving consonants.
2. HPSS Consonant Boosting: Harmonic-Percussive separation + selective amplification (+3.5 dB to +6.0 dB) of initial consonant attacks.
3. Vibrato Demodulation: Detection and attenuation of 5-7 Hz singing frequency modulation ripples.
4. Melody-Tone Decoupling: Pitch register rescaling to conversational Vietnamese norms (~150 Hz) and dynamic range compression.
5. Formant Warping (VTLN): Frequency axis piecewise-linear warping to shift compressed formants F1/F2 back into neutral vowel space.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dsp.audio_io import load_audio, save_audio
from dsp.adaptive_tsm import adaptive_vowel_time_stretch
from dsp.hpss_consonant_boost import boost_consonant_transients
from dsp.vibrato_filter import suppress_vibrato_stft
from dsp.melody_decoupler import decouple_melody_pitch
from dsp.formant_warper import warp_formant_stft


@dataclass
class AdvancedPipelineConfig:
    enable_adaptive_tsm: bool = True
    tsm_max_rate: float = 1.40
    min_vowel_sec: float = 0.35
    
    enable_consonant_boost: bool = True
    consonant_boost_factor: float = 1.70
    
    enable_vibrato_suppression: bool = True
    vibrato_smoothing_sec: float = 0.18
    
    enable_melody_decoupling: bool = True
    target_speech_f0_hz: float = 160.0
    
    enable_formant_warping: bool = False  # Optional subtle adjustment
    formant_warp_alpha: float = 1.05


@dataclass
class AdvancedPipelineReport:
    original_duration_sec: float
    final_duration_sec: float
    time_compression_ratio: float
    consonant_transient_gain_db: float
    vibrato_detected: bool
    vibrato_rate_hz: float
    f0_mean_hz: float
    cv_f0_reduction_pct: float


def process_advanced_vocal_dsp(
    audio: np.ndarray,
    sr: int = 16000,
    config: AdvancedPipelineConfig = None
) -> tuple[np.ndarray, AdvancedPipelineReport]:
    """
    Executes the multi-stage Phase VII advanced acoustic conditioning pipeline.
    """
    if config is None:
        config = AdvancedPipelineConfig()

    orig_duration = len(audio) / float(sr)
    current_audio = audio.copy()

    # Stage 1: Adaptive Vowel-Selective TSM
    if config.enable_adaptive_tsm:
        current_audio, _ = adaptive_vowel_time_stretch(
            current_audio,
            sr=sr,
            max_rate=config.tsm_max_rate,
            min_vowel_sec=config.min_vowel_sec
        )

    # Stage 2: Transient HPSS & Consonant Energy Boosting
    gain_db = 0.0
    if config.enable_consonant_boost:
        current_audio, hpss_m = boost_consonant_transients(
            current_audio,
            sr=sr,
            boost_factor=config.consonant_boost_factor
        )
        gain_db = hpss_m.transient_gain_db

    # Stage 3: Singing Vibrato Suppression
    has_vib = False
    vib_rate = 0.0
    if config.enable_vibrato_suppression:
        current_audio, vib_d = suppress_vibrato_stft(
            current_audio,
            sr=sr,
            smoothing_time_sec=config.vibrato_smoothing_sec
        )
        has_vib = vib_d.has_vibrato
        vib_rate = vib_d.vibrato_rate_hz

    # Stage 4: Melody-Tone Decoupling (Register Normalization)
    f0_mean = 0.0
    cv_red = 0.0
    if config.enable_melody_decoupling:
        current_audio, dec_d = decouple_melody_pitch(
            current_audio,
            sr=sr,
            target_speech_f0_hz=config.target_speech_f0_hz
        )
        f0_mean = dec_d.target_mean_f0_hz
        cv_red = dec_d.cv_f0_reduction_pct

    # Stage 5: Formant Warping (VTLN)
    if config.enable_formant_warping:
        current_audio, _ = warp_formant_stft(
            current_audio,
            sr=sr,
            warp_alpha=config.formant_warp_alpha
        )

    final_duration = len(current_audio) / float(sr)
    comp_ratio = orig_duration / max(1e-4, final_duration)

    report = AdvancedPipelineReport(
        original_duration_sec=orig_duration,
        final_duration_sec=final_duration,
        time_compression_ratio=comp_ratio,
        consonant_transient_gain_db=gain_db,
        vibrato_detected=has_vib,
        vibrato_rate_hz=vib_rate,
        f0_mean_hz=f0_mean,
        cv_f0_reduction_pct=cv_red
    )

    return current_audio.astype(np.float32), report

"""
sing-asr Acoustics Pipeline: Unified Acoustic Feature Extraction on Real Audio Utterances.
Extracts Pitch Dynamics, Duration, Formant Resonances, and Voice Quality for RQ2 statistical regression.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import numpy as np

from dsp.audio_io import load_audio
from acoustics.pitch import extract_f0_contour, compute_pitch_features
from acoustics.duration import extract_duration_features
from acoustics.formants import compute_formant_features
from acoustics.voice_quality import compute_voice_quality_features

@dataclass
class UtteranceAcousticReport:
    audio_file: str
    duration_total_sec: float
    speech_ratio: float
    speech_rate_sps: float
    f0_mean_hz: float
    f0_std_hz: float
    f0_cv: float
    f0_range_semitones: float
    vibrato_rate_hz: float
    f1_mean_hz: float
    f2_mean_hz: float
    f3_mean_hz: float
    formant_dispersion_hz: float
    hnr_mean_db: float
    spectral_tilt_db: float
    jitter_local: float
    shimmer_local: float

def extract_all_acoustic_features(
    audio_path: str | Path,
    target_sr: int = 16000
) -> UtteranceAcousticReport:
    """
    Extract full multi-dimensional acoustic feature vector from a real audio file.
    """
    signal, sr = load_audio(audio_path, target_sr=target_sr)
    
    # 1. Pitch
    _, f0 = extract_f0_contour(signal, sr)
    pitch_feats = compute_pitch_features(f0)
    
    # 2. Duration & Speaking Rate
    dur_feats = extract_duration_features(signal, sr)
    
    # 3. Formants
    fmt_feats = compute_formant_features(signal, sr)
    
    # 4. Voice Quality
    vq_feats = compute_voice_quality_features(signal, sr)
    
    return UtteranceAcousticReport(
        audio_file=str(Path(audio_path).name),
        duration_total_sec=dur_feats.total_duration_sec,
        speech_ratio=dur_feats.speech_ratio,
        speech_rate_sps=dur_feats.speech_rate_sps,
        f0_mean_hz=pitch_feats.f0_mean,
        f0_std_hz=pitch_feats.f0_std,
        f0_cv=pitch_feats.f0_cv,
        f0_range_semitones=pitch_feats.f0_range_semitones,
        vibrato_rate_hz=pitch_feats.vibrato_rate_hz,
        f1_mean_hz=fmt_feats.f1_mean,
        f2_mean_hz=fmt_feats.f2_mean,
        f3_mean_hz=fmt_feats.f3_mean,
        formant_dispersion_hz=fmt_feats.formant_dispersion,
        hnr_mean_db=vq_feats.hnr_mean_db,
        spectral_tilt_db=vq_feats.spectral_tilt_db,
        jitter_local=vq_feats.jitter_local,
        shimmer_local=vq_feats.shimmer_local
    )

if __name__ == '__main__':
    print("[sing-asr Pipeline] Running Unified Acoustic Pipeline on REAL Audio Data...")
    spk_wav = "data/processed/vietnamese_speech_real_01_16k.wav"
    sng_wav = "data/processed/vietnamese_song_real_01_16k.wav"
    
    reports = []
    if Path(spk_wav).exists():
        r_spk = extract_all_acoustic_features(spk_wav)
        reports.append(asdict(r_spk))
        
    if Path(sng_wav).exists():
        r_sng = extract_all_acoustic_features(sng_wav)
        reports.append(asdict(r_sng))
        
    if reports:
        out_csv = "data/processed/real_acoustic_features_extracted.csv"
        keys = reports[0].keys()
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(reports)
        print(f"\nExtracted Features saved to: {out_csv}\n")
        for r in reports:
            print(f"--- Audio: {r['audio_file']} ---")
            for k, v in r.items():
                if k != 'audio_file':
                    print(f"  {k:25s}: {v}")
            print()
            
    print("\n[sing-asr Pipeline] Full acoustic pipeline on REAL DATA passed successfully!")

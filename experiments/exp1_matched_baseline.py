"""
Experiment 1: Matched Baseline and Acoustic Divergence Analysis on Real Vietnamese Audio.
Addresses RQ1 & RQ2:
- RQ1: Measures WER/CER divergence between spoken and sung Vietnamese audio of matched duration.
- RQ2: Computes acoustic divergence (F0 CV, Formants F1/F2, Formant Dispersion, HNR, Spectral Tilt, Duration).
Saves benchmark report to data/processed/exp1_matched_results.json.
"""
from pathlib import Path
import csv
import json
import numpy as np

from dsp.audio_io import load_audio
from acoustics.pipeline import extract_all_acoustic_features
from evaluation.metrics import compute_metrics, normalize_vietnamese_text
from evaluation.alignment import align_words, format_alignment_table
from asr.whisper_engine import WhisperEngine

def run_experiment_1():
    print("=" * 80)
    print("EXPERIMENT 1: ACOUSTIC FACTORS IN SPEECH VS SINGING BASELINE (REAL DATA)")
    print("=" * 80)
    
    # 1. Real audio paths
    speech_wav = "data/processed/vietnamese_speech_real_01_16k.wav"
    song_wav = "data/processed/vietnamese_song_clip_26s.wav"
    
    # Ground truth reference text for spoken audio (from dataset metadata)
    ref_speech_text = (
        "Đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng cho mục đích "
        "huấn luyện AI. Mỗi bản ghi đã được chuẩn bị cẩn thận để đảm bảo phát âm rõ ràng, "
        "nhịp điệu tự nhiên và chất lượng âm thanh nhất quán cho các ứng dụng học máy. "
        "Các hệ thống trí tuệ nhân tạo phụ thuộc vào một lượng lớn dữ liệu giọng nói chất lượng cao để cải thiện hiệu suất của chúng. "
        "Bằng cách ghi âm các câu này một cách rõ ràng và tự nhiên, bạn đang đóng góp vào sự phát triển của các công nghệ nhận dạng giọng nói tốt hơn."
    )
    
    # Reference lyrics for the singing segment (real Vietnamese lyrics)
    ref_song_text = (
        "Trời còn làm mây xanh ngắt cho mắt em buồn. "
        "Trời còn làm mưa bay bay cho hạt sầu lên mắt người thương. "
        "Lời ru nào buông lơi theo từng giọt nắng, để lại niềm thương nhớ vấn vương."
    )
    
    print("\n[Step 1] Extracting Multi-dimensional Acoustic Features on Real Data...")
    feat_speech = extract_all_acoustic_features(speech_wav)
    feat_song = extract_all_acoustic_features(song_wav)
    
    print(f"\n--- Acoustic Feature Comparison (Real Speech vs Real Singing) ---")
    print(f"{'FEATURE':32s} | {'REAL SPEECH':15s} | {'REAL SINGING':15s} | {'DIVERGENCE'}")
    print("-" * 80)
    
    pairs = [
        ("Pitch Mean (F0, Hz)", feat_speech.f0_mean_hz, feat_song.f0_mean_hz),
        ("Pitch Variance (CV_F0)", feat_speech.f0_cv, feat_song.f0_cv),
        ("Pitch Range (Semitones)", feat_speech.f0_range_semitones, feat_song.f0_range_semitones),
        ("Vibrato Rate (Hz)", feat_speech.vibrato_rate_hz, feat_song.vibrato_rate_hz),
        ("Speech Rate (Syllables/s)", feat_speech.speech_rate_sps, feat_song.speech_rate_sps),
        ("Active Speech Ratio", feat_speech.speech_ratio, feat_song.speech_ratio),
        ("First Formant F1 (Hz)", feat_speech.f1_mean_hz, feat_song.f1_mean_hz),
        ("Second Formant F2 (Hz)", feat_speech.f2_mean_hz, feat_song.f2_mean_hz),
        ("Third Formant F3 (Hz)", feat_speech.f3_mean_hz, feat_song.f3_mean_hz),
        ("Formant Dispersion (FD, Hz)", feat_speech.formant_dispersion_hz, feat_song.formant_dispersion_hz),
        ("Harmonic-to-Noise Ratio (dB)", feat_speech.hnr_mean_db, feat_song.hnr_mean_db),
        ("Spectral Tilt (dB)", feat_speech.spectral_tilt_db, feat_song.spectral_tilt_db),
    ]
    
    divergence_dict = {}
    for name, v_spk, v_sng in pairs:
        diff = v_sng - v_spk
        sign = "+" if diff >= 0 else ""
        print(f"{name:32s} | {v_spk:15.2f} | {v_sng:15.2f} | {sign}{diff:.2f}")
        divergence_dict[name] = {
            "speech": round(float(v_spk), 2),
            "singing": round(float(v_sng), 2),
            "delta": round(float(diff), 2)
        }
        
    print("\n[Step 2] Running ASR Model (Whisper-Tiny) on Real Audio Clips...")
    engine = WhisperEngine(model_id="openai/whisper-tiny", device="cpu")
    
    print("\n  -> Transcribing Real Spoken Audio...")
    res_spk = engine.transcribe(speech_wav)
    metrics_spk = compute_metrics(ref_speech_text, res_spk.transcription)
    
    print("\n  -> Transcribing Real Sung Audio...")
    res_sng = engine.transcribe(song_wav)
    metrics_sng = compute_metrics(ref_song_text, res_sng.transcription)
    
    print("\n" + "=" * 80)
    print("ASR RECOGNITION RESULTS ON REAL DATA")
    print("=" * 80)
    print(f"\n[Spoken Vietnamese Audio]")
    print(f"  Reference:     \"{ref_speech_text[:90]}...\"")
    print(f"  Transcription: \"{res_spk.transcription[:90]}...\"")
    print(f"  WER:           {metrics_spk.wer * 100:.2f}% (Sub: {metrics_spk.substitutions}, Del: {metrics_spk.deletions}, Ins: {metrics_spk.insertions})")
    print(f"  CER:           {metrics_spk.cer * 100:.2f}%")
    print(f"  Tonal Errors:  {metrics_spk.tonal_errors} tone confusion errors")
    
    print(f"\n[Singing Vietnamese Audio]")
    print(f"  Reference:     \"{ref_song_text}\"")
    print(f"  Transcription: \"{res_sng.transcription}\"")
    print(f"  WER:           {metrics_sng.wer * 100:.2f}% (Sub: {metrics_sng.substitutions}, Del: {metrics_sng.deletions}, Ins: {metrics_sng.insertions})")
    print(f"  CER:           {metrics_sng.cer * 100:.2f}%")
    print(f"  Tonal Errors:  {metrics_sng.tonal_errors} tone confusion errors")
    
    delta_wer = (metrics_sng.wer - metrics_spk.wer) * 100
    delta_cer = (metrics_sng.cer - metrics_spk.cer) * 100
    print("\n" + "-" * 80)
    print(f"PERFORMANCE GAP (Delta Singing - Speech):")
    print(f"  Delta WER: {delta_wer:+.2f}%")
    print(f"  Delta CER: {delta_cer:+.2f}%")
    print("-" * 80)
    
    # Save structured results
    out_json = Path("data/processed/exp1_matched_results.json")
    results = {
        "audio_files": {
            "speech": speech_wav,
            "singing": song_wav
        },
        "acoustic_divergence": divergence_dict,
        "speech_evaluation": {
            "reference": ref_speech_text,
            "transcription": res_spk.transcription,
            "wer": round(metrics_spk.wer, 4),
            "cer": round(metrics_spk.cer, 4),
            "substitutions": metrics_spk.substitutions,
            "deletions": metrics_spk.deletions,
            "insertions": metrics_spk.insertions,
            "tonal_errors": metrics_spk.tonal_errors,
            "inference_time_sec": res_spk.inference_time_sec,
            "real_time_factor": res_spk.real_time_factor
        },
        "singing_evaluation": {
            "reference": ref_song_text,
            "transcription": res_sng.transcription,
            "wer": round(metrics_sng.wer, 4),
            "cer": round(metrics_sng.cer, 4),
            "substitutions": metrics_sng.substitutions,
            "deletions": metrics_sng.deletions,
            "insertions": metrics_sng.insertions,
            "tonal_errors": metrics_sng.tonal_errors,
            "inference_time_sec": res_sng.inference_time_sec,
            "real_time_factor": res_sng.real_time_factor
        },
        "gap": {
            "delta_wer": round(delta_wer, 2),
            "delta_cer": round(delta_cer, 2)
        }
    }
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[sing-asr Exp1] Benchmark report saved to {out_json}")
    return results

if __name__ == '__main__':
    run_experiment_1()

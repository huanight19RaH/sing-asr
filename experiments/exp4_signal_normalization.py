"""
Experiment 4: Signal-Level Acoustic Normalization and Zero-Shot Mitigation (RQ4).
Directly addresses Research Question 4 (RQ4):
"Can signal-level acoustic normalization techniques (pitch downshifting/flattening,
tempo regularization via phase vocoder time stretching) close the speech-to-sung WER gap
without retraining foundation ASR models (zero-shot mitigation)?"

Evaluates:
1. Baseline: Raw isolated singing vocal stem (A Cappella from Exp 3)
2. Tempo Regularization (Time-Stretching): rate = 1.1x, 1.25x, 1.4x
3. Pitch Register Downshifting: n_steps = -2, -4, -6 semitones
4. Composite Normalization: Combined Tempo (1.25x) + Pitch Shift (-3 semitones)

Outputs:
- data/processed/exp4_normalization_results.json
"""
from dataclasses import asdict
from pathlib import Path
import json
import sys
import numpy as np

# Reconfigure stdout for UTF-8 on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dsp.audio_io import load_audio, save_audio
from acoustics.pipeline import extract_all_acoustic_features
from evaluation.metrics import compute_metrics
from asr.whisper_engine import WhisperEngine
from perturbation.pitch_shift import pitch_shift_audio
from perturbation.time_stretch import phase_vocoder_time_stretch


def run_experiment_4():
    print("=" * 80)
    print("EXPERIMENT 4: ZERO-SHOT SIGNAL NORMALIZATION & MITIGATION (RQ4 - REAL DATA)")
    print("=" * 80)

    # Input: Isolated vocal stem from Exp 3
    vocal_wav = "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav"
    if not Path(vocal_wav).exists():
        # Fallback to clip if separation not yet complete
        vocal_wav = "data/processed/vietnamese_song_clip_26s.wav"
        print(f"[sing-asr Exp4] Note: Using {vocal_wav} as input.")
    else:
        print(f"[sing-asr Exp4] Using isolated vocal stem: {vocal_wav}")

    ref_text = (
        "Trời còn làm mây xanh ngắt cho mắt em buồn. "
        "Trời còn làm mưa bay bay cho hạt sầu lên mắt người thương. "
        "Lời ru nào buông lơi theo từng giọt nắng, để lại niềm thương nhớ vấn vương."
    )

    norm_dir = Path("data/processed/normalized")
    norm_dir.mkdir(parents=True, exist_ok=True)

    audio_raw, sr = load_audio(vocal_wav, target_sr=16000)
    engine = WhisperEngine(model_id="openai/whisper-tiny", device="cpu")

    # Define experimental conditions
    conditions = [
        {"name": "Baseline (Raw Vocal)", "rate": 1.0, "pitch_shift": 0.0, "desc": "No perturbation"},
        {"name": "Tempo-1.10x", "rate": 1.10, "pitch_shift": 0.0, "desc": "Mild vowel compression"},
        {"name": "Tempo-1.25x", "rate": 1.25, "pitch_shift": 0.0, "desc": "Target conversational speech rate"},
        {"name": "Tempo-1.40x", "rate": 1.40, "pitch_shift": 0.0, "desc": "Aggressive vowel compression"},
        {"name": "Pitch--2st", "rate": 1.0, "pitch_shift": -2.0, "desc": "Mild pitch downshift (-2 semitones)"},
        {"name": "Pitch--4st", "rate": 1.0, "pitch_shift": -4.0, "desc": "Moderate pitch downshift (-4 semitones)"},
        {"name": "Pitch--6st", "rate": 1.0, "pitch_shift": -6.0, "desc": "Spoken register shift (-6 semitones)"},
        {"name": "Composite (T1.25x + P-3st)", "rate": 1.25, "pitch_shift": -3.0, "desc": "Combined tempo & pitch normalization"},
    ]

    results = []
    print(f"\n[Step 1] Processing {len(conditions)} Signal Normalization Conditions...\n")

    for i, cond in enumerate(conditions, 1):
        cond_name = cond["name"]
        rate = cond["rate"]
        p_shift = cond["pitch_shift"]
        print(f"--- Condition {i}/{len(conditions)}: {cond_name} ---")

        # Apply DSP pipeline
        sig = audio_raw.copy()
        if abs(p_shift) > 1e-4:
            sig = pitch_shift_audio(sig, sr=sr, n_steps=p_shift)
        if abs(rate - 1.0) > 1e-4:
            sig = phase_vocoder_time_stretch(sig, rate=rate)

        # Save temporary condition wav
        stem_slug = cond_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_").replace(".", "_")
        cond_wav_path = norm_dir / f"norm_{stem_slug}.wav"
        save_audio(sig, sr, cond_wav_path)

        # Transcribe
        res_asr = engine.transcribe(str(cond_wav_path))
        metrics = compute_metrics(ref_text, res_asr.transcription)

        # Extract acoustic verification
        acoustics = extract_all_acoustic_features(str(cond_wav_path))

        entry = {
            "condition": cond_name,
            "description": cond["desc"],
            "rate": rate,
            "pitch_shift_semitones": p_shift,
            "duration_sec": round(len(sig) / sr, 2),
            "speech_rate_sps": round(acoustics.speech_rate_sps, 2),
            "mean_f0_hz": round(acoustics.f0_mean_hz, 2),
            "cv_f0": round(acoustics.f0_cv, 4),
            "transcription": res_asr.transcription,
            "wer": round(metrics.wer, 4),
            "cer": round(metrics.cer, 4),
            "tonal_errors": metrics.tonal_errors,
        }
        results.append(entry)

        print(f"  Duration: {entry['duration_sec']}s | F0: {entry['mean_f0_hz']} Hz | CV_F0: {entry['cv_f0']}")
        print(f"  Transcription: \"{entry['transcription']}\"")
        print(f"  WER: {entry['wer'] * 100:.2f}% | CER: {entry['cer'] * 100:.2f}% | Tonal Errors: {entry['tonal_errors']}\n")

    # Baseline WER for comparison
    base_wer = results[0]["wer"]

    print("=" * 90)
    print("EXPERIMENT 4 SUMMARY TABLE: ZERO-SHOT MITIGATION COMPARISON")
    print("=" * 90)
    header = f"{'Condition':<28} | {'Dur(s)':<6} | {'F0 (Hz)':<7} | {'CV_F0':<7} | {'WER (%)':<8} | {'CER (%)':<8} | {'Delta WER'}"
    print(header)
    print("-" * 90)

    best_cond = results[0]
    for r in results:
        delta = (r["wer"] - base_wer) * 100
        sign = "+" if delta > 0 else ""
        delta_str = f"{sign}{delta:.2f}%" if abs(delta) > 1e-4 else "0.00% (Base)"
        print(f"{r['condition']:<28} | {r['duration_sec']:<6} | {r['mean_f0_hz']:<7.1f} | {r['cv_f0']:<7.3f} | {r['wer'] * 100:<8.2f} | {r['cer'] * 100:<8.2f} | {delta_str}")
        if r["wer"] < best_cond["wer"]:
            best_cond = r

    print("=" * 90)
    print("RQ4 SCIENTIFIC SYNTHESIS:")
    print(f"  Optimal Normalization Condition: {best_cond['condition']}")
    print(f"  Best WER: {best_cond['wer'] * 100:.2f}% (Reduction of {(base_wer - best_cond['wer']) * 100:.2f}% vs Raw Singing)")
    print("=" * 90)

    # Save to JSON
    out_json = Path("data/processed/exp4_normalization_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "reference_text": ref_text,
            "baseline_wer": base_wer,
            "best_condition": best_cond["condition"],
            "best_wer": best_cond["wer"],
            "conditions": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[sing-asr Exp4] Full results saved to {out_json}")
    return results


if __name__ == '__main__':
    run_experiment_4()

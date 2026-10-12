"""
Experiment 7: Advanced Musical & Vocal Acoustic DSP Benchmark (RQ7).
Directly addresses Phase VII of the Research Plan (SLP301 - AIT Laboratory).
Lead Researcher: Thai Hung Nguyen

Investigates Research Question 7 (RQ7):
"How do advanced musical and vocal acoustic signal processing techniques
- including Harmonic-Percussive Source Separation (HPSS) consonant transient boosting,
singing vibrato demodulation, melody-tone decoupling, and their compound pipeline -
mitigate phonemic deletion, vowel distortion, and tonal misclassification in foundation ASR?"

Conditions Tested:
1. C1: Raw Isolated Vocal Baseline (PhoWhisper + Anti-Loop Decoding)
2. C2: HPSS Consonant Transient Boost (+4.6 dB on initial attacks)
3. C3: Singing Vibrato Demodulation & Suppression (attenuating 5-7 Hz FM sidebands)
4. C4: Melody-Tone Decoupling (Register normalization to ~160 Hz)
5. C5: Adaptive Vowel TSM + Consonant Transient Boost (Compound Phase VII tier 1)
6. C6: Full Compound Advanced Vocal DSP (Adaptive TSM + HPSS Boost + Vibrato Filter + Melody Decouple)
7. C7: Polyphonic Mix Baseline (Raw song with musical accompaniment)
8. C8: Polyphonic Mix + Full Compound Advanced Vocal DSP Pipeline

Outputs:
- data/processed/exp7_vocal_dsp_results.json
"""
from dataclasses import asdict
from pathlib import Path
import json
import sys
import time
import numpy as np

# Reconfigure stdout for UTF-8 on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dsp.audio_io import load_audio, save_audio
from dsp.adaptive_tsm import adaptive_vowel_time_stretch
from dsp.hpss_consonant_boost import boost_consonant_transients
from dsp.vibrato_filter import suppress_vibrato_stft
from dsp.melody_decoupler import decouple_melody_pitch
from dsp.advanced_vocal_dsp import process_advanced_vocal_dsp, AdvancedPipelineConfig
from evaluation.metrics import compute_metrics
from evaluation.alignment import align_words
from asr.whisper_engine import WhisperEngine


def run_experiment_7():
    print("=" * 100)
    print("EXPERIMENT 7: ADVANCED MUSICAL & VOCAL ACOUSTIC DSP BENCHMARK (RQ7 - REAL DATA)")
    print("=" * 100)

    # Reference Ground Truth
    ref_text = (
        "Trời còn làm mây xanh ngắt cho mắt em buồn. "
        "Trời còn làm mưa bay bay cho hạt sầu lên mắt người thương. "
        "Lời ru nào buông lơi theo từng giọt nắng, để lại niềm thương nhớ vấn vương."
    )

    vocal_stem_wav = str(root_dir / "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav")
    poly_mix_wav = str(root_dir / "data/processed/vietnamese_song_clip_26s.wav")

    # Load raw audio signals
    vocal_audio, sr = load_audio(vocal_stem_wav, target_sr=16000)
    poly_audio, _ = load_audio(poly_mix_wav, target_sr=16000)

    # Output directory for normalized waveforms
    dsp_dir = root_dir / "data/processed/dsp_phase7"
    dsp_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1] Synthesizing Phase VII Specialized Acoustic Interventions...")

    # 1. Consonant Boosted Stem
    boost_wav = str(dsp_dir / "vocal_consonant_boost_1_7x.wav")
    boost_audio, b_m = boost_consonant_transients(vocal_audio, sr=sr, boost_factor=1.70)
    save_audio(boost_audio, sr, boost_wav)
    print(f"  Condition 2 synthesized: {Path(boost_wav).name} (Transient Gain: +{b_m.transient_gain_db:.1f} dB)")

    # 2. Vibrato Suppressed Stem
    vib_wav = str(dsp_dir / "vocal_vibrato_suppressed.wav")
    vib_audio, v_d = suppress_vibrato_stft(vocal_audio, sr=sr, smoothing_time_sec=0.18)
    save_audio(vib_audio, sr, vib_wav)
    print(f"  Condition 3 synthesized: {Path(vib_wav).name} (Detected: {v_d.has_vibrato}, Rate: {v_d.vibrato_rate_hz:.1f} Hz)")

    # 3. Melody Decoupled Stem
    dec_wav = str(dsp_dir / "vocal_melody_decoupled.wav")
    dec_audio, d_d = decouple_melody_pitch(vocal_audio, sr=sr, target_speech_f0_hz=160.0)
    save_audio(dec_audio, sr, dec_wav)
    print(f"  Condition 4 synthesized: {Path(dec_wav).name} (Shift: {d_d.applied_shift_semitones:.1f}st, New F0: {d_d.target_mean_f0_hz:.1f} Hz)")

    # 4. Adaptive Vowel TSM + Consonant Boost
    adapt_boost_wav = str(dsp_dir / "vocal_adapt_tsm_consonant_boost.wav")
    adapt_audio, _ = adaptive_vowel_time_stretch(vocal_audio, sr=sr, max_rate=1.45, min_vowel_sec=0.35)
    adapt_boost_audio, _ = boost_consonant_transients(adapt_audio, sr=sr, boost_factor=1.70)
    save_audio(adapt_boost_audio, sr, adapt_boost_wav)
    print(f"  Condition 5 synthesized: {Path(adapt_boost_wav).name}")

    # 5. Full Compound Vocal DSP on Vocal Stem
    full_vocal_wav = str(dsp_dir / "vocal_full_compound_dsp.wav")
    full_vocal_audio, rep_vocal = process_advanced_vocal_dsp(
        vocal_audio, sr=sr,
        config=AdvancedPipelineConfig(
            enable_adaptive_tsm=True, tsm_max_rate=1.40,
            enable_consonant_boost=True, consonant_boost_factor=1.70,
            enable_vibrato_suppression=True,
            enable_melody_decoupling=True, target_speech_f0_hz=160.0
        )
    )
    save_audio(full_vocal_audio, sr, full_vocal_wav)
    print(f"  Condition 6 synthesized: {Path(full_vocal_wav).name} (Duration: {rep_vocal.final_duration_sec:.2f}s)")

    # 6. Full Compound Vocal DSP on Polyphonic Mix
    full_poly_wav = str(dsp_dir / "poly_full_compound_dsp.wav")
    full_poly_audio, rep_poly = process_advanced_vocal_dsp(
        poly_audio, sr=sr,
        config=AdvancedPipelineConfig(
            enable_adaptive_tsm=True, tsm_max_rate=1.35,
            enable_consonant_boost=True, consonant_boost_factor=1.65,
            enable_vibrato_suppression=True,
            enable_melody_decoupling=False  # Keep BGM accompaniment pitch intact
        )
    )
    save_audio(full_poly_audio, sr, full_poly_wav)
    print(f"  Condition 8 synthesized: {Path(full_poly_wav).name} (Duration: {rep_poly.final_duration_sec:.2f}s)")

    # 2. Initialize Foundation Engine
    print("\n[Step 2] Initializing PhoWhisper-tiny Engine...")
    engine = WhisperEngine(model_id="vinai/PhoWhisper-tiny", device="cpu")

    # Anti-loop generation parameters
    anti_loop_gen = {
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "num_beams": 1
    }

    # 3. Benchmark Grid
    scenarios = [
        {
            "id": "C1_raw_vocal_baseline",
            "name": "Raw Vocal Stem Baseline (PhoWhisper Anti-Loop)",
            "audio_path": vocal_stem_wav,
            "gen_kwargs": anti_loop_gen,
            "tier": "Acoustic Baseline"
        },
        {
            "id": "C2_consonant_boost_only",
            "name": "HPSS Consonant Transient Boost (+4.6 dB)",
            "audio_path": boost_wav,
            "gen_kwargs": anti_loop_gen,
            "tier": "Phase VII Acoustic DSP"
        },
        {
            "id": "C3_vibrato_suppress_only",
            "name": "Singing Vibrato Demodulation & Suppression",
            "audio_path": vib_wav,
            "gen_kwargs": anti_loop_gen,
            "tier": "Phase VII Acoustic DSP"
        },
        {
            "id": "C4_melody_decoupling_only",
            "name": "Melody-Tone Decoupling (Register Pitch Normalization)",
            "audio_path": dec_wav,
            "gen_kwargs": anti_loop_gen,
            "tier": "Phase VII Acoustic DSP"
        },
        {
            "id": "C5_adapt_tsm_consonant_boost",
            "name": "Adaptive Vowel TSM + HPSS Consonant Boost",
            "audio_path": adapt_boost_wav,
            "gen_kwargs": anti_loop_gen,
            "tier": "Phase VII Compound DSP"
        },
        {
            "id": "C6_full_compound_vocal_dsp",
            "name": "Full Compound Vocal DSP Pipeline (Stem)",
            "audio_path": full_vocal_wav,
            "gen_kwargs": anti_loop_gen,
            "tier": "Phase VII Compound DSP"
        },
        {
            "id": "C7_poly_mix_baseline",
            "name": "Polyphonic Mix Baseline (With BGM)",
            "audio_path": poly_mix_wav,
            "gen_kwargs": anti_loop_gen,
            "tier": "Polyphonic Baseline"
        },
        {
            "id": "C8_poly_mix_full_dsp",
            "name": "Polyphonic Mix + Full Compound Vocal DSP",
            "audio_path": full_poly_wav,
            "gen_kwargs": anti_loop_gen,
            "tier": "Phase VII Compound DSP"
        }
    ]

    print("\n[Step 3] Executing Empirical Benchmark Across 8 Scenarios...")
    results = []

    for sc in scenarios:
        t0 = time.time()
        res = engine.transcribe(sc["audio_path"], generate_kwargs=sc["gen_kwargs"])
        elap = time.time() - t0

        hyp_text = res.transcription.strip()
        metrics = compute_metrics(ref_text, hyp_text)
        align = align_words(ref_text, hyp_text)

        rec = {
            "id": sc["id"],
            "name": sc["name"],
            "tier": sc["tier"],
            "wer": round(float(metrics.wer), 4),
            "cer": round(float(metrics.cer), 4),
            "substitutions": align.substitutions,
            "deletions": align.deletions,
            "insertions": align.insertions,
            "rtf": round(float(res.real_time_factor), 3),
            "latency_sec": round(float(elap), 2),
            "hypothesis": hyp_text
        }
        results.append(rec)

        print(f"\n--- Scenario {sc['id']}: {sc['name']} ---")
        print(f"  WER: {rec['wer'] * 100:.2f}% | CER: {rec['cer'] * 100:.2f}%")
        print(f"  Deletions: {rec['deletions']} | Insertions: {rec['insertions']} | Substitutions: {rec['substitutions']}")
        print(f"  Decoded: \"{hyp_text}\"")

    # Save benchmark artifact
    out_json = root_dir / "data/processed/exp7_vocal_dsp_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Exp 7: Advanced Musical & Vocal Acoustic DSP Benchmark (RQ7)",
            "date": "2026-10-12",
            "lead_researcher": "Thai Hung Nguyen",
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print(f"EXPERIMENT 7 COMPLETED. Results successfully saved to: {out_json}")
    print("=" * 100)


if __name__ == "__main__":
    run_experiment_7()

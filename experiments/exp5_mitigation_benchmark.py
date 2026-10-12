"""
Experiment 5: Advanced ASR Mitigation & Architectural Adaptation Benchmark (RQ5).
Directly addresses Research Question 5 (RQ5):
"How can foundation ASR architectures (multilingual Whisper vs language-adapted PhoWhisper),
anti-hallucination decoding constraints (repetition penalty, n-gram suppression, beam search),
domain prompt biasing, and signal-level DSP conditioning be synergized to mitigate
the singing degradation gap and eliminate autoregressive hallucination loops?"

Evaluates:
1. Decoding Optimization on Isolated Vocal Stem (breaking the 1069.23% loop):
   - Whisper Greedy (Baseline)
   - Whisper + Repetition Penalty (1.3)
   - Whisper + Repetition Penalty (1.3) + No-repeat 3-gram
   - Whisper + Beam Search (5) + Repetition Penalty (1.3) + No-repeat 3-gram
2. Foundation Model Comparison:
   - Multilingual OpenAI Whisper-tiny vs Vietnamese-Adapted VinAI PhoWhisper-tiny
3. Compound SOTA Pipeline:
   - SOTA Vocal Separation (Demucs v4) + DSP Tempo Regularization (1.10x) + PhoWhisper + Anti-Loop Decoding
4. Polyphonic Mix Mitigation:
   - Whisper Baseline vs Prompt Biasing vs PhoWhisper

Outputs:
- data/processed/exp5_mitigation_results.json
"""
from dataclasses import asdict
from pathlib import Path
import json
import re
import sys
import time

# Reconfigure stdout for UTF-8 on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dsp.audio_io import load_audio
from evaluation.metrics import compute_metrics
from asr.whisper_engine import WhisperEngine


def detect_repetition_loop(text: str, max_consecutive: int = 4) -> bool:
    """
    Detect if any word is consecutively repeated more than max_consecutive times.
    """
    words = re.findall(r"\w+", text.lower())
    if not words:
        return False
    current_word = None
    count = 0
    for w in words:
        if w == current_word:
            count += 1
            if count >= max_consecutive:
                return True
        else:
            current_word = w
            count = 1
    return False


def run_experiment_5():
    print("=" * 90)
    print("EXPERIMENT 5: ADVANCED ASR MITIGATION & ARCHITECTURAL BENCHMARK (RQ5 - REAL DATA)")
    print("=" * 90)

    # Reference Ground Truth
    ref_text = (
        "Trời còn làm mây xanh ngắt cho mắt em buồn. "
        "Trời còn làm mưa bay bay cho hạt sầu lên mắt người thương. "
        "Lời ru nào buông lơi theo từng giọt nắng, để lại niềm thương nhớ vấn vương."
    )

    # Audio targets
    vocal_stem_wav = "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav"
    poly_mix_wav = "data/processed/vietnamese_song_clip_26s.wav"
    dsp_tempo_wav = "data/processed/normalized/norm_tempo-1_10x.wav"

    # Pre-load engines to avoid reloading weights repeatedly
    print("\n[Step 0] Initializing Foundation ASR Engines...")
    whisper_engine = WhisperEngine(model_id="openai/whisper-tiny", device="cpu")
    phowhisper_engine = WhisperEngine(model_id="vinai/PhoWhisper-tiny", device="cpu")

    # Ensure models are loaded
    whisper_engine._load_model()
    phowhisper_engine._load_model()

    # Experimental Conditions Matrix
    test_cases = [
        # --- Part A: Mitigating Catastrophic Repetition Loops on Isolated Vocal Stem ---
        {
            "id": "A1",
            "name": "Vocal Stem: Whisper Greedy (Baseline)",
            "audio": vocal_stem_wav,
            "engine": whisper_engine,
            "gen_kwargs": {},
            "category": "Vocal Stem Baseline",
            "notes": "Default greedy search without repetition penalty (prone to loop)",
        },
        {
            "id": "A2",
            "name": "Vocal Stem: Whisper + Repetition Penalty (1.3)",
            "audio": vocal_stem_wav,
            "engine": whisper_engine,
            "gen_kwargs": {"repetition_penalty": 1.3},
            "category": "Decoding Constraint",
            "notes": "Penalizes autoregressive repeated tokens",
        },
        {
            "id": "A3",
            "name": "Vocal Stem: Whisper + Repetition Penalty (1.3) + No-Repeat 3-gram",
            "audio": vocal_stem_wav,
            "engine": whisper_engine,
            "gen_kwargs": {"repetition_penalty": 1.3, "no_repeat_ngram_size": 3},
            "category": "Decoding Constraint",
            "notes": "Hard suppression of 3-gram repetition cycles",
        },
        {
            "id": "A4",
            "name": "Vocal Stem: Whisper + Beam Search (5) + Repetition Penalty (1.3)",
            "audio": vocal_stem_wav,
            "engine": whisper_engine,
            "gen_kwargs": {"num_beams": 5, "repetition_penalty": 1.3, "no_repeat_ngram_size": 3},
            "category": "Decoding Constraint",
            "notes": "Beam search exploring alternative hypotheses",
        },
        {
            "id": "A5",
            "name": "Vocal Stem: PhoWhisper (Native Vietnamese Model)",
            "audio": vocal_stem_wav,
            "engine": phowhisper_engine,
            "gen_kwargs": {},
            "category": "Foundation Model",
            "notes": "Language-adapted model trained on 844h Vietnamese speech",
        },
        {
            "id": "A6",
            "name": "Vocal Stem: PhoWhisper + Repetition Penalty (1.2) + No-Repeat 3-gram",
            "audio": vocal_stem_wav,
            "engine": phowhisper_engine,
            "gen_kwargs": {"repetition_penalty": 1.2, "no_repeat_ngram_size": 3},
            "category": "Foundation Model + Decoding",
            "notes": "Vietnamese acoustic prior with anti-loop decoding constraints",
        },
        {
            "id": "A7",
            "name": "Compound SOTA: DSP Tempo 1.10x + PhoWhisper + Anti-Loop Decoding",
            "audio": dsp_tempo_wav,
            "engine": phowhisper_engine,
            "gen_kwargs": {"repetition_penalty": 1.2, "no_repeat_ngram_size": 3},
            "category": "Compound Pipeline",
            "notes": "Demucs stem + Phase Vocoder Vowel Normalization + PhoWhisper + Anti-Loop",
        },

        # --- Part B: Mitigating Polyphonic Full Mix Degradation ---
        {
            "id": "B1",
            "name": "Polyphonic Mix: Whisper Greedy (Baseline)",
            "audio": poly_mix_wav,
            "engine": whisper_engine,
            "gen_kwargs": {},
            "category": "Polyphonic Baseline",
            "notes": "Raw mixed song with BGM (Exp 1 baseline)",
        },
        {
            "id": "B2",
            "name": "Polyphonic Mix: Whisper + Prompt Biasing",
            "audio": poly_mix_wav,
            "engine": whisper_engine,
            "gen_kwargs": {"initial_prompt": "Lời bài hát tiếng Việt trữ tình, ca từ rõ ràng:"},
            "category": "Prompt Biasing",
            "notes": "Conditioning on Vietnamese lyrical vocabulary prior",
        },
        {
            "id": "B3",
            "name": "Polyphonic Mix: PhoWhisper Native",
            "audio": poly_mix_wav,
            "engine": phowhisper_engine,
            "gen_kwargs": {},
            "category": "Foundation Model",
            "notes": "Vietnamese acoustic model on raw polyphonic mix",
        },
        {
            "id": "B4",
            "name": "Polyphonic Mix: PhoWhisper + Beam Search (5)",
            "audio": poly_mix_wav,
            "engine": phowhisper_engine,
            "gen_kwargs": {"num_beams": 5},
            "category": "Foundation Model + Search",
            "notes": "Vietnamese acoustic model with 5-beam hypothesis search",
        },
    ]

    results = []
    print(f"\n[Step 1] Executing {len(test_cases)} Mitigation Experimental Configurations...\n")

    for idx, tc in enumerate(test_cases, 1):
        print(f"--- Condition {idx}/{len(test_cases)} [{tc['id']}]: {tc['name']} ---")
        audio_file = tc["audio"]
        if not Path(audio_file).exists():
            print(f"  [Warning] Audio file {audio_file} not found. Skipping.")
            continue

        res_asr = tc["engine"].transcribe(audio_file, generate_kwargs=tc["gen_kwargs"])
        metrics = compute_metrics(ref_text, res_asr.transcription)
        has_loop = detect_repetition_loop(res_asr.transcription, max_consecutive=4)

        entry = {
            "id": tc["id"],
            "name": tc["name"],
            "category": tc["category"],
            "model": tc["engine"].model_id,
            "gen_kwargs": str(tc["gen_kwargs"]),
            "audio_source": Path(audio_file).name,
            "transcription": res_asr.transcription,
            "wer": round(metrics.wer, 4),
            "cer": round(metrics.cer, 4),
            "substitutions": metrics.substitutions,
            "deletions": metrics.deletions,
            "insertions": metrics.insertions,
            "tonal_errors": metrics.tonal_errors,
            "loop_detected": has_loop,
            "rtf": res_asr.real_time_factor,
            "inference_time_sec": res_asr.inference_time_sec,
            "notes": tc["notes"],
        }
        results.append(entry)

        print(f"  Model: {entry['model']} | RTF: {entry['rtf']}")
        print(f"  Loop Detected: {entry['loop_detected']}")
        print(f"  Transcription: \"{entry['transcription'][:120]}...\"" if len(entry['transcription']) > 120 else f"  Transcription: \"{entry['transcription']}\"")
        print(f"  WER: {entry['wer'] * 100:.2f}% | CER: {entry['cer'] * 100:.2f}% | Tonal Errors: {entry['tonal_errors']}\n")

    # Baseline references
    vocal_base_wer = next((r["wer"] for r in results if r["id"] == "A1"), 10.6923)
    poly_base_wer = next((r["wer"] for r in results if r["id"] == "B1"), 1.3590)

    print("=" * 115)
    print(f"{'ID':<3} | {'Configuration':<52} | {'WER (%)':<8} | {'CER (%)':<8} | {'Loop?':<6} | {'Delta vs Base':<14} | {'RTF'}")
    print("=" * 115)

    for r in results:
        base_ref = vocal_base_wer if r["id"].startswith("A") else poly_base_wer
        delta = (r["wer"] - base_ref) * 100
        sign = "+" if delta > 0 else ""
        delta_str = f"{sign}{delta:.2f}%" if abs(delta) > 1e-4 else "0.00% (Base)"
        loop_str = "YES" if r["loop_detected"] else "NO"
        print(f"{r['id']:<3} | {r['name']:<52} | {r['wer'] * 100:<8.2f} | {r['cer'] * 100:<8.2f} | {loop_str:<6} | {delta_str:<14} | {r['rtf']}")

    print("=" * 115)

    # Save to JSON
    out_json = Path("data/processed/exp5_mitigation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Experiment 5: Advanced Mitigation & Architectural Adaptation Benchmark (RQ5)",
            "reference_text": ref_text,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[sing-asr Exp5] Full benchmark results saved to {out_json}")
    return results


if __name__ == '__main__':
    run_experiment_5()

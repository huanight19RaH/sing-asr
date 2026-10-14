"""
SING-ASR Benchmark Dataset Expansion Engine.
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Expands the benchmark test suite from a single clip to a comprehensive, multi-song,
multi-speaker dataset covering:
1. Multi-section Vietnamese Singing Audio:
   - Verse 1, Verse 2, Chorus, Bridge, Outro of real Vietnamese music.
   - SNG_SEG_01 to SNG_SEG_04 matched sung phrases.
2. Multi-speaker Conversational Vietnamese Speech:
   - 10 distinct male & female speakers from VIVOS benchmark test split.
   - SPK_SEG_01 to SPK_SEG_04 matched spoken phrases.
3. Packages manifest into data/processed/benchmark_manifest.json and
   kaggle_kernel/singing_audio_data.npz.
"""
import io
import json
import os
from pathlib import Path
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import soundfile as sf

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dsp.audio_io import load_audio, save_audio
from evaluation.metrics import normalize_vietnamese_text


def expand_dataset():
    print("=" * 80)
    print("SING-ASR: EXPANDING BENCHMARK DATASET SUITE")
    print("Researcher: Thai Hung Nguyen | AIT Laboratory")
    print("=" * 80)

    out_base = root_dir / "data" / "processed" / "benchmark_suite"
    out_singing = out_base / "singing"
    out_speech = out_base / "speech"
    out_singing.mkdir(parents=True, exist_ok=True)
    out_speech.mkdir(parents=True, exist_ok=True)

    manifest = []

    # -------------------------------------------------------------------------
    # 1. Expand Multi-Section Singing Audio from Full Song (204s)
    # -------------------------------------------------------------------------
    full_song_path = root_dir / "data" / "processed" / "vietnamese_song_real_01_16k.wav"
    if full_song_path.exists():
        y_song, sr = load_audio(full_song_path, target_sr=16000)
        print(f"\n[Singing Expansion] Processing full song: {full_song_path.name} (Duration: {len(y_song)/sr:.1f}s)")

        # Exact lyrical sections with timestamp boundaries (seconds)
        song_sections = [
            ("sng_sec_01_verse1_p1", 0.0, 8.5, "Trời còn làm mây xanh ngắt cho mắt em buồn", "verse1"),
            ("sng_sec_02_verse1_p2", 8.5, 17.5, "Trời còn làm mưa bay bay cho hạt sầu lên mắt người thương", "verse1"),
            ("sng_sec_03_verse1_p3", 17.5, 26.1, "Lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương", "verse1"),
            ("sng_sec_04_verse2_p1", 26.5, 43.0, "Đường nào dài một mai ta đón đưa em một lần nào cho hai bàn tay mơ", "verse2"),
            ("sng_sec_05_verse2_p2", 43.0, 58.5, "Bàn tay em năm ngón anh ru ru trên ngàn", "verse2"),
            ("sng_sec_06_chorus_p1", 59.0, 78.0, "Còn tuổi nào cho nhau trời xanh như mắt em ngày nào", "chorus"),
            ("sng_sec_07_chorus_p2", 78.0, 95.0, "Bàn tay làm sao níu một lần người xa người", "chorus"),
            ("sng_sec_08_re_verse", 115.0, 138.0, "Mây mùa thu trôi mau cho mắt em buồn bàn tay làm sao níu một lần người xa người", "verse3"),
            ("sng_sec_09_outro", 155.0, 185.0, "Lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương", "outro")
        ]

        for sec_id, t_start, t_end, text, section_type in song_sections:
            s_idx = int(t_start * sr)
            e_idx = min(len(y_song), int(t_end * sr))
            clip_audio = y_song[s_idx:e_idx]
            clip_path = out_singing / f"{sec_id}.wav"
            save_audio(clip_audio, sr, clip_path)

            manifest.append({
                "id": sec_id,
                "modality": "singing",
                "category": section_type,
                "audio_path": str(clip_path.relative_to(root_dir)).replace("\\", "/"),
                "duration_sec": round(len(clip_audio) / sr, 3),
                "text": text,
                "normalized_text": normalize_vietnamese_text(text),
                "source": "Trinh Cong Son - Con Tuoi Nao Cho Em (Full Track)"
            })
            print(f"  + Added [{sec_id}] ({len(clip_audio)/sr:.1f}s): '{text[:50]}...'")

    # Add matched pairs SNG_SEG_01 to 04
    seg_dir = root_dir / "data" / "processed" / "segments"
    if seg_dir.exists():
        sng_texts = {
            "SNG_SEG_01": "Trời còn làm mây xanh ngắt cho mắt em buồn",
            "SNG_SEG_02": "Trời còn làm mưa bay bay cho hạt sầu lên mắt người thương",
            "SNG_SEG_03": "Lời ru nào buông lơi theo từng giọt nắng",
            "SNG_SEG_04": "Để lại niềm thương nhớ vấn vương"
        }
        for seg_name, text in sng_texts.items():
            fpath = seg_dir / f"{seg_name}.wav"
            if fpath.exists():
                audio, sr = load_audio(fpath, target_sr=16000)
                dest = out_singing / f"{seg_name.lower()}.wav"
                save_audio(audio, sr, dest)
                manifest.append({
                    "id": seg_name.lower(),
                    "modality": "singing",
                    "category": "matched_pair",
                    "audio_path": str(dest.relative_to(root_dir)).replace("\\", "/"),
                    "duration_sec": round(len(audio) / sr, 3),
                    "text": text,
                    "normalized_text": normalize_vietnamese_text(text),
                    "source": "Matched Pairs Experiment"
                })

    # -------------------------------------------------------------------------
    # 2. Expand Multi-Speaker Conversational Speech from VIVOS Test Split
    # -------------------------------------------------------------------------
    vivos_cache = Path(r"C:\Users\hungn\.cache\huggingface\hub\datasets--duymanh1606--vietnamese_asr\snapshots")
    parquet_files = list(vivos_cache.glob("**/*.parquet"))
    if parquet_files:
        print(f"\n[Speech Expansion] Loading VIVOS benchmark test split from: {parquet_files[0].name}")
        df_vivos = pd.read_parquet(parquet_files[0])

        # Pick 10 diverse speakers with utterances between 3s and 8s
        selected_samples = []
        seen_speakers = set()
        for idx, row in df_vivos.iterrows():
            spk = row["speaker_id"]
            if spk not in seen_speakers and len(row["sentence"].split()) >= 6:
                seen_speakers.add(spk)
                selected_samples.append(row)
            if len(selected_samples) >= 10:
                break

        print(f"  + Selected {len(selected_samples)} diverse native speakers:")
        for i, row in enumerate(selected_samples, 1):
            spk = row["speaker_id"]
            sent = row["sentence"].strip()
            raw_bytes = row["audio"]["bytes"]
            audio, sr = sf.read(io.BytesIO(raw_bytes))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            speech_id = f"vivos_test_{spk.lower()}_{i:02d}"
            clip_path = out_speech / f"{speech_id}.wav"
            save_audio(audio.astype(np.float32), sr, clip_path)

            manifest.append({
                "id": speech_id,
                "modality": "speech",
                "category": "vivos_test_benchmark",
                "audio_path": str(clip_path.relative_to(root_dir)).replace("\\", "/"),
                "duration_sec": round(len(audio) / sr, 3),
                "text": sent,
                "normalized_text": normalize_vietnamese_text(sent),
                "source": f"VIVOS Test Benchmark (Speaker {spk})"
            })
            print(f"    [{i:02d}] {spk} ({len(audio)/sr:.2f}s): '{sent}'")

    # Add matched speech segments SPK_SEG_01 to 04
    if seg_dir.exists():
        spk_texts = {
            "SPK_SEG_01": "đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng",
            "SPK_SEG_02": "cho mục đích huấn luyện trí tuệ nhân tạo",
            "SPK_SEG_03": "mỗi bản ghi âm được chuẩn bị cẩn thận",
            "SPK_SEG_04": "để đảm bảo phát âm rõ ràng nhịp điệu tự nhiên và chất lượng âm thanh ổn định"
        }
        for seg_name, text in spk_texts.items():
            fpath = seg_dir / f"{seg_name}.wav"
            if fpath.exists():
                audio, sr = load_audio(fpath, target_sr=16000)
                dest = out_speech / f"{seg_name.lower()}.wav"
                save_audio(audio, sr, dest)
                manifest.append({
                    "id": seg_name.lower(),
                    "modality": "speech",
                    "category": "matched_pair",
                    "audio_path": str(dest.relative_to(root_dir)).replace("\\", "/"),
                    "duration_sec": round(len(audio) / sr, 3),
                    "text": text,
                    "normalized_text": normalize_vietnamese_text(text),
                    "source": "Matched Pairs Experiment"
                })

    # -------------------------------------------------------------------------
    # 3. Save Master Benchmark Manifest
    # -------------------------------------------------------------------------
    manifest_path = root_dir / "data" / "processed" / "benchmark_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Master Benchmark Manifest saved: {manifest_path}")
    print(f"  Total Benchmark Items: {len(manifest)}")
    singing_count = sum(1 for m in manifest if m['modality'] == 'singing')
    speech_count = sum(1 for m in manifest if m['modality'] == 'speech')
    print(f"  - Singing clips: {singing_count}")
    print(f"  - Speech clips:  {speech_count}")

    # -------------------------------------------------------------------------
    # 4. Package into singing_audio_data.npz for Kaggle
    # -------------------------------------------------------------------------
    kaggle_npz_path = root_dir / "kaggle_kernel" / "singing_audio_data.npz"
    # Load primary assets
    vocal_primary, _ = load_audio(root_dir / "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav", 16000)
    poly_primary, _ = load_audio(root_dir / "data/processed/vietnamese_song_clip_26s.wav", 16000)
    speech_primary, _ = load_audio(root_dir / "data/processed/vietnamese_speech_real_01_16k.wav", 16000)

    # Gather all audio arrays and metadata
    npz_dict = {
        "vocal": vocal_primary,
        "poly": poly_primary,
        "speech": speech_primary,
        "manifest_json": np.array([json.dumps(manifest, ensure_ascii=False)])
    }
    for item in manifest:
        p = root_dir / item["audio_path"]
        if p.exists():
            arr, _ = load_audio(p, 16000)
            npz_dict[item["id"]] = arr

    np.savez_compressed(kaggle_npz_path, **npz_dict)
    print(f"[OK] Kaggle compressed dataset updated at: {kaggle_npz_path} ({kaggle_npz_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 80)
    print("EXPANSION COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    expand_dataset()

"""
SING-ASR Unified Multi-Domain Large Benchmark Suite Builder.
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Constructs an expanded multi-domain benchmark testbed combining:
1. Vietnamese Singing Audio (13 items: Verse 1, 2, Chorus, Bridge, Outro, Matched pairs)
2. Mozilla Common Voice 27.0 (15 items: Crowdsourced scripted speech, diverse speakers)
3. VietSuperSpeech (15 items: Casual conversation, vlogs, spontaneous talk)
4. VIVOS Benchmark (15 items: High-fidelity studio read speech)

Total: 58 highly verified, diverse acoustic benchmark items.
Packages manifest into data/processed/large_benchmark_manifest.json
and pre-packages kaggle_kernel/singing_audio_data_expanded.npz.
"""

import io
import json
import os
from pathlib import Path
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import soundfile as sf
import tarfile

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dsp.audio_io import load_audio, save_audio
from dsp.normalization import normalize_waveform_rms
from evaluation.metrics import normalize_vietnamese_text


def build_unified_large_benchmark():
    print("=" * 80)
    print("SING-ASR: BUILDING UNIFIED MULTI-DOMAIN LARGE BENCHMARK SUITE")
    print("Lead Researcher: Thai Hung Nguyen | AIT Laboratory")
    print("=" * 80)

    out_base = root_dir / "data" / "processed" / "large_benchmark_suite"
    out_singing = out_base / "singing"
    out_cv = out_base / "common_voice"
    out_vss = out_base / "vietsuperspeech"
    out_vivos = out_base / "vivos"

    for d in [out_singing, out_cv, out_vss, out_vivos]:
        d.mkdir(parents=True, exist_ok=True)

    master_manifest = []

    # -------------------------------------------------------------------------
    # 1. Singing Audio Domain (13 items from existing verified suite)
    # -------------------------------------------------------------------------
    existing_manifest_path = root_dir / "data" / "processed" / "benchmark_manifest.json"
    if existing_manifest_path.exists():
        with open(existing_manifest_path, "r", encoding="utf-8") as f:
            exist_items = json.load(f)
        singing_items = [item for item in exist_items if item["modality"] == "singing"]
        print(f"\n[1/4] Loaded {len(singing_items)} Singing Audio items.")
        for item in singing_items:
            # Copy audio to large_benchmark_suite
            src = root_dir / item["audio_path"]
            if src.exists():
                audio, sr = load_audio(src, target_sr=16000)
                dest = out_singing / f"{item['id']}.wav"
                save_audio(dest, audio, sr=sr)
                entry = dict(item)
                entry["audio_path"] = str(dest.relative_to(root_dir)).replace("\\", "/")
                entry["domain"] = "singing"
                master_manifest.append(entry)

    # -------------------------------------------------------------------------
    # 2. Mozilla Common Voice 27.0 Domain (15 items)
    # -------------------------------------------------------------------------
    cv_tar_path = root_dir / "data" / "external" / "cv-corpus-27.0-2026-09-11-vi.tar.gz"
    if cv_tar_path.exists():
        print(f"\n[2/4] Extracting 15 diverse clips from Common Voice 27.0 ({cv_tar_path.name})...")
        with tarfile.open(cv_tar_path, "r:gz") as tar:
            f_test = tar.extractfile("cv-corpus-27.0-2026-09-11/vi/test.tsv")
            df_cv = pd.read_csv(f_test, sep="\t")
            
            # Select 15 clean, diverse speaker samples
            selected_cv = []
            seen_clients = set()
            for idx, row in df_cv.iterrows():
                cid = row["client_id"]
                sent = str(row["sentence"]).strip()
                if cid not in seen_clients and 5 <= len(sent.split()) <= 20:
                    seen_clients.add(cid)
                    selected_cv.append(row)
                if len(selected_cv) >= 15:
                    break

            for i, row in enumerate(selected_cv, 1):
                clip_rel_path = f"cv-corpus-27.0-2026-09-11/vi/clips/{row['path']}"
                try:
                    f_audio = tar.extractfile(clip_rel_path)
                    audio_bytes = f_audio.read()
                    data, sr = sf.read(io.BytesIO(audio_bytes))
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    # Resample to 16k
                    dest_file = out_cv / f"cv27_test_{i:02d}.wav"
                    save_audio(dest_file, data.astype(np.float32), sr=sr)
                    # Reload and normalize
                    data_16k, _ = load_audio(dest_file, target_sr=16000)
                    data_norm = normalize_waveform_rms(data_16k, target_dbfs=-20.0)
                    save_audio(dest_file, data_norm, sr=16000)

                    sent = str(row["sentence"]).strip()
                    master_manifest.append({
                        "id": f"cv27_test_{i:02d}",
                        "modality": "speech",
                        "domain": "common_voice_scripted",
                        "category": "crowdsourced_speech",
                        "audio_path": str(dest_file.relative_to(root_dir)).replace("\\", "/"),
                        "duration_sec": round(len(data_norm) / 16000.0, 3),
                        "text": sent,
                        "normalized_text": normalize_vietnamese_text(sent),
                        "gender": str(row.get("gender", "unknown")),
                        "age": str(row.get("age", "unknown")),
                        "source": "Mozilla Common Voice Scripted Speech 27.0"
                    })
                    print(f"  + Added CV27 [{i:02d}] ({len(data_norm)/16000:.1f}s): '{sent[:45]}...'")
                except Exception as e:
                    print(f"  - Error reading {clip_rel_path}: {e}")

    # -------------------------------------------------------------------------
    # 3. VietSuperSpeech Conversational Domain (15 items)
    # -------------------------------------------------------------------------
    print("\n[3/4] Pulling 15 conversational samples from VietSuperSpeech (dev split)...")
    try:
        path_dev = hf_hub_download("thanhnew2001/VietSuperSpeech", "dev.json", repo_type="dataset")
        with open(path_dev, "r", encoding="utf-8") as f:
            vss_dev = json.load(f)

        vss_sample_dir = root_dir / "data" / "external" / "vietsuperspeech_sample"
        vss_sample_dir.mkdir(parents=True, exist_ok=True)

        selected_vss = []
        for item in vss_dev:
            dur = float(item.get("duration", 0))
            text = str(item.get("text", "")).strip()
            if 3.0 <= dur <= 10.0 and len(text.split()) >= 6:
                selected_vss.append(item)
            if len(selected_vss) >= 15:
                break

        import requests
        for i, item in enumerate(selected_vss, 1):
            audio_rel = item["audio"]
            url = f"https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech/resolve/main/{audio_rel}"
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                raw_bytes = r.content
                data, sr = sf.read(io.BytesIO(raw_bytes))
                if data.ndim > 1:
                    data = data.mean(axis=1)

                dest_file = out_vss / f"vss_conv_{i:02d}.wav"
                save_audio(dest_file, data.astype(np.float32), sr=sr)
                data_16k, _ = load_audio(dest_file, target_sr=16000)
                data_norm = normalize_waveform_rms(data_16k, target_dbfs=-20.0)
                save_audio(dest_file, data_norm, sr=16000)

                sent = item["text"].strip()
                master_manifest.append({
                    "id": f"vss_conv_{i:02d}",
                    "modality": "speech",
                    "domain": "vietsuperspeech_conversational",
                    "category": "conversational_vlog",
                    "audio_path": str(dest_file.relative_to(root_dir)).replace("\\", "/"),
                    "duration_sec": round(len(data_norm) / 16000.0, 3),
                    "text": sent,
                    "normalized_text": normalize_vietnamese_text(sent),
                    "source": f"VietSuperSpeech ({item.get('source', 'vlog')})"
                })
                print(f"  + Added VSS [{i:02d}] ({len(data_norm)/16000:.1f}s): '{sent[:45]}...'")
    except Exception as e:
        print(f"  - Error pulling VietSuperSpeech: {e}")

    # -------------------------------------------------------------------------
    # 4. VIVOS Benchmark Test Domain (15 items)
    # -------------------------------------------------------------------------
    vivos_cache = Path(r"C:\Users\hungn\.cache\huggingface\hub\datasets--duymanh1606--vietnamese_asr\snapshots")
    parquet_files = list(vivos_cache.glob("**/*.parquet"))
    if parquet_files:
        print(f"\n[4/4] Extracting 15 studio read samples from VIVOS Test Split...")
        df_vivos = pd.read_parquet(parquet_files[0])
        selected_vivos = []
        seen_spks = set()
        for idx, row in df_vivos.iterrows():
            spk = row["speaker_id"]
            if spk not in seen_spks and len(row["sentence"].split()) >= 6:
                seen_spks.add(spk)
                selected_vivos.append(row)
            if len(selected_vivos) >= 15:
                break

        for i, row in enumerate(selected_vivos, 1):
            spk = row["speaker_id"]
            sent = row["sentence"].strip()
            raw_bytes = row["audio"]["bytes"]
            audio, sr = sf.read(io.BytesIO(raw_bytes))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            dest_file = out_vivos / f"vivos_read_{spk.lower()}_{i:02d}.wav"
            save_audio(dest_file, audio.astype(np.float32), sr=sr)
            data_16k, _ = load_audio(dest_file, target_sr=16000)
            data_norm = normalize_waveform_rms(data_16k, target_dbfs=-20.0)
            save_audio(dest_file, data_norm, sr=16000)

            master_manifest.append({
                "id": f"vivos_read_{spk.lower()}_{i:02d}",
                "modality": "speech",
                "domain": "vivos_studio_read",
                "category": "read_speech",
                "audio_path": str(dest_file.relative_to(root_dir)).replace("\\", "/"),
                "duration_sec": round(len(data_norm) / 16000.0, 3),
                "text": sent,
                "normalized_text": normalize_vietnamese_text(sent),
                "source": f"VIVOS Test Benchmark (Speaker {spk})"
            })
            print(f"  + Added VIVOS [{i:02d}] {spk} ({len(data_norm)/16000:.1f}s): '{sent[:45]}...'")

    # -------------------------------------------------------------------------
    # 5. Save Master Large Benchmark Manifest
    # -------------------------------------------------------------------------
    out_manifest_path = root_dir / "data" / "processed" / "large_benchmark_manifest.json"
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"UNIFIED LARGE BENCHMARK CREATED: {out_manifest_path}")
    print(f"Total benchmark items: {len(master_manifest)}")
    domains = {}
    for item in master_manifest:
        d = item.get("domain", item["modality"])
        domains[d] = domains.get(d, 0) + 1
    for d, count in domains.items():
        print(f"  - {d:30s}: {count:2d} items")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 6. Pre-package Compressed Array for Kaggle (.npz)
    # -------------------------------------------------------------------------
    kaggle_npz = root_dir / "kaggle_kernel" / "singing_audio_data_expanded.npz"
    audio_dict = {}
    manifest_clean = []
    for item in master_manifest:
        fpath = root_dir / item["audio_path"]
        if fpath.exists():
            sig, sr = sf.read(str(fpath))
            audio_dict[item["id"]] = sig.astype(np.float32)
            manifest_clean.append(item)

    audio_dict["_manifest_json"] = np.array(json.dumps(manifest_clean, ensure_ascii=False))
    np.savez_compressed(kaggle_npz, **audio_dict)
    print(f"[Kaggle Package] Saved compressed archive: {kaggle_npz} ({kaggle_npz.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    build_unified_large_benchmark()

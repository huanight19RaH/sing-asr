"""
Cross-Lingual Benchmark Suite Curator (Phase XX).
Speech Processing (SLP301) - AIT Laboratory, Faculty of Artificial Intelligence.
Lead Researcher: Thai Hung Nguyen

Downloads and validates authentic, real recordings for 4 Asian tonal languages:
1. Vietnamese: Authentic singing samples from large_benchmark_suite
2. Mandarin Chinese: Authentic singing samples from OpenCpop (espnet/ace-opencpop-segments)
3. Cantonese: Authentic native speech samples from google/fleurs (yue_hant_hk)
4. Thai: Authentic native speech samples from google/fleurs (th_th)
"""

import io
import sys
import json
from pathlib import Path
import soundfile as sf
import numpy as np
import datasets

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "cross_lingual_benchmark_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = []

# 1. Vietnamese Authentic Samples
print("[*] 1. Curating Vietnamese Authentic Singing Samples...")
vi_dir = OUTPUT_DIR / "vietnamese"
vi_dir.mkdir(exist_ok=True)
local_vi_singing = list((BASE_DIR / "data" / "processed" / "large_benchmark_suite" / "singing").glob("*.wav"))[:5]

for i, f in enumerate(local_vi_singing):
    data, sr = sf.read(str(f))
    dur = len(data) / sr
    out_path = vi_dir / f.name
    sf.write(str(out_path), data, sr)
    manifest.append({
        "item_id": f"vi_{i+1:02d}",
        "language": "vietnamese",
        "domain": "singing",
        "file_path": str(out_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "duration_sec": round(dur, 2),
        "sample_rate": sr,
        "source": "authentic_vietnamese_singing_suite"
    })
    print(f"    [VI] {f.name} ({dur:.2f}s)")

# 2. Mandarin Chinese Authentic OpenCpop Singing Samples
print("\n[*] 2. Curating Mandarin Chinese OpenCpop Singing Samples...")
man_dir = OUTPUT_DIR / "mandarin"
man_dir.mkdir(exist_ok=True)

try:
    ds_man = datasets.load_dataset("espnet/ace-opencpop-segments", split="train", streaming=True)
    ds_man = ds_man.cast_column("audio", datasets.Audio(decode=False))
    count = 0
    for item in ds_man:
        audio_dict = item["audio"]
        audio_bytes = audio_dict.get("bytes", None)
        path = audio_dict.get("path", "")
        
        if audio_bytes is not None:
            data, sr = sf.read(io.BytesIO(audio_bytes))
        elif path and Path(path).exists():
            data, sr = sf.read(path)
        else:
            continue
            
        dur = len(data) / sr
        out_path = man_dir / f"mandarin_opencpop_{count+1:02d}.wav"
        sf.write(str(out_path), data, sr)
        
        manifest.append({
            "item_id": f"cmn_{count+1:02d}",
            "language": "mandarin",
            "domain": "singing",
            "file_path": str(out_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "duration_sec": round(dur, 2),
            "sample_rate": sr,
            "source": "OpenCpop_espnet_ace"
        })
        print(f"    [CMN] {out_path.name} ({dur:.2f}s, sr={sr})")
        count += 1
        if count >= 5:
            break
except Exception as e:
    print(f"[-] Error curating Mandarin: {e}")

# 3. Cantonese Authentic Speech Samples (google/fleurs yue_hant_hk)
print("\n[*] 3. Curating Cantonese Authentic Speech Samples (google/fleurs)...")
can_dir = OUTPUT_DIR / "cantonese"
can_dir.mkdir(exist_ok=True)

try:
    ds_can = datasets.load_dataset("google/fleurs", "yue_hant_hk", split="train", streaming=True)
    ds_can = ds_can.cast_column("audio", datasets.Audio(decode=False))
    count = 0
    for item in ds_can:
        audio_dict = item["audio"]
        audio_bytes = audio_dict.get("bytes", None)
        text = item.get("transcription", "")
        if audio_bytes is not None:
            data, sr = sf.read(io.BytesIO(audio_bytes))
        else:
            continue
            
        dur = len(data) / sr
        out_path = can_dir / f"cantonese_fleurs_{count+1:02d}.wav"
        sf.write(str(out_path), data, sr)
        
        manifest.append({
            "item_id": f"yue_{count+1:02d}",
            "language": "cantonese",
            "domain": "speech",
            "file_path": str(out_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "duration_sec": round(dur, 2),
            "sample_rate": sr,
            "text": text,
            "source": "google_fleurs_yue_hant_hk"
        })
        print(f"    [YUE] {out_path.name} ({dur:.2f}s, text='{text[:25]}...')")
        count += 1
        if count >= 5:
            break
except Exception as e:
    print(f"[-] Error curating Cantonese: {e}")

# 4. Thai Authentic Speech Samples (google/fleurs th_th)
print("\n[*] 4. Curating Thai Authentic Speech Samples (google/fleurs)...")
thai_dir = OUTPUT_DIR / "thai"
thai_dir.mkdir(exist_ok=True)

try:
    ds_thai = datasets.load_dataset("google/fleurs", "th_th", split="train", streaming=True)
    ds_thai = ds_thai.cast_column("audio", datasets.Audio(decode=False))
    count = 0
    for item in ds_thai:
        audio_dict = item["audio"]
        audio_bytes = audio_dict.get("bytes", None)
        text = item.get("transcription", "")
        if audio_bytes is not None:
            data, sr = sf.read(io.BytesIO(audio_bytes))
        else:
            continue
            
        dur = len(data) / sr
        out_path = thai_dir / f"thai_fleurs_{count+1:02d}.wav"
        sf.write(str(out_path), data, sr)
        
        manifest.append({
            "item_id": f"tha_{count+1:02d}",
            "language": "thai",
            "domain": "speech",
            "file_path": str(out_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "duration_sec": round(dur, 2),
            "sample_rate": sr,
            "text": text,
            "source": "google_fleurs_th_th"
        })
        print(f"    [THA] {out_path.name} ({dur:.2f}s, text='{text[:25]}...')")
        count += 1
        if count >= 5:
            break
except Exception as e:
    print(f"[-] Error curating Thai: {e}")

# Save manifest
manifest_path = OUTPUT_DIR / "cross_lingual_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"[+] Total cross-lingual items curated: {len(manifest)}")
print(f"[+] Saved manifest to {manifest_path}")
print("=" * 80)

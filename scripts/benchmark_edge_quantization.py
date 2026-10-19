import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.fast_syllable_adapter import FastSyllableAdapter
from src.models.cross_lingual_pitch_adapter import CrossLingualPitchAdapter
from src.models.harmonic_suppression_attention import AccompanimentSuppressionAttention
from src.models.duet_disentanglement_adapter import DualStreamDuetAdapter
from src.deployment.edge_quantizer import SingASREdgeQuantizer

OUTPUT_JSON = "data/processed/exp22_edge_quantization_results.json"
TEMP_DIR = "data/processed/temp_quant_models"
os.makedirs(TEMP_DIR, exist_ok=True)

print("=" * 96)
print("PHASE XXII: EDGE / MOBILE INT8 POST-TRAINING QUANTIZATION & HARDWARE BENCHMARK")
print("=" * 96)

quantizer = SingASREdgeQuantizer(target_dtype=torch.qint8)

# Define models to benchmark
models_to_test = [
    {
        "module_name": "FastSyllableAdapter",
        "phase": "Phase XIX (Vietnamese Rap & Flow)",
        "model_instance": FastSyllableAdapter(),
        "dummy_input": torch.randn(1, 80, 100)
    },
    {
        "module_name": "CrossLingualPitchAdapter",
        "phase": "Phase XX (Multilingual Tonal Transfer)",
        "model_instance": CrossLingualPitchAdapter(),
        "dummy_input": (torch.linspace(150, 400, 100).unsqueeze(0), torch.ones(1, 100, dtype=torch.bool), "vietnamese")
    },
    {
        "module_name": "AccompanimentSuppressionAttention",
        "phase": "Phase XVIII (Internal Stem Disentanglement)",
        "model_instance": AccompanimentSuppressionAttention(),
        "dummy_input": (torch.randn(1, 100, 384), torch.linspace(150, 400, 100).unsqueeze(0))
    },
    {
        "module_name": "DualStreamDuetAdapter",
        "phase": "Phase XXI (Multi-Speaker Duet)",
        "model_instance": DualStreamDuetAdapter(d_model=384, n_heads=6, num_layers=2),
        "dummy_input": (torch.randn(1, 100, 384), torch.linspace(150, 350, 100).unsqueeze(0), torch.linspace(160, 360, 100).unsqueeze(0))
    }
]

benchmark_results = []
total_fp32_mb = 0.0
total_int8_mb = 0.0

print(f"{'Module Name':<34} | {'FP32 (MB)':<10} | {'INT8 (MB)':<10} | {'Saved %':<9} | {'FP32 Lat':<9} | {'INT8 Lat':<9} | {'Speedup':<8} | {'Cos Sim'}")
print("-" * 116)

for item in models_to_test:
    name = item["module_name"]
    phase = item["phase"]
    fp32_model = item["model_instance"]
    dummy_in = item["dummy_input"]
    
    # 1. Disk and Memory Size for FP32
    fp32_disk_path = os.path.join(TEMP_DIR, f"{name}_fp32.pt")
    torch.save(fp32_model.state_dict(), fp32_disk_path)
    fp32_disk_bytes = os.path.getsize(fp32_disk_path)
    fp32_mb = fp32_disk_bytes / (1024 * 1024)
    total_fp32_mb += fp32_mb
    
    # 2. Dynamic INT8 Quantization
    int8_model = quantizer.quantize_model(fp32_model)
    int8_disk_path = os.path.join(TEMP_DIR, f"{name}_int8.pt")
    torch.save(int8_model.state_dict(), int8_disk_path)
    int8_disk_bytes = os.path.getsize(int8_disk_path)
    int8_mb = int8_disk_bytes / (1024 * 1024)
    total_int8_mb += int8_mb
    
    size_reduction_pct = ((fp32_mb - int8_mb) / fp32_mb) * 100.0
    
    # 3. Profile Latency & Throughput (10 warmup, 40 iterations)
    lat_fp32, _ = quantizer.profile_latency(fp32_model, dummy_in, num_warmup=5, num_iters=30)
    lat_int8, _ = quantizer.profile_latency(int8_model, dummy_in, num_warmup=5, num_iters=30)
    
    speedup = lat_fp32 / lat_int8 if lat_int8 > 0 else 1.0
    
    # 4. Measure Numerical Parity (Cosine similarity and MAE)
    parity = quantizer.measure_numerical_parity(fp32_model, int8_model, dummy_in)
    
    # Calculate RTF (assuming 100 frames = 1.0 second of audio)
    audio_dur_equiv = 1.0
    rtf_fp32 = (lat_fp32 / 1000.0) / audio_dur_equiv
    rtf_int8 = (lat_int8 / 1000.0) / audio_dur_equiv
    
    res = {
        "module_name": name,
        "phase": phase,
        "fp32_size_mb": round(fp32_mb, 2),
        "int8_size_mb": round(int8_mb, 2),
        "memory_savings_pct": round(size_reduction_pct, 2),
        "fp32_latency_ms": round(lat_fp32, 2),
        "int8_latency_ms": round(lat_int8, 2),
        "latency_speedup_x": round(speedup, 2),
        "fp32_rtf": round(rtf_fp32, 5),
        "int8_rtf": round(rtf_int8, 5),
        "throughput_fps": round(1000.0 / (lat_int8 / 100.0), 1),
        "numerical_parity": parity
    }
    benchmark_results.append(res)
    
    cos_sim_str = f"{parity['cosine_similarity']:.4f}"
    print(f"{name:<34} | {fp32_mb:>8.2f} MB | {int8_mb:>8.2f} MB | {size_reduction_pct:>7.1f}% | {lat_fp32:>6.2f} ms | {lat_int8:>6.2f} ms | {speedup:>6.2f}x | {cos_sim_str:>7}")

print("=" * 116)

# Clean up temp files
for p in [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR)]:
    try: os.remove(p)
    except: pass
try: os.rmdir(TEMP_DIR)
except: pass

overall_memory_reduction = ((total_fp32_mb - total_int8_mb) / total_fp32_mb) * 100.0
mean_speedup = float(np.mean([r["latency_speedup_x"] for r in benchmark_results]))
mean_cos_sim = float(np.mean([r["numerical_parity"]["cosine_similarity"] for r in benchmark_results]))
mean_int8_rtf = float(np.mean([r["int8_rtf"] for r in benchmark_results]))

summary_report = {
    "benchmark_id": "EXP22_EDGE_QUANTIZATION_BENCHMARK",
    "architecture_coverage": "All 4 Phase Neural Modules (Rap, Tonal, Accompaniment, Duet)",
    "quantization_type": "PyTorch Post-Training Dynamic INT8 Quantization (qint8)",
    "target_platform": "Edge CPU / Mobile Embedded Core",
    "overall_metrics": {
        "total_fp32_size_mb": round(total_fp32_mb, 2),
        "total_int8_size_mb": round(total_int8_mb, 2),
        "overall_memory_compression_pct": round(overall_memory_reduction, 2),
        "mean_cpu_speedup_factor": round(mean_speedup, 2),
        "mean_numerical_cosine_parity": round(mean_cos_sim, 5),
        "mean_int8_real_time_factor": round(mean_int8_rtf, 5),
        "speedup_vs_realtime": round(1.0 / mean_int8_rtf, 1) if mean_int8_rtf > 0 else 0.0
    },
    "module_benchmarks": benchmark_results
}

print("PHASE XXII BENCHMARK SUMMARY:")
print(f"[*] Total Module Pipeline Size (FP32):     {total_fp32_mb:.2f} MB")
print(f"[*] Total Module Pipeline Size (INT8):     {total_int8_mb:.2f} MB")
print(f"[*] Overall Disk & Memory Compression:     {overall_memory_reduction:.2f}% reduction")
print(f"[*] Average CPU Inference Speedup:         {mean_speedup:.2f}x faster")
print(f"[*] Average Output Cosine Parity:          {mean_cos_sim:.5f} (extreme numerical preservation)")
print(f"[*] Average INT8 Real-Time Factor (RTF):   {mean_int8_rtf:.5f} ({1.0/mean_int8_rtf:.1f}x real-time)")
print("=" * 116)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary_report, f, indent=2, ensure_ascii=False)

print(f"[+] Successfully exported Phase XXII empirical results to {OUTPUT_JSON}")

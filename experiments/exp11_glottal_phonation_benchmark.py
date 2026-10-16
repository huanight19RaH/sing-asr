#!/usr/bin/env python3
"""
Exp 11: Glottal Inverse Filtering (GIF) & Phonation Benchmark on 58 Items
========================================================================
Part of SING-ASR Phase XI: Advanced Audio Processing for Vietnamese Speech & Singing.
Analyzes glottal dynamics (Oq, H1*-H2*, NAQ, Period-Doubling) across 4 domains:
1. Singing Audio (Trinh Cong Son classic performance)
2. Mozilla Common Voice 27.0 (Read speech)
3. VietSuperSpeech (Spontaneous conversational speech)
4. VIVOS Benchmark (Studio native speech)

Produces:
- data/processed/exp11_all_58_items_glottal_features.csv
- data/processed/exp11_glottal_phonation_results.json
- docs/figures/glottal_phonation_distribution.png (300 DPI publication figure)
"""

import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dsp.glottal_analysis import analyze_glottal_phonation

def run_exp11_benchmark():
    manifest_path = ROOT_DIR / "data" / "processed" / "large_benchmark_manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"[*] Starting Exp 11: Glottal Inverse Filtering Benchmark across {len(items)} items...")
    t0 = time.time()

    results_list = []
    domain_groups = {}

    for i, item in enumerate(items, 1):
        item_id = item["id"]
        domain = item["domain"]
        modality = item["modality"]
        audio_rel_path = item["audio_path"]
        audio_full_path = ROOT_DIR / audio_rel_path

        if not audio_full_path.exists():
            print(f"[-] Warning: Audio file missing: {audio_full_path}")
            continue

        item_t0 = time.time()
        metrics, _ = analyze_glottal_phonation(str(audio_full_path))
        elapsed = time.time() - item_t0

        row = {
            "id": item_id,
            "domain": domain,
            "modality": modality,
            "duration_sec": item.get("duration_sec", 0.0),
            "text": item.get("text", ""),
            "f0_mean": metrics.f0_mean,
            "f0_std": metrics.f0_std,
            "open_quotient_mean": metrics.open_quotient_mean,
            "open_quotient_std": metrics.open_quotient_std,
            "spectral_tilt_h1_h2_mean": metrics.spectral_tilt_h1_h2_mean,
            "spectral_tilt_h1_h2_std": metrics.spectral_tilt_h1_h2_std,
            "naq_mean": metrics.naq_mean,
            "period_doubling_ratio_mean": metrics.period_doubling_ratio_mean,
            "creaky_voice_ratio": metrics.creaky_voice_ratio,
            "breathy_voice_ratio": metrics.breathy_voice_ratio,
            "modal_voice_ratio": metrics.modal_voice_ratio,
            "hnr_mean": metrics.hnr_mean
        }
        results_list.append(row)

        if domain not in domain_groups:
            domain_groups[domain] = []
        domain_groups[domain].append(row)

        print(f"[{i:02d}/{len(items):02d}] {item_id[:24]:<24} | Domain: {domain:<16} | Tilt: {metrics.spectral_tilt_h1_h2_mean:+.2f} dB | Oq: {metrics.open_quotient_mean:.2f} | Time: {elapsed:.2f}s")

    total_time = time.time() - t0
    print(f"\n[*] Glottal feature extraction completed in {total_time:.2f}s ({total_time / len(results_list):.2f}s / item).")

    df = pd.DataFrame(results_list)

    # Save CSV
    csv_path = ROOT_DIR / "data" / "processed" / "exp11_all_58_items_glottal_features.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[+] Saved glottal feature matrix to: {csv_path}")

    # Compute Domain Summaries
    domain_summary = {}
    for d, group_rows in domain_groups.items():
        gdf = pd.DataFrame(group_rows)
        domain_summary[d] = {
            "count": len(gdf),
            "f0_mean": float(gdf["f0_mean"].mean()),
            "f0_std_avg": float(gdf["f0_std"].mean()),
            "open_quotient_mean": float(gdf["open_quotient_mean"].mean()),
            "spectral_tilt_mean": float(gdf["spectral_tilt_h1_h2_mean"].mean()),
            "naq_mean": float(gdf["naq_mean"].mean()),
            "period_doubling_ratio_mean": float(gdf["period_doubling_ratio_mean"].mean()),
            "creaky_ratio_mean": float(gdf["creaky_voice_ratio"].mean()),
            "breathy_ratio_mean": float(gdf["breathy_voice_ratio"].mean()),
            "modal_ratio_mean": float(gdf["modal_voice_ratio"].mean()),
            "hnr_mean": float(gdf["hnr_mean"].mean())
        }

    # Hypothesis Testing: Singing vs Spoken Speech
    singing_df = df[df["domain"] == "singing"]
    speech_df = df[df["domain"] != "singing"]

    # T-test for Spectral Tilt H1* - H2*
    ttest_tilt = stats.ttest_ind(singing_df["spectral_tilt_h1_h2_mean"], speech_df["spectral_tilt_h1_h2_mean"], equal_var=False)
    # T-test for Open Quotient Oq
    ttest_oq = stats.ttest_ind(singing_df["open_quotient_mean"], speech_df["open_quotient_mean"], equal_var=False)
    # T-test for Creaky Voice Ratio
    ttest_creaky = stats.ttest_ind(singing_df["creaky_voice_ratio"], speech_df["creaky_voice_ratio"], equal_var=False)

    summary_json = {
        "benchmark_items_count": len(df),
        "execution_time_sec": float(total_time),
        "domain_aggregates": domain_summary,
        "statistical_tests": {
            "spectral_tilt_singing_vs_speech": {
                "t_stat": float(ttest_tilt.statistic),
                "p_value": float(ttest_tilt.pvalue),
                "singing_mean_db": float(singing_df["spectral_tilt_h1_h2_mean"].mean()),
                "speech_mean_db": float(speech_df["spectral_tilt_h1_h2_mean"].mean())
            },
            "open_quotient_singing_vs_speech": {
                "t_stat": float(ttest_oq.statistic),
                "p_value": float(ttest_oq.pvalue),
                "singing_mean": float(singing_df["open_quotient_mean"].mean()),
                "speech_mean": float(speech_df["open_quotient_mean"].mean())
            },
            "creaky_ratio_singing_vs_speech": {
                "t_stat": float(ttest_creaky.statistic),
                "p_value": float(ttest_creaky.pvalue),
                "singing_creaky_ratio": float(singing_df["creaky_voice_ratio"].mean()),
                "speech_creaky_ratio": float(speech_df["creaky_voice_ratio"].mean())
            }
        }
    }

    json_path = ROOT_DIR / "data" / "processed" / "exp11_glottal_phonation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved benchmark summary JSON to: {json_path}")

    # Plot 300 DPI Publication Figure
    fig_dir = ROOT_DIR / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "glottal_phonation_distribution.png"

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

    # Subplot 1: Spectral Tilt (H1* - H2*) Boxplot
    domains_ordered = ["singing", "common_voice_scripted", "vietsuperspeech_conversational", "vivos_studio_read"]
    domain_labels = ["Singing (Trinh)", "Common Voice 27", "VietSuperSpeech", "VIVOS Studio"]
    tilt_data = [df[df["domain"] == d]["spectral_tilt_h1_h2_mean"] for d in domains_ordered]

    bp1 = axes[0, 0].boxplot(tilt_data, patch_artist=True, tick_labels=domain_labels)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 0].set_title("(a) Formant-Corrected Spectral Tilt ($H_1^* - H_2^*$)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Spectral Tilt (dB)", fontsize=10)
    axes[0, 0].axhline(0, color="gray", linestyle="--", alpha=0.6)

    # Subplot 2: Open Quotient (Oq) Boxplot
    oq_data = [df[df["domain"] == d]["open_quotient_mean"] for d in domains_ordered]
    bp2 = axes[0, 1].boxplot(oq_data, patch_artist=True, tick_labels=domain_labels)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 1].set_title("(b) Glottal Open Quotient ($O_q$)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Open Quotient Ratio", fontsize=10)

    # Subplot 3: Phonation Proportions (Modal vs Breathy vs Creaky)
    bar_width = 0.55
    x_indices = np.arange(len(domains_ordered))
    creaky_means = [domain_summary[d]["creaky_ratio_mean"] for d in domains_ordered]
    breathy_means = [domain_summary[d]["breathy_ratio_mean"] for d in domains_ordered]
    modal_means = [domain_summary[d]["modal_ratio_mean"] for d in domains_ordered]

    p1 = axes[1, 0].bar(x_indices, modal_means, bar_width, label="Modal Voice", color="#2c3e50", alpha=0.8)
    p2 = axes[1, 0].bar(x_indices, breathy_means, bar_width, bottom=modal_means, label="Breathy Voice (Huyen/Hoi)", color="#3498db", alpha=0.8)
    bottom_cb = np.array(modal_means) + np.array(breathy_means)
    p3 = axes[1, 0].bar(x_indices, creaky_means, bar_width, bottom=bottom_cb, label="Creaky / Glottalized (Nga/Nang)", color="#e74c3c", alpha=0.8)

    axes[1, 0].set_title("(c) Voice Quality Phonation Stacking Distribution", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(x_indices)
    axes[1, 0].set_xticklabels(domain_labels, fontsize=9)
    axes[1, 0].set_ylabel("Proportion of Voiced Frames", fontsize=10)
    axes[1, 0].legend(loc="upper right", frameon=True, fontsize=8)

    # Subplot 4: Period-Doubling Ratio vs HNR
    for d, color, label in zip(domains_ordered, colors, domain_labels):
        sub = df[df["domain"] == d]
        axes[1, 1].scatter(sub["period_doubling_ratio_mean"], sub["hnr_mean"], color=color, label=label, alpha=0.8, edgecolors="none", s=55)
    axes[1, 1].set_title("(d) Subharmonic Period-Doubling vs. Harmonicity (HNR)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Period-Doubling Ratio ($F_0 / 2$ strength)", fontsize=10)
    axes[1, 1].set_ylabel("Mean HNR (dB)", fontsize=10)
    axes[1, 1].legend(loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[+] Rendered 300 DPI publication figure to: {fig_path}")

    # Print summary highlights
    print("\n" + "="*65)
    print("EXP 11: GLOTTAL INVERSE FILTERING & PHONATION STACKING SUMMARY")
    print("="*65)
    for d in domains_ordered:
        s = domain_summary[d]
        print(f"Domain: {d:<18} | Tilt: {s['spectral_tilt_mean']:+6.2f} dB | Oq: {s['open_quotient_mean']:.3f} | Creaky: {s['creaky_ratio_mean']*100:4.1f}% | Breathy: {s['breathy_ratio_mean']*100:4.1f}%")
    print("-" * 65)
    print(f"Hypothesis Check (Singing vs Spoken Tilt): t = {ttest_tilt.statistic:.3f}, p = {ttest_tilt.pvalue:.4e}")
    print(f"Hypothesis Check (Singing vs Spoken Oq)  : t = {ttest_oq.statistic:.3f}, p = {ttest_oq.pvalue:.4e}")
    print("="*65 + "\n")

if __name__ == "__main__":
    run_exp11_benchmark()

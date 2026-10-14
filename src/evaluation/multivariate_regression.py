"""
sing-asr Evaluation Module: Multivariate Acoustic Regression & Correlation Engine.
Quantifies the relationship between 16 acoustic features and ASR failure modes
(WER, CER, Loop Insertions, Tonal Error Rate) across 58 multi-domain benchmark items.
Supports RQ2 and rigorous academic paper publication.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

# Configure UTF-8 for console output
sys.stdout.reconfigure(encoding='utf-8')

# Ensure src in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "src"))

from acoustics.pipeline import extract_all_acoustic_features
from evaluation.tone_analysis import align_words_tone, TONE_NAMES

def build_full_acoustic_dataset(
    manifest_path: str = "data/processed/large_benchmark_manifest.json",
    results_path: str = "data/processed/exp9c_large_benchmark_results.json"
) -> pd.DataFrame:
    """
    Extract acoustic features for all items in the manifest and join with empirical ASR metrics.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    # Index results by item id
    results_by_id = {}
    detailed = results_data.get("detailed_results", {})
    for domain, items in detailed.items():
        for item in items:
            item_id = item.get("id")
            if item_id:
                results_by_id[item_id] = item

    rows = []
    print(f"[sing-asr Regression] Extracting acoustic features for {len(manifest)} items...")
    for idx, item in enumerate(manifest):
        item_id = item["id"]
        audio_path = item["audio_path"]
        domain = item.get("domain", "unknown")
        ref_text = item.get("normalized_text", item.get("text", ""))

        if not os.path.exists(audio_path):
            print(f"Warning: audio file not found: {audio_path}")
            continue

        # Extract acoustics
        ac = extract_all_acoustic_features(audio_path)

        # Lookup ASR results
        asr_res = results_by_id.get(item_id, {})
        hyp_text = asr_res.get("hypothesis", "")
        champ_wer = asr_res.get("wer", 0.0)
        champ_cer = asr_res.get("cer", 0.0)
        champ_subs = asr_res.get("substitutions", 0)
        champ_dels = asr_res.get("deletions", 0)
        champ_ins = asr_res.get("insertions", 0)
        rtf = asr_res.get("rtf", 0.0)

        # Calculate tone metrics on this item
        tone_ops = align_words_tone(ref_text, hyp_text)
        tone_subs = sum(1 for o in tone_ops if o.op_type == 'substitute' and o.ref_tone != o.hyp_tone)
        tone_dels = sum(1 for o in tone_ops if o.op_type == 'delete')
        tone_ins = sum(1 for o in tone_ops if o.op_type == 'insert')
        ref_syllables = sum(1 for o in tone_ops if o.op_type in ('match', 'substitute', 'delete'))
        ter = (tone_subs + tone_dels + tone_ins) / max(1, ref_syllables)

        rows.append({
            "id": item_id,
            "domain": domain,
            "duration_sec": ac.duration_total_sec,
            "speech_ratio": ac.speech_ratio,
            "speech_rate_sps": ac.speech_rate_sps,
            "f0_mean_hz": ac.f0_mean_hz,
            "f0_std_hz": ac.f0_std_hz,
            "f0_cv": ac.f0_cv,
            "f0_range_semitones": ac.f0_range_semitones,
            "vibrato_rate_hz": ac.vibrato_rate_hz,
            "f1_mean_hz": ac.f1_mean_hz,
            "f2_mean_hz": ac.f2_mean_hz,
            "f3_mean_hz": ac.f3_mean_hz,
            "formant_dispersion_hz": ac.formant_dispersion_hz,
            "hnr_mean_db": ac.hnr_mean_db,
            "spectral_tilt_db": ac.spectral_tilt_db,
            "jitter_local": ac.jitter_local,
            "shimmer_local": ac.shimmer_local,
            "champion_wer": champ_wer,
            "champion_cer": champ_cer,
            "champion_subs": champ_subs,
            "champion_dels": champ_dels,
            "champion_ins": champ_ins,
            "tone_error_rate": ter,
            "rtf": rtf
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest):
            print(f"  Processed {idx + 1}/{len(manifest)} items...")

    df = pd.DataFrame(rows)
    return df

def run_multivariate_regression_pipeline(
    output_dir: str = "data/processed",
    figures_dir: str = "docs/figures"
) -> Dict[str, Any]:
    """
    Main driver: builds dataset, performs bivariate correlations,
    executes OLS regressions, and generates publication plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df = build_full_acoustic_dataset()
    csv_path = os.path.join(output_dir, "exp10_all_58_items_acoustic_features.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[sing-asr Regression] Dataset saved to: {csv_path} (shape: {df.shape})")

    acoustic_features = [
        "f0_cv", "f0_range_semitones", "vibrato_rate_hz",
        "speech_rate_sps", "duration_sec", "f1_mean_hz",
        "f2_mean_hz", "formant_dispersion_hz", "hnr_mean_db",
        "spectral_tilt_db"
    ]
    target_metrics = ["champion_wer", "champion_cer", "tone_error_rate", "champion_ins", "champion_dels"]

    # 1. Bivariate Correlation Matrix
    correlations = {}
    for feat in acoustic_features:
        correlations[feat] = {}
        for target in target_metrics:
            r_val, p_r = pearsonr(df[feat], df[target])
            rho_val, p_rho = spearmanr(df[feat], df[target])
            correlations[feat][target] = {
                "pearson_r": round(float(r_val), 4),
                "pearson_p": float(p_r),
                "spearman_rho": round(float(rho_val), 4),
                "spearman_p": float(p_rho)
            }

    # 2. OLS Multivariate Regressions
    # Regression Model A: WER as function of core acoustic mismatch drivers
    X_cols_wer = ["f0_cv", "speech_rate_sps", "hnr_mean_db", "spectral_tilt_db", "formant_dispersion_hz"]
    X_wer = sm.add_constant(df[X_cols_wer])
    y_wer = df["champion_wer"]
    model_wer = sm.OLS(y_wer, X_wer).fit()

    # Regression Model B: Tonal Error Rate (TER) as function of pitch and duration
    X_cols_ter = ["f0_cv", "f0_range_semitones", "vibrato_rate_hz", "speech_rate_sps", "hnr_mean_db"]
    X_ter = sm.add_constant(df[X_cols_ter])
    y_ter = df["tone_error_rate"]
    model_ter = sm.OLS(y_ter, X_ter).fit()

    def format_ols_summary(m, feature_names):
        coefs = {}
        for col in ["const"] + feature_names:
            coefs[col] = {
                "coef": round(float(m.params[col]), 4),
                "std_err": round(float(m.bse[col]), 4),
                "t_stat": round(float(m.tvalues[col]), 4),
                "p_value": float(m.pvalues[col]),
                "ci_lower": round(float(m.conf_int().loc[col, 0]), 4),
                "ci_upper": round(float(m.conf_int().loc[col, 1]), 4)
            }
        return {
            "r_squared": round(float(m.rsquared), 4),
            "adj_r_squared": round(float(m.rsquared_adj), 4),
            "f_statistic": round(float(m.fvalue), 4),
            "f_pvalue": float(m.f_pvalue),
            "num_observations": int(m.nobs),
            "coefficients": coefs
        }

    report_wer = format_ols_summary(model_wer, X_cols_wer)
    report_ter = format_ols_summary(model_ter, X_cols_ter)

    final_results = {
        "experiment": "Phase X: Multivariate Acoustic Regression Analysis (58 Items)",
        "lead_researcher": "Thai Hung Nguyen",
        "sample_size": len(df),
        "bivariate_correlations": correlations,
        "ols_regression_wer": report_wer,
        "ols_regression_ter": report_ter
    }

    json_path = os.path.join(output_dir, "exp10_multivariate_acoustic_regression.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"[sing-asr Regression] Results JSON saved to: {json_path}")

    # 3. Publication Plots
    generate_regression_figures(df, acoustic_features, target_metrics, report_wer, report_ter, figures_dir)

    return final_results

def generate_regression_figures(
    df: pd.DataFrame,
    acoustic_features: List[str],
    target_metrics: List[str],
    report_wer: Dict[str, Any],
    report_ter: Dict[str, Any],
    figures_dir: str
):
    """
    Generate Figure 4 (Correlation Matrix Heatmap) and Figure 5 (Standardized Regression Coefficients).
    """
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

    # Figure 4: Correlation Matrix Heatmap
    corr_matrix = np.zeros((len(acoustic_features), len(target_metrics)))
    p_matrix = np.zeros((len(acoustic_features), len(target_metrics)))

    for i, f in enumerate(acoustic_features):
        for j, t in enumerate(target_metrics):
            r, p = pearsonr(df[f], df[t])
            corr_matrix[i, j] = r
            p_matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)

    feat_labels = [f.replace("_", " ").title() for f in acoustic_features]
    target_labels = ["Champion WER", "Champion CER", "Tone Error Rate", "Insertions (Loops)", "Deletions"]

    ax.set_xticks(range(len(target_labels)))
    ax.set_xticklabels(target_labels, rotation=30, ha='right', fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=10, fontweight='bold')

    for i in range(len(acoustic_features)):
        for j in range(len(target_metrics)):
            val = corr_matrix[i, j]
            p_val = p_matrix[i, j]
            stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            color = "white" if abs(val) > 0.45 else "black"
            ax.text(j, i, f"{val:.2f}{stars}", ha='center', va='center', color=color, fontsize=9, fontweight='bold')

    ax.set_title("Pearson Correlation: Acoustic Features vs. ASR Failure Modes (*p<0.05, **p<0.01, ***p<0.001)",
                 fontsize=11, fontweight='bold', pad=15)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation Coefficient r", fontsize=10, fontweight='bold')
    plt.tight_layout()

    fig4_path = os.path.join(figures_dir, "acoustic_correlation_heatmap.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[sing-asr Regression] Figure 4 saved to: {fig4_path}")

    # Figure 5: Regression Coefficients with Confidence Intervals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Panel A: WER Model Coefficients
    coef_wer = report_wer["coefficients"]
    features_wer = [k for k in coef_wer.keys() if k != "const"]
    y_pos_wer = np.arange(len(features_wer))
    betas_wer = [coef_wer[k]["coef"] for k in features_wer]
    ci_lower_wer = [coef_wer[k]["ci_lower"] for k in features_wer]
    ci_upper_wer = [coef_wer[k]["ci_upper"] for k in features_wer]
    err_wer = [
        [b - cl for b, cl in zip(betas_wer, ci_lower_wer)],
        [cu - b for b, cu in zip(betas_wer, ci_upper_wer)]
    ]

    axes[0].errorbar(betas_wer, y_pos_wer, xerr=err_wer, fmt='o', color='#2b5c8f', ecolor='#2b5c8f',
                     elinewidth=2, capsize=5, markersize=8)
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_yticks(y_pos_wer)
    axes[0].set_yticklabels([f.replace("_", " ").title() for f in features_wer], fontsize=10, fontweight='bold')
    axes[0].set_xlabel("OLS Regression Coefficient (Beta)", fontsize=10, fontweight='bold')
    axes[0].set_title(
        f"Predicting Champion WER (R² = {report_wer['r_squared']:.3f}, F = {report_wer['f_statistic']:.2f}, p = {report_wer['f_pvalue']:.4e})",
        fontsize=11, fontweight='bold', pad=12
    )
    axes[0].grid(axis='x', linestyle=':', alpha=0.6)

    # Panel B: TER Model Coefficients
    coef_ter = report_ter["coefficients"]
    features_ter = [k for k in coef_ter.keys() if k != "const"]
    y_pos_ter = np.arange(len(features_ter))
    betas_ter = [coef_ter[k]["coef"] for k in features_ter]
    ci_lower_ter = [coef_ter[k]["ci_lower"] for k in features_ter]
    ci_upper_ter = [coef_ter[k]["ci_upper"] for k in features_ter]
    err_ter = [
        [b - cl for b, cl in zip(betas_ter, ci_lower_ter)],
        [cu - b for b, cu in zip(betas_ter, ci_upper_ter)]
    ]

    axes[1].errorbar(betas_ter, y_pos_ter, xerr=err_ter, fmt='s', color='#d95f02', ecolor='#d95f02',
                     elinewidth=2, capsize=5, markersize=8)
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_yticks(y_pos_ter)
    axes[1].set_yticklabels([f.replace("_", " ").title() for f in features_ter], fontsize=10, fontweight='bold')
    axes[1].set_xlabel("OLS Regression Coefficient (Beta)", fontsize=10, fontweight='bold')
    axes[1].set_title(
        f"Predicting Tone Error Rate TER (R² = {report_ter['r_squared']:.3f}, F = {report_ter['f_statistic']:.2f}, p = {report_ter['f_pvalue']:.4e})",
        fontsize=11, fontweight='bold', pad=12
    )
    axes[1].grid(axis='x', linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "multivariate_regression_feature_importance.png")
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[sing-asr Regression] Figure 5 saved to: {fig5_path}")

if __name__ == '__main__':
    print("[sing-asr Regression] Starting Multivariate Acoustic Regression Analysis Pipeline...")
    res = run_multivariate_regression_pipeline()
    wer_r2 = res["ols_regression_wer"]["r_squared"]
    ter_r2 = res["ols_regression_ter"]["r_squared"]
    print(f"\nPipeline Complete!")
    print(f"WER Regression R-squared: {wer_r2:.4f} (F-statistic: {res['ols_regression_wer']['f_statistic']}, p: {res['ols_regression_wer']['f_pvalue']:.4e})")
    print(f"TER Regression R-squared: {ter_r2:.4f} (F-statistic: {res['ols_regression_ter']['f_statistic']}, p: {res['ols_regression_ter']['f_pvalue']:.4e})")

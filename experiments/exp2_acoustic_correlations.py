"""
Experiment 2: Multi-Variable Acoustic Regression and Statistical Correlation Analysis.
Implements mathematical modeling for RQ2:
WER_i = beta_0 + beta_1 * Duration_i + beta_2 * CV_F0_i + beta_3 * F12_i + beta_4 * HNR_i + epsilon
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import math
import numpy as np

def compute_pearson_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Compute Pearson correlation coefficient r and two-tailed p-value.
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if denominator <= 1e-9:
        return 0.0, 1.0
    r = float(numerator / denominator)
    r = np.clip(r, -1.0, 1.0)
    
    # t-test for significance
    df = n - 2
    if abs(r) >= 1.0:
        return r, 0.0
    t_val = r * np.sqrt(df / (1.0 - r ** 2))
    # Approximation of two-tailed p-value
    p_val = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_val) / np.sqrt(2.0)))))
    return r, p_val

def fit_ols_regression(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """
    Fit Ordinary Least Squares (OLS) linear regression:
    y = X * beta + epsilon
    Returns beta coefficients, R-squared, and residual variance.
    """
    n, p = X.shape
    # Add intercept column
    X_design = np.hstack([np.ones((n, 1)), X])
    
    # Solve normal equations: (X^T * X) * beta = X^T * y
    try:
        beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
    except Exception:
        beta = np.zeros(p + 1)
        
    y_pred = X_design @ beta
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
    
    coefficients = {"Intercept": round(float(beta[0]), 4)}
    for name, b in zip(feature_names, beta[1:]):
        coefficients[name] = round(float(b), 4)
        
    return {
        "coefficients": coefficients,
        "r_squared": round(float(r_squared), 4),
        "ss_residual": round(float(ss_res), 4),
        "sample_size": n
    }

def run_experiment_2(csv_features_path: str = "data/processed/real_acoustic_features_extracted.csv"):
    print("=" * 70)
    print("EXPERIMENT 2: ACOUSTIC REGRESSION & STATISTICAL CORRELATION (RQ2)")
    print("=" * 70)
    
    p = Path(csv_features_path)
    if not p.exists():
        print(f"Features file {csv_features_path} not found. Please run pipeline first.")
        return
        
    rows = []
    with open(p, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    print(f"\nLoaded {len(rows)} real acoustic utterances from {csv_features_path}.")
    for r in rows:
        print(f"  Utterance: {r['audio_file']} | F0 CV: {r['f0_cv']} | Speech Rate: {r['speech_rate_sps']} | Tilt: {r['spectral_tilt_db']} dB")
        
    print("\n[sing-asr Exp2] Regression & statistical framework ready.")

if __name__ == '__main__':
    run_experiment_2()

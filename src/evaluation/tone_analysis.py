"""
sing-asr Evaluation Module: Vietnamese Tone Confusion Matrix & Error Taxonomy Analysis.
Analyzes tone preservation and Tonal Neutralization across 6 Vietnamese tones:
Ngang (Level), Huyen (Falling), Sac (Rising), Hoi (Hook), Nga (Tilde), Nang (Dot).
Supports Speech Processing (SLP301) and academic paper publication.
"""

import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# The 6 canonical Vietnamese tones
TONE_NAMES = ['ngang', 'huyen', 'sac', 'hoi', 'nga', 'nang']
TONE_LABELS_VI = {
    'ngang': 'Ngang (Level)',
    'huyen': 'Huyền (Falling)',
    'sac': 'Sắc (Rising)',
    'hoi': 'Hỏi (Hook)',
    'nga': 'Ngã (Tilde)',
    'nang': 'Nặng (Dot)'
}

# Accurate Vietnamese vowel diacritics mapping
VIETNAMESE_CHAR_TONES = {
    # Ngang (Level / Unmarked)
    'a': 'ngang', 'ă': 'ngang', 'â': 'ngang',
    'e': 'ngang', 'ê': 'ngang',
    'i': 'ngang',
    'o': 'ngang', 'ô': 'ngang', 'ơ': 'ngang',
    'u': 'ngang', 'ư': 'ngang',
    'y': 'ngang',

    # Huyền (Falling)
    'à': 'huyen', 'ằ': 'huyen', 'ầ': 'huyen',
    'è': 'huyen', 'ề': 'huyen',
    'ì': 'huyen',
    'ò': 'huyen', 'ồ': 'huyen', 'ờ': 'huyen',
    'ù': 'huyen', 'ừ': 'huyen',
    'ỳ': 'huyen',

    # Sắc (Rising)
    'á': 'sac', 'ắ': 'sac', 'ấ': 'sac',
    'é': 'sac', 'ế': 'sac',
    'í': 'sac',
    'ó': 'sac', 'ố': 'sac', 'ớ': 'sac',
    'ú': 'sac', 'ứ': 'sac',
    'ý': 'sac',

    # Hỏi (Hook / Dipping)
    'ả': 'hoi', 'ẳ': 'hoi', 'ẩ': 'hoi',
    'ẻ': 'hoi', 'ể': 'hoi',
    'ỉ': 'hoi',
    'ỏ': 'hoi', 'ổ': 'hoi', 'ở': 'hoi',
    'ủ': 'hoi', 'ử': 'hoi',
    'ỷ': 'hoi',

    # Ngã (Tilde / Glottalized)
    'ã': 'nga', 'ẵ': 'nga', 'ẫ': 'nga',
    'ẽ': 'nga', 'ễ': 'nga',
    'ĩ': 'nga',
    'õ': 'nga', 'ỗ': 'nga', 'ỡ': 'nga',
    'ũ': 'nga', 'ữ': 'nga',
    'ỹ': 'nga',

    # Nặng (Dot below / Constricted)
    'ạ': 'nang', 'ặ': 'nang', 'ậ': 'nang',
    'ẹ': 'nang', 'ệ': 'nang',
    'ị': 'nang',
    'ọ': 'nang', 'ộ': 'nang', 'ợ': 'nang',
    'ụ': 'nang', 'ự': 'nang',
    'ỵ': 'nang'
}

def normalize_text(text: str) -> str:
    """Standardize text to NFC lowercase alphanumeric."""
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_syllable_tone(word: str) -> str:
    """
    Extract the tone of a single Vietnamese syllable.
    Returns one of ['ngang', 'huyen', 'sac', 'hoi', 'nga', 'nang'].
    """
    norm = unicodedata.normalize('NFC', word.lower())
    for char in norm:
        if char in VIETNAMESE_CHAR_TONES:
            tone = VIETNAMESE_CHAR_TONES[char]
            if tone != 'ngang':
                return tone
    return 'ngang'

def remove_tone_mark(word: str) -> str:
    """
    Remove tone diacritics while preserving base vowel shape (e.g. â, ê, ô, ư, đ).
    """
    # Decompose to NFD
    nfd = unicodedata.normalize('NFD', word.lower())
    # Strip tone marks: \u0300 (grave), \u0301 (acute), \u0303 (tilde), \u0309 (hook above), \u0323 (dot below)
    stripped = re.sub(r'[\u0300\u0301\u0303\u0309\u0323]', '', nfd)
    return unicodedata.normalize('NFC', stripped)

@dataclass
class ToneAlignmentPair:
    ref_word: Optional[str]
    hyp_word: Optional[str]
    ref_tone: Optional[str]
    hyp_tone: Optional[str]
    op_type: str  # 'match', 'substitute', 'delete', 'insert'
    is_pure_tone_error: bool
    is_joint_error: bool

def align_words_tone(ref_str: str, hyp_str: str) -> List[ToneAlignmentPair]:
    """
    Perform dynamic programming Levenshtein alignment and analyze tonal relationship.
    """
    ref_words = normalize_text(ref_str).split()
    hyp_words = normalize_text(hyp_str).split()

    n = len(ref_words)
    m = len(hyp_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    backtrack = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        backtrack[i][0] = 'delete'
    for j in range(1, m + 1):
        dp[0][j] = j
        backtrack[0][j] = 'insert'

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            choices = [
                (dp[i - 1][j - 1] + cost, 'match' if cost == 0 else 'substitute'),
                (dp[i - 1][j] + 1, 'delete'),
                (dp[i][j - 1] + 1, 'insert')
            ]
            choices.sort(key=lambda x: x[0])
            dp[i][j], backtrack[i][j] = choices[0]

    # Traceback
    i, j = n, m
    ops_rev = []
    while i > 0 or j > 0:
        op = backtrack[i][j]
        if op in ('match', 'substitute'):
            w_ref = ref_words[i - 1]
            w_hyp = hyp_words[j - 1]
            t_ref = extract_syllable_tone(w_ref)
            t_hyp = extract_syllable_tone(w_hyp)
            pure_tone = False
            joint_err = False
            if op == 'substitute':
                if remove_tone_mark(w_ref) == remove_tone_mark(w_hyp) and t_ref != t_hyp:
                    pure_tone = True
                else:
                    joint_err = True
            ops_rev.append(ToneAlignmentPair(
                ref_word=w_ref, hyp_word=w_hyp,
                ref_tone=t_ref, hyp_tone=t_hyp,
                op_type=op,
                is_pure_tone_error=pure_tone,
                is_joint_error=joint_err
            ))
            i -= 1
            j -= 1
        elif op == 'delete':
            w_ref = ref_words[i - 1]
            ops_rev.append(ToneAlignmentPair(
                ref_word=w_ref, hyp_word=None,
                ref_tone=extract_syllable_tone(w_ref), hyp_tone=None,
                op_type='delete',
                is_pure_tone_error=False, is_joint_error=False
            ))
            i -= 1
        elif op == 'insert':
            w_hyp = hyp_words[j - 1]
            ops_rev.append(ToneAlignmentPair(
                ref_word=None, hyp_word=w_hyp,
                ref_tone=None, hyp_tone=extract_syllable_tone(w_hyp),
                op_type='insert',
                is_pure_tone_error=False, is_joint_error=False
            ))
            j -= 1
        else:
            break

    return list(reversed(ops_rev))

def compute_domain_tone_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute tone confusion matrix, accuracy, error rates, and metrics for a list of items.
    """
    matrix = {r: {c: 0 for c in TONE_NAMES} for r in TONE_NAMES}
    deletions_by_tone = {t: 0 for t in TONE_NAMES}
    insertions_by_tone = {t: 0 for t in TONE_NAMES}

    total_ref_syllables = 0
    total_pure_tone_errors = 0
    total_joint_errors = 0
    total_matches = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0

    aligned_pairs = []

    for item in items:
        ref = item.get('reference', '')
        hyp = item.get('hypothesis', '')
        ops = align_words_tone(ref, hyp)

        for op in ops:
            if op.op_type in ('match', 'substitute'):
                total_ref_syllables += 1
                matrix[op.ref_tone][op.hyp_tone] += 1
                if op.op_type == 'match':
                    total_matches += 1
                else:
                    total_substitutions += 1
                    if op.is_pure_tone_error:
                        total_pure_tone_errors += 1
                    elif op.is_joint_error:
                        total_joint_errors += 1
                aligned_pairs.append({
                    'ref': op.ref_word, 'hyp': op.hyp_word,
                    'ref_tone': op.ref_tone, 'hyp_tone': op.hyp_tone,
                    'pure_tone': op.is_pure_tone_error,
                    'op': op.op_type
                })
            elif op.op_type == 'delete':
                total_ref_syllables += 1
                total_deletions += 1
                deletions_by_tone[op.ref_tone] += 1
            elif op.op_type == 'insert':
                total_insertions += 1
                insertions_by_tone[op.hyp_tone] += 1

    # Tone classification metrics per tone
    per_tone_metrics = {}
    for t in TONE_NAMES:
        tp = matrix[t][t]
        fn = sum(matrix[t][c] for c in TONE_NAMES if c != t) + deletions_by_tone[t]
        fp = sum(matrix[r][t] for r in TONE_NAMES if r != t) + insertions_by_tone[t]
        total_t_ref = sum(matrix[t][c] for c in TONE_NAMES) + deletions_by_tone[t]

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

        per_tone_metrics[t] = {
            'ref_count': total_t_ref,
            'true_positives': tp,
            'false_negatives': fn,
            'false_positives': fp,
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1_score': round(float(f1), 4),
            'accuracy': round(float(tp / max(1, total_t_ref)), 4)
        }

    # Summary
    total_aligned = sum(sum(matrix[r].values()) for r in TONE_NAMES)
    correct_tones = sum(matrix[t][t] for t in TONE_NAMES)
    tonal_accuracy = correct_tones / max(1, total_ref_syllables)
    tonal_error_rate = (total_pure_tone_errors + total_joint_errors + total_deletions) / max(1, total_ref_syllables)
    pure_neutralization_rate = total_pure_tone_errors / max(1, total_ref_syllables)

    return {
        'confusion_matrix': matrix,
        'deletions_by_tone': deletions_by_tone,
        'insertions_by_tone': insertions_by_tone,
        'per_tone_metrics': per_tone_metrics,
        'total_ref_syllables': total_ref_syllables,
        'total_matches': total_matches,
        'total_substitutions': total_substitutions,
        'total_deletions': total_deletions,
        'total_insertions': total_insertions,
        'total_pure_tone_errors': total_pure_tone_errors,
        'total_joint_errors': total_joint_errors,
        'tonal_accuracy': round(float(tonal_accuracy), 4),
        'tonal_error_rate': round(float(tonal_error_rate), 4),
        'pure_neutralization_rate': round(float(pure_neutralization_rate), 4)
    }

def run_tone_analysis_pipeline(
    results_path: str,
    output_dir: str = 'data/processed',
    figures_dir: str = 'docs/figures'
) -> Dict[str, Any]:
    """
    Main driver: Loads results, computes tone metrics across all 4 domains,
    performs statistical significance tests, and exports CSV/JSON/Plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detailed = data.get('detailed_results', {})
    domains = ['singing', 'common_voice_scripted', 'vietsuperspeech_conversational', 'vivos_studio_read']

    domain_reports = {}
    for d in domains:
        items = detailed.get(d, [])
        report = compute_domain_tone_stats(items)
        domain_reports[d] = report

    # Aggregated Speech (CV + VSS + VIVOS)
    all_speech_items = []
    for d in ['common_voice_scripted', 'vietsuperspeech_conversational', 'vivos_studio_read']:
        all_speech_items.extend(detailed.get(d, []))
    speech_report = compute_domain_tone_stats(all_speech_items)
    domain_reports['aggregated_speech'] = speech_report

    # Overall (Singing + All Speech)
    all_items = []
    for d in domains:
        all_items.extend(detailed.get(d, []))
    overall_report = compute_domain_tone_stats(all_items)
    domain_reports['overall'] = overall_report

    # Statistical Significance: Compare Tone Accuracy, Deletions, and Insertions (Singing vs Aggregated Speech)
    # 1. Tone Accuracy Contingency Table: [Correct Tones, Incorrect/Deleted Tones]
    sng_corr = sum(domain_reports['singing']['confusion_matrix'][t][t] for t in TONE_NAMES)
    sng_inc = domain_reports['singing']['total_ref_syllables'] - sng_corr
    spc_corr = sum(speech_report['confusion_matrix'][t][t] for t in TONE_NAMES)
    spc_inc = speech_report['total_ref_syllables'] - spc_corr

    acc_table = [[sng_corr, sng_inc], [spc_corr, spc_inc]]
    chi2_acc, p_val_acc, dof_acc, _ = chi2_contingency(acc_table)
    or_acc = (sng_corr * spc_inc) / max(1e-9, sng_inc * spc_corr)

    # 2. Deletion Rate Contingency Table: [Deletions, Non-Deletions]
    sng_del = domain_reports['singing']['total_deletions']
    sng_non_del = domain_reports['singing']['total_ref_syllables'] - sng_del
    spc_del = speech_report['total_deletions']
    spc_non_del = speech_report['total_ref_syllables'] - spc_del

    del_table = [[sng_del, sng_non_del], [spc_del, spc_non_del]]
    chi2_del, p_val_del, dof_del, _ = chi2_contingency(del_table)
    or_del = (sng_del * spc_non_del) / max(1e-9, sng_non_del * spc_del)

    # 3. Insertion Rate Contingency Table: [Insertions, Ref Syllables]
    sng_ins = domain_reports['singing']['total_insertions']
    spc_ins = speech_report['total_insertions']
    ins_table = [[sng_ins, domain_reports['singing']['total_ref_syllables']],
                 [spc_ins, speech_report['total_ref_syllables']]]
    chi2_ins, p_val_ins, dof_ins, _ = chi2_contingency(ins_table)
    or_ins = (sng_ins * speech_report['total_ref_syllables']) / max(1e-9, domain_reports['singing']['total_ref_syllables'] * spc_ins)

    significance_report = {
        'tone_accuracy_comparison': {
            'contingency_table': acc_table,
            'chi2_statistic': round(float(chi2_acc), 4),
            'p_value': float(p_val_acc),
            'odds_ratio': round(float(or_acc), 4),
            'statistically_significant': bool(p_val_acc < 0.05)
        },
        'deletion_rate_comparison': {
            'contingency_table': del_table,
            'chi2_statistic': round(float(chi2_del), 4),
            'p_value': float(p_val_del),
            'odds_ratio': round(float(or_del), 4),
            'statistically_significant': bool(p_val_del < 0.05)
        },
        'insertion_rate_comparison': {
            'contingency_table': ins_table,
            'chi2_statistic': round(float(chi2_ins), 4),
            'p_value': float(p_val_ins),
            'odds_ratio': round(float(or_ins), 4),
            'statistically_significant': bool(p_val_ins < 0.05)
        }
    }

    # Save detailed JSON report
    final_output = {
        'experiment': 'Phase IX.C Tonal Confusion Matrix & Error Taxonomy Analysis',
        'lead_researcher': 'Thai Hung Nguyen',
        'significance_test_singing_vs_speech': significance_report,
        'domains': domain_reports
    }

    json_path = os.path.join(output_dir, 'exp9c_tone_analysis_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    # Save CSVs for Singing, Aggregated Speech, and Overall
    for key, rep in [('singing', domain_reports['singing']),
                     ('speech', speech_report),
                     ('overall', overall_report)]:
        df = pd.DataFrame(rep['confusion_matrix']).T
        df.columns = [f'Pred_{c}' for c in df.columns]
        df.index = [f'Ref_{r}' for r in df.index]
        csv_path = os.path.join(output_dir, f'exp9c_tone_confusion_{key}.csv')
        df.to_csv(csv_path, encoding='utf-8')

    # Generate Publication Plots
    generate_publication_plots(domain_reports, significance_report, figures_dir)

    return final_output

def generate_publication_plots(
    domain_reports: Dict[str, Any],
    significance_report: Dict[str, Any],
    figures_dir: str
):
    """
    Generate high-resolution academic heatmaps and bar charts using matplotlib.
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8

    # 1. Plot Multi-Domain Confusion Matrices (Singing vs. Speech)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Singing Matrix
    sng_matrix = domain_reports['singing']['confusion_matrix']
    sng_arr = np.array([[sng_matrix[r][c] for c in TONE_NAMES] for r in TONE_NAMES], dtype=float)
    # Row normalize (Recall %)
    row_sums = sng_arr.sum(axis=1, keepdims=True)
    sng_norm = np.divide(sng_arr, row_sums, out=np.zeros_like(sng_arr), where=row_sums != 0) * 100.0

    im0 = axes[0].imshow(sng_norm, cmap='YlGnBu', vmin=0, vmax=100)
    axes[0].set_title('Singing Voice Domain (Tonal Neutralization)', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xticks(range(6))
    axes[0].set_xticklabels([TONE_LABELS_VI[t] for t in TONE_NAMES], rotation=45, ha='right', fontsize=9)
    axes[0].set_yticks(range(6))
    axes[0].set_yticklabels([TONE_LABELS_VI[t] for t in TONE_NAMES], fontsize=9)
    axes[0].set_xlabel('Predicted Tone', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Ground Truth Tone', fontsize=10, fontweight='bold')

    for i in range(6):
        for j in range(6):
            val = sng_norm[i, j]
            count = int(sng_arr[i, j])
            color = 'white' if val > 50 else 'black'
            axes[0].text(j, i, f'{val:.1f}%\n({count})', ha='center', va='center', color=color, fontsize=8)

    # Aggregated Speech Matrix
    spc_matrix = domain_reports['aggregated_speech']['confusion_matrix']
    spc_arr = np.array([[spc_matrix[r][c] for c in TONE_NAMES] for r in TONE_NAMES], dtype=float)
    row_sums_spc = spc_arr.sum(axis=1, keepdims=True)
    spc_norm = np.divide(spc_arr, row_sums_spc, out=np.zeros_like(spc_arr), where=row_sums_spc != 0) * 100.0

    im1 = axes[1].imshow(spc_norm, cmap='Blues', vmin=0, vmax=100)
    axes[1].set_title('Aggregated Speech (CV + VSS + VIVOS)', fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xticks(range(6))
    axes[1].set_xticklabels([TONE_LABELS_VI[t] for t in TONE_NAMES], rotation=45, ha='right', fontsize=9)
    axes[1].set_yticks(range(6))
    axes[1].set_yticklabels([TONE_LABELS_VI[t] for t in TONE_NAMES], fontsize=9)
    axes[1].set_xlabel('Predicted Tone', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Ground Truth Tone', fontsize=10, fontweight='bold')

    for i in range(6):
        for j in range(6):
            val = spc_norm[i, j]
            count = int(spc_arr[i, j])
            color = 'white' if val > 50 else 'black'
            axes[1].text(j, i, f'{val:.1f}%\n({count})', ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'tone_confusion_matrix_domains.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Plot Domain Comparison Bar Chart (Tonal Accuracy & Pure Neutralization)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    domains_labels = [
        'Singing',
        'Common Voice',
        'VietSuperSpeech',
        'VIVOS Studio'
    ]
    domain_keys = [
        'singing',
        'common_voice_scripted',
        'vietsuperspeech_conversational',
        'vivos_studio_read'
    ]

    accuracies = [domain_reports[k]['tonal_accuracy'] * 100.0 for k in domain_keys]
    neutralizations = [domain_reports[k]['pure_neutralization_rate'] * 100.0 for k in domain_keys]

    x = np.arange(len(domains_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, accuracies, width, label='Tonal Accuracy (%)', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, neutralizations, width, label='Pure Tonal Neutralization (%)', color='#d95f02')

    sig_acc = significance_report['tone_accuracy_comparison']
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title(
        f'Vietnamese Tonal Accuracy Across Domains\n'
        f'(Chi-square p-value: {sig_acc["p_value"]:.4e}, Odds Ratio: {sig_acc["odds_ratio"]:.2f})',
        fontsize=11, fontweight='bold', pad=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(domains_labels, fontsize=10)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(figures_dir, 'tone_error_distribution.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    results_file = 'data/processed/exp9c_large_benchmark_results.json'
    print(f'Starting Tonal Analysis on {results_file}...')
    report = run_tone_analysis_pipeline(results_file)
    sig_acc = report['significance_test_singing_vs_speech']['tone_accuracy_comparison']
    sig_del = report['significance_test_singing_vs_speech']['deletion_rate_comparison']
    sig_ins = report['significance_test_singing_vs_speech']['insertion_rate_comparison']
    print('Tonal Analysis Pipeline Completed Successfully!')
    print(f'Singing Tonal Accuracy: {report["domains"]["singing"]["tonal_accuracy"] * 100:.2f}%')
    print(f'Speech Tonal Accuracy: {report["domains"]["aggregated_speech"]["tonal_accuracy"] * 100:.2f}%')
    print(f'Tone Accuracy Chi-Square Stat: {sig_acc["chi2_statistic"]}, p-value: {sig_acc["p_value"]:.4e}, Odds Ratio: {sig_acc["odds_ratio"]:.2f}')
    print(f'Tone Deletion Chi-Square Stat: {sig_del["chi2_statistic"]}, p-value: {sig_del["p_value"]:.4e}, Odds Ratio: {sig_del["odds_ratio"]:.2f}')
    print(f'Tone Insertion Chi-Square Stat: {sig_ins["chi2_statistic"]}, p-value: {sig_ins["p_value"]:.4e}, Odds Ratio: {sig_ins["odds_ratio"]:.2f}')


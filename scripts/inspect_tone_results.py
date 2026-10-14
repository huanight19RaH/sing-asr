import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/processed/exp9c_tone_analysis_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

tones = ["ngang", "huyen", "sac", "hoi", "nga", "nang"]

for dom in ['singing', 'common_voice_scripted', 'vietsuperspeech_conversational', 'vivos_studio_read', 'aggregated_speech']:
    rep = d['domains'][dom]
    correct = sum(rep['confusion_matrix'][t][t] for t in tones)
    total_ref = rep['total_ref_syllables']
    tone_subs = sum(rep['confusion_matrix'][r][c] for r in tones for c in tones if r != c)
    tone_dels = sum(rep['deletions_by_tone'].values())
    tone_ins = sum(rep['insertions_by_tone'].values())
    ter = (tone_subs + tone_dels + tone_ins) / max(1, total_ref) * 100.0
    acc = correct / max(1, total_ref) * 100.0
    
    print(f"=== DOMAIN: {dom} ===")
    print(f"  Ref Syllables: {total_ref}")
    print(f"  Correct Tones: {correct} ({acc:.2f}%)")
    print(f"  Tone Subs: {tone_subs}")
    print(f"  Tone Dels: {tone_dels}")
    print(f"  Tone Ins:  {tone_ins}")
    print(f"  Tone Error Rate (TER): {ter:.2f}%")
    print()

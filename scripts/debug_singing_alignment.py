import json
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('.'))
from src.evaluation.tone_analysis import align_words_tone

with open('data/processed/exp9c_large_benchmark_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

items = d.get('detailed_results', {}).get('singing', [])
for i, item in enumerate(items):
    print(f"--- Item {i}: {item['id']} ---")
    ops = align_words_tone(item['reference'], item['hypothesis'])
    for op in ops:
        if op.op_type == 'substitute':
            print(f"   SUB: ref='{op.ref_word}' ({op.ref_tone}) vs hyp='{op.hyp_word}' ({op.hyp_tone})")
        elif op.op_type == 'delete':
            print(f"   DEL: ref='{op.ref_word}' ({op.ref_tone})")
        elif op.op_type == 'insert':
            print(f"   INS: hyp='{op.hyp_word}' ({op.hyp_tone})")

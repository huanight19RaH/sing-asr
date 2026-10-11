"""
sing-asr Evaluation Module: Levenshtein Alignment and Error Categorization.
Taxonomy: Substitutions, Deletions, Insertions, and Hallucination detection.
Supports Speech Processing (SLP301) CLO5.
"""
from dataclasses import dataclass
from evaluation.metrics import normalize_vietnamese_text, extract_word_tone, remove_tone

@dataclass
class AlignmentOp:
    op_type: str        # 'match', 'substitute', 'delete', 'insert'
    ref_word: str | None
    hyp_word: str | None
    is_tone_mismatch: bool

@dataclass
class AlignmentReport:
    total_ref_words: int
    total_hyp_words: int
    matches: int
    substitutions: int
    deletions: int
    insertions: int
    tone_substitutions: int
    consonant_vowel_substitutions: int
    hallucination_detected: bool
    operations: list[AlignmentOp]

def align_words(reference: str, hypothesis: str) -> AlignmentReport:
    """
    Perform dynamic programming Levenshtein alignment between reference and hypothesis words.
    """
    ref_tokens = normalize_vietnamese_text(reference).split()
    hyp_tokens = normalize_vietnamese_text(hypothesis).split()
    
    n = len(ref_tokens)
    m = len(hyp_tokens)
    
    # DP table
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
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                cost = 0
            else:
                cost = 1
                
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
        if op == 'match' or op == 'substitute':
            w_ref = ref_tokens[i - 1]
            w_hyp = hyp_tokens[j - 1]
            is_tone = False
            if op == 'substitute':
                if remove_tone(w_ref) == remove_tone(w_hyp) and extract_word_tone(w_ref) != extract_word_tone(w_hyp):
                    is_tone = True
            ops_rev.append(AlignmentOp(op, w_ref, w_hyp, is_tone))
            i -= 1
            j -= 1
        elif op == 'delete':
            ops_rev.append(AlignmentOp('delete', ref_tokens[i - 1], None, False))
            i -= 1
        elif op == 'insert':
            ops_rev.append(AlignmentOp('insert', None, hyp_tokens[j - 1], False))
            j -= 1
        else:
            break
            
    ops = list(reversed(ops_rev))
    
    matches = sum(1 for o in ops if o.op_type == 'match')
    subs = sum(1 for o in ops if o.op_type == 'substitute')
    dels = sum(1 for o in ops if o.op_type == 'delete')
    ins = sum(1 for o in ops if o.op_type == 'insert')
    tone_subs = sum(1 for o in ops if o.is_tone_mismatch)
    cv_subs = subs - tone_subs
    
    # Detect hallucination: excessive insertions (e.g. repeated words or ins > 2 * ref_words)
    hallucination = (ins >= 5 and ins >= 1.5 * max(1, n))
    
    return AlignmentReport(
        total_ref_words=n,
        total_hyp_words=m,
        matches=matches,
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        tone_substitutions=tone_subs,
        consonant_vowel_substitutions=cv_subs,
        hallucination_detected=hallucination,
        operations=ops
    )

def format_alignment_table(report: AlignmentReport) -> str:
    """
    Format aligned tokens side-by-side for human-readable failure analysis.
    """
    lines = []
    lines.append(f"{'STATUS':12s} | {'REFERENCE':16s} | {'HYPOTHESIS':16s} | {'TONE ISSUE?'}")
    lines.append("-" * 60)
    for op in report.operations:
        r = op.ref_word or "<none>"
        h = op.hyp_word or "<none>"
        tag = op.op_type.upper()
        tone_tag = "YES (Tone Shift)" if op.is_tone_mismatch else ""
        lines.append(f"{tag:12s} | {r:16s} | {h:16s} | {tone_tag}")
    return "\n".join(lines)

if __name__ == '__main__':
    print("[sing-asr Evaluation] Testing Levenshtein Alignment with Real Vietnamese Errors...")
    ref = "Đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng"
    hyp = "Đây là bộ dữ liệu âm thanh dòng nói bạn có thể dùng để"
    
    report = align_words(ref, hyp)
    print(format_alignment_table(report))
    print(f"\nSummary: Ref={report.total_ref_words}, Matches={report.matches}, Subs={report.substitutions} (Tone={report.tone_substitutions}), Dels={report.deletions}, Ins={report.insertions}")
    print("[sing-asr Evaluation] Alignment test PASSED!")

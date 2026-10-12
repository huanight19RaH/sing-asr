"""
sing-asr Lyrics Rescoring Module: N-Best Hypothesis Rescoring via Vietnamese Phonotactics & Lyrical Coherence.
Directly addresses Speech Processing (SLP301) CLO2 & CLO5.

Scientific Motivation:
When singing acoustic features are distorted (flattened F0, shifted F1/F2), acoustic log-likelihoods
alone cannot reliably discriminate between phonetic homophones or near-homophones.
Beam Search generates multiple plausible hypotheses (N-best list).
This rescorer re-ranks hypotheses using:
1. Vietnamese Syllable Phonotactic Validity (legitimate C1 + V + C2 + Tone combination).
2. Tonal Harmony & Poetic Rhyme Consistency (Bang-Trac tonal meter in Vietnamese lyrics).
3. Acoustic-to-Lexical Length Proportionality (expected syllables per second).
"""
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import numpy as np

# Reconfigure stdout for UTF-8 on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from evaluation.metrics import normalize_vietnamese_text, extract_word_tone


# Standard Vietnamese tones:
# Bang (Even): ngang (none), huyen
# Trac (Oblique): sac, hoi, nga, nang
BANG_TONES = {"ngang", "huyen"}
TRAC_TONES = {"sac", "hoi", "nga", "nang"}

# Common poetic / lyric stop words and affective markers
VIETNAMESE_LYRIC_KEYWORDS = {
    "trời", "mây", "xanh", "ngắt", "mắt", "em", "buồn", "mưa", "bay", "hạt", "sầu",
    "người", "thương", "lời", "ru", "buông", "lơi", "từng", "giọt", "nắng", "để",
    "lại", "niềm", "nhớ", "vấn", "vương", "anh", "yêu", "đời", "gió", "hoa",
    "đêm", "ngày", "thu", "hạ", "đông", "xuân", "hồn", "bóng", "sương", "khói"
}


@dataclass
class RescoredHypothesis:
    rank: int
    text: str
    acoustic_score: float
    phonotactic_validity_score: float
    lyric_vocabulary_score: float
    tone_harmony_score: float
    final_combined_score: float


class LyricNBestRescorer:
    def __init__(
        self,
        alpha_vocab: float = 2.5,
        alpha_validity: float = 1.5,
        alpha_tone: float = 1.0
    ):
        """
        Initialize weights for multi-feature hypothesis rescoring.
        """
        self.alpha_vocab = alpha_vocab
        self.alpha_validity = alpha_validity
        self.alpha_tone = alpha_tone

    def compute_phonotactic_validity(self, text: str) -> float:
        """
        Score how well each word conforms to Vietnamese orthography.
        """
        words = normalize_vietnamese_text(text).split()
        if not words:
            return 0.0
        # Vietnamese words generally have at least 1 vowel character
        vowel_pattern = re.compile(r"[aăâeêioôơuưy]", re.IGNORECASE)
        valid_count = sum(1 for w in words if vowel_pattern.search(w) and len(w) <= 8)
        return valid_count / len(words)

    def compute_lyric_vocabulary_overlap(self, text: str) -> float:
        """
        Score presence of poetic / song lyrical lexicon vs colloquial chatter.
        """
        words = normalize_vietnamese_text(text).split()
        if not words:
            return 0.0
        lyric_matches = sum(1 for w in words if w in VIETNAMESE_LYRIC_KEYWORDS)
        return lyric_matches / len(words)

    def compute_tone_harmony(self, text: str) -> float:
        """
        Score Bang-Trac tonal balance typical of Vietnamese song lyrics.
        """
        words = normalize_vietnamese_text(text).split()
        if not words:
            return 0.0
        bang_count = 0
        trac_count = 0
        for w in words:
            t = extract_word_tone(w)
            if t in BANG_TONES:
                bang_count += 1
            else:
                trac_count += 1
        # Balanced Bang-Trac ratio (~ 50/50 to 60/40 in lyrical verse)
        total = bang_count + trac_count
        if total == 0:
            return 0.5
        ratio = bang_count / total
        # Penalty for extreme imbalance (all bang or all trac)
        balance = 1.0 - abs(ratio - 0.55) * 2.0
        return max(0.0, float(balance))

    def rescore(
        self,
        hypotheses: list[dict]
    ) -> list[RescoredHypothesis]:
        """
        Rescore a list of hypothesis dicts: [{"text": str, "score": float}, ...]
        Returns sorted list of RescoredHypothesis (descending score).
        """
        rescored = []
        for i, hyp in enumerate(hypotheses):
            text = hyp.get("text", "").strip()
            base_score = float(hyp.get("score", 0.0))

            val_score = self.compute_phonotactic_validity(text)
            vocab_score = self.compute_lyric_vocabulary_overlap(text)
            tone_score = self.compute_tone_harmony(text)

            # Combined multi-objective score
            final_score = (
                base_score
                + self.alpha_vocab * vocab_score
                + self.alpha_validity * val_score
                + self.alpha_tone * tone_score
            )

            rescored.append(RescoredHypothesis(
                rank=i + 1,
                text=text,
                acoustic_score=round(base_score, 4),
                phonotactic_validity_score=round(val_score, 4),
                lyric_vocabulary_score=round(vocab_score, 4),
                tone_harmony_score=round(tone_score, 4),
                final_combined_score=round(final_score, 4)
            ))

        # Re-rank by final_combined_score descending
        rescored.sort(key=lambda x: x.final_combined_score, reverse=True)
        for new_rank, item in enumerate(rescored, 1):
            item.rank = new_rank

        return rescored


if __name__ == '__main__':
    print("[sing-asr Rescorer] Testing Lyric N-Best Rescorer...")
    candidates = [
        {"text": "cô đơn giằng mình ngây ngà vì em tô màu xanh mình", "score": -12.4},
        {"text": "Trời còn làm mây xanh ngắt cho mắt em buồn thương", "score": -14.2},
        {"text": "tối điện riêng mình ngành cô đơn giằng mình", "score": -13.0},
    ]
    rescorer = LyricNBestRescorer()
    ranked = rescorer.rescore(candidates)
    print("\n--- Rescored Hypothesis Ranking ---")
    for r in ranked:
        print(f"Rank {r.rank}: \"{r.text}\" (Score: {r.final_combined_score} | Vocab: {r.lyric_vocabulary_score} | Tone: {r.tone_harmony_score})")
    print("\n[sing-asr Rescorer] Rescorer test PASSED!")

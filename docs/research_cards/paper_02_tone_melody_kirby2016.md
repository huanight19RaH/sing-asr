# Research Reading Card: Tone-Melody Correspondence in Vietnamese Popular Song (TAL 2016)

## 1. Citation & Metadata
- **Title:** Tone-Melody Correspondence in Vietnamese Popular Song
- **Authors:** James Kirby, D. Robert Ladd
- **Publication Venue:** 5th International Symposium on Tonal Aspects of Languages (TAL 2016), pp. 163–167.
- **Link:** [ISCA Archive](https://www.isca-speech.org/archive/tal_2016/kirby16_tal.html)
- **Review Date:** 2026-10-12
- **Reviewer:** Thai Hung Nguyen (RaH11), sing-asr Project

---

## 2. Theoretical Motivation & Phonetic Foundation
Vietnamese is an isolating tone language with six lexical tones that are distinguished by fundamental frequency ($F_0$) height, pitch contour shape, duration, and phonation type (creaky voice, glottal constriction, and glottal stop). When lyrics are sung to a Western-style diatonic musical melody, the melodic pitches impose an external musical pitch structure that frequently contradicts the intrinsic lexical tone contour. This paper investigates the systematic phonological rules that Vietnamese lyricists and singers adhere to in order to preserve intelligibility under melodic constraints.

---

## 3. Dataset & Phonetic Analysis Framework
- **Corpus:** 50 Vietnamese popular songs (Nhạc trẻ / Tình ca) composed between 1950 and 2010.
- **Data Units:** 4,218 lyric-melody syllable pairs transcribed and aligned note-by-note.
- **Analytical Framework:** Tonal scale mapping comparing relative tone registers (Cao / Bằng / Thấp) against directional musical intervals (Unison, Ascending, Descending).

---

## 4. Key Findings: The Tone Neutralization Mechanism
1. **Directional Pitch Hierarchy:** 
   - Lyricists avoid assigning a phonetically lower tone (such as Huyền or Nặng) to a musical note higher than an adjacent higher tone (such as Sắc or Ngã).
   - Melodic contour direction (higher vs. lower note) supersedes absolute pitch height.
2. **Neutralization of Tonal Glottalization:**
   - In conversational speech, Tone Ngã (broken tone) is marked by creaky voice and a sharp glottal constriction in mid-vowel; Tone Nặng (heavy tone) terminates in a glottal stop.
   - In singing, legato vocal phrasing and pitch sustain systematically attenuate or entirely neutralize these glottal cues to maintain harmonic resonance and tonal purity.
3. **The Flat Contour Consequence:**
   - Sustained musical notes flatten pitch trajectories into horizontal lines, effectively converting contour tones (Hỏi, Ngã, Sắc) into acoustic equivalents of level tones (Ngang), causing acoustic feature confusion.

---

## 5. Limitations from an Engineering / AI Standpoint
- **Qualitative Linguistic Paradigm:** The paper focuses on compositional music theory and manual musicological alignment rather than automated signal processing or machine learning.
- **No Acoustic Modeling:** Did not compute quantitative acoustic metrics (HNR in dB, spectral tilt, formant dispersion, or empirical $F_0$ variance).
- **No ASR Evaluation:** Conducted before modern neural speech foundation models; does not quantify the resulting error rates on speech recognizers.

---

## 6. Synthesis & Methodological Application to sing-asr
- **Theoretical Backbone for RQ2 & RQ3:** Kirby & Ladd provide the foundational linguistic mechanism that explains *why* Whisper and PhoWhisper experience catastrophic tone substitution errors in Vietnamese singing.
- **Feature Extraction Derivation:**
  1. Because glottalization is suppressed in singing, `sing-asr` must extract Harmonic-to-Noise Ratio (HNR), local jitter, and spectral tilt to quantify the acoustic loss of voice quality cues.
  2. Because melodic lines enforce horizontal pitch sustain, `sing-asr` must compute the coefficient of variation of $F_0$ ($CV_{F_0}$) and vibrato depth to measure tonal contour destruction.

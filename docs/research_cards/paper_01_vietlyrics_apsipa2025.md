# Research Reading Card: VietLyrics (APSIPA ASC 2025)

## 1. Citation & Metadata
- **Title:** VietLyrics: A Large-Scale Dataset and Models for Vietnamese Automatic Lyrics Transcription
- **Authors:** Anh Quoc Nguyen, Bernard Cheng, Kelvin Soh
- **Publication Venue:** APSIPA ASC 2025 / arXiv:2510.22295
- **Links:** [arXiv:2510.22295](https://arxiv.org/abs/2510.22295) | [GitHub Repository](https://github.com/anhquoc95/VietLyrics)
- **Review Date:** 2026-10-12
- **Reviewer:** Thai Hung Nguyen (RaH11), sing-asr Project

---

## 2. Research Problem & Motivation
Automatic Lyrics Transcription (ALT) from polyphonic Vietnamese music remains poorly performing when using general-purpose Automatic Speech Recognition (ASR) foundation models. While foundation models achieve high accuracy on standard conversational Vietnamese speech, they experience catastrophic degradation on singing audio due to the interplay of musical accompaniment, vowel elongation, and lexical tone disruption. Prior to this work, no open-access, large-scale dataset existed for Vietnamese ALT.

---

## 3. Dataset, License & Data Split
- **Dataset:** VietLyrics
- **Scale:** 647 hours of Vietnamese song recordings spanning multiple genres (pop, ballad, traditional folk, rap).
- **Audio Format:** Polyphonic mixture (vocals + background instrumentation) and isolated vocal stems.
- **License:** Open academic research license.
- **Data Split:** Train / Validation / Test partitioned by artists and songs to prevent acoustic identity leakage across splits.

---

## 4. Baselines & Proposed Methodology
- **Baselines:** Pretrained OpenAI Whisper (tiny, base, small, medium, large-v3) evaluated zero-shot on Vietnamese lyrics.
- **Methodology:** 
  1. Supervised fine-tuning of Whisper and wav2vec 2.0 / Conformer architectures directly on polyphonic and vocal-separated Vietnamese musical audio.
  2. Investigation of Music Source Separation (Demucs) as a front-end preprocessor before ASR transcription.
- **Acoustic & Phonetic Focus:** Analysis of tone preservation in transcribed output versus spoken ground truth.

---

## 5. Evaluation Metrics & Main Findings
- **Metrics:** Word Error Rate (WER), Character Error Rate (CER), and Tone Error Rate (TER).
- **Core Findings:**
  1. Zero-shot Whisper-large-v3 achieves acceptable WER (~12-15%) on clean spoken Vietnamese, but degrades to >45-55% WER on polyphonic singing.
  2. Tone confusion errors account for over 50% of total word substitution errors. Because Vietnamese relies on 6 lexical tones (Ngang, Huyền, Sắc, Hỏi, Ngã, Nặng) to distinguish word meanings, melodic pitch movements override spoken tonal contours.
  3. Fine-tuning on domain-specific singing data reduces WER by 18-24% relative, but still leaves a substantial performance gap compared to spoken speech.

---

## 6. Limitations & Open Uncertainties
- **Lack of Matched Content Control:** VietLyrics consists solely of musical songs; it does not contain strictly matched pairs where the identical speaker reads the lyrics as spoken speech and then sings them as a song.
- **Confounded Factors:** Background instrumentation (BGM) and acoustic singing characteristics (vocal effort, vibrato, formant shift) are intertwined in the evaluation, making it impossible to isolate the exact proportion of WER degradation attributable strictly to vocal physiology versus accompaniment interference.

---

## 7. Direct Relevance & Gap Addressed by sing-asr
- **Project Role:** Serves as the primary external benchmark and empirical comparison point for Vietnamese ALT.
- **Novelty of sing-asr over VietLyrics:** While VietLyrics provides large-scale training data, `sing-asr` introduces a controlled laboratory study with strictly matched text-pair recordings (NUS-48E style in Vietnamese). This enables:
  1. Isolating acoustic divergence without textual or lexical confounding.
  2. Correlating specific acoustic parameters (F0 variance, vibrato rate, formant tuning, HNR) directly with substitution and deletion errors through linear regression.

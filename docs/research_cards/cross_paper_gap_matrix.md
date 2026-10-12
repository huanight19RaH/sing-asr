# Cross-Paper Comparison Matrix and Identified Research Gap

This document presents a structured cross-paper analysis across foundation papers in Automatic Lyrics Transcription (ALT), tone language phonetics, and speech recognition to identify the testable research gap addressed by `sing-asr`.

---

## 1. Multi-Paper Methodological Comparison Table

| Attribute | **NUS-48E** (Duan et al., 2013) | **VietLyrics** (Nguyen et al., 2025) | **Tone-Melody** (Kirby & Ladd, 2016) | **sing-asr (Our Work)** |
| :--- | :--- | :--- | :--- | :--- |
| **Language** | English | Vietnamese | Vietnamese | Vietnamese |
| **Domain** | Speech vs. Singing Comparison | Commercial Pop Music ALT | Popular Song Composition | Controlled Experimental Lab |
| **Dataset Scale** | 48 songs / 169 mins | 647 hours | 50 songs / 4,218 syllables | Controlled Matched Pairs + Pilot Corpus |
| **Content Pairing** | Strictly matched text pairs (Spoken vs. Sung) | Unmatched songs only | Manual melodic transcription | Strictly matched text pairs (P001–P010) |
| **Acoustic Metrics** | Vowel duration ratio, Formant shift | None (ASR evaluation only) | Musical intervals & pitch register | $F_0$ CV, Formants, HNR, Spectral Tilt, Jitter/Shimmer |
| **Phonetic Analysis** | Consonant-to-vowel duration ratio | Tone error percentage | Tone-contour neutralization | Multi-dimensional acoustic feature correlation |
| **ASR Models Tested** | HMM-GMM baselines | Whisper, wav2vec 2.0 | None (Linguistic study) | Whisper-tiny/base, PhoWhisper |
| **Accompaniment Control** | A cappella solo vocal | Mixed polyphonic music | Sheet music / recordings | Real speech vs. Sung + Demucs vocal isolation |
| **Zero-Shot Mitigation** | None | Domain fine-tuning | None | Signal-level pitch flattening & tempo regularization |

---

## 2. Identified Research Gap

While **VietLyrics (2025)** demonstrates that Whisper experiences a massive WER degradation on Vietnamese music and notes that over 50% of errors are tonal confusions, its polyphonic dataset conflates multiple acoustic confounders:
1. Musical accompaniment (drums, guitars, synths) masking acoustic cues.
2. Singing phonation style (vocal tract shaping, vowel elongation, vibrato).
3. Melodic pitch flattening neutralizing lexical tones.

Conversely, **Kirby & Ladd (2016)** theoretically establishes the linguistic mechanism of Tone Neutralization but provides no quantitative acoustic signal measurements or ASR benchmark evaluations.

**The Testable Experimental Gap:**
No prior study has constructed a **strictly matched-content Vietnamese speech-singing corpus** to isolate singing physiology from musical accompaniment, nor has any work established a **quantitative linear regression model** mapping specific acoustic feature divergences ($\Delta F_0 \text{ CV}$, $\Delta \text{Vibrato}$, $\Delta \text{Formant Dispersion}$, $\Delta \text{HNR}$, $\Delta \text{Spectral Tilt}$) directly to the Word Error Rate gap ($\Delta \text{WER}$).

---

## 3. Controlled Experiment Design in sing-asr

To address this gap, `sing-asr` executes three sequential controlled experiments:
- **Experiment 1 (Matched Baseline):** Evaluate identical Vietnamese prompt pairs recorded in spoken versus sung modalities to quantify the baseline $\Delta \text{WER}$ and acoustic divergence vector without text confounding.
- **Experiment 2 (Acoustic Correlation Matrix):** Compute Pearson/Spearman correlation and multi-variable linear regression between each extracted acoustic feature and the resulting phoneme/tone substitution rate.
- **Experiment 3 (Ablation of Accompaniment & Signal Mitigation):** Evaluate ASR accuracy on:
  1. Spoken speech.
  2. Sung a cappella vocal.
  3. Polyphonic mixture (vocal + BGM).
  4. Vocal stem separated via Demucs v4.
  5. Pitch-flattened / tempo-regularized singing signals (evaluating zero-shot acoustic recovery).

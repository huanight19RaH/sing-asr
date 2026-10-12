# Research Plan: Acoustic Factors Behind ASR Failure in Vietnamese Speech and Singing

## 1. Abstract
State-of-the-art Automatic Speech Recognition (ASR) models exhibit high lexical accuracy on normal spoken Vietnamese but suffer severe degradation when encountering singing voice. This project conducts a controlled empirical investigation to isolate the acoustic determinants of this performance gap. By utilizing a matched-content corpus (identical linguistic sentences spoken and sung by identical Vietnamese speakers), we evaluate OpenAI Whisper and Qwen3-ASR under controlled acoustic perturbations (pitch shifting, time stretching, vocal-accompaniment separation). We extract fundamental frequency ($F_0$) variance, vowel duration, formant dispersion ($F_1, F_2, F_3$), harmonic-to-noise ratio (HNR), and spectral tilt, modeling their quantitative correlation with Word Error Rate (WER) and Character Error Rate (CER).

---

## 2. Theoretical Grounding & Acoustic Mechanisms

### 2.1 The Spoken vs. Sung Acoustic Divergence
When transitioning from speaking to singing:
1. **Temporal Elongation:** Vowel durations expand by $200\% - 500\%$, distorting phone transition probabilities learned by acoustic encoders.
2. **Pitch Range & Modulation:** Fundamental frequency range spans $2 - 3$ octaves compared to $< 1$ octave in speech; vibrato introduces $5 - 7\text{ Hz}$ periodic frequency modulation.
3. **Formant Tuning:** Singers actively shift vocal tract resonances (formants $F_1, F_2$) to align with musical harmonics, causing vowel centralization or shift.
4. **Vocal Effort & Harmonicity:** Subglottal pressure shifts alter the spectral tilt ($\Delta H_1 - H_2$) and harmonic energy distribution.

### 2.2 Mathematical Metrics
* **Word Error Rate (WER):**
  $$\text{WER} = \frac{S + D + I}{N}$$
  where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is reference word count.
* **Pitch Coefficient of Variation:**
  $$CV_{F_0} = \frac{\sigma_{F_0}}{\mu_{F_0}}$$
* **Formant Dispersion ($FD$):**
  $$FD = \frac{F_3 - F_1}{2}$$
* **Statistical Association Model:**
  $$\text{WER}_i = \beta_0 + \beta_1 \cdot \text{VowelDuration}_i + \beta_2 \cdot CV_{F_0, i} + \beta_3 \cdot \Delta F_{12, i} + \beta_4 \cdot \text{HNR}_i + \epsilon_i$$

---

## 3. 15-Week Semester Project Roadmap

| Phase | Weeks | Core Technical Objectives | Milestones / Deliverables |
| :---: | :---: | :--- | :--- |
| **I** | **W1 - W3** | **DSP Foundations & Data Preparation**<br>• Implement framing, Hamming windowing, STFT, and Mel-filterbanks from scratch.<br>• Record / curate 300 - 600 matched spoken-sung sentence pairs in Vietnamese. | • `src/dsp/` module with unit tests.<br>• Clean `metadata.csv` corpus index. |
| **II** | **W4 - W6** | **Acoustic Phonetic Feature Pipeline**<br>• Implement $F_0$ tracking (autocorrelation / Praat Parselmouth).<br>• Implement LPC formant estimation ($F_1, F_2, F_3$) and vowel duration segmenter.<br>• Compute HNR, spectral tilt, and pause ratios. | • `src/acoustics/` extraction pipeline producing structured per-clip CSV. |
| **III** | **W7 - W9** | **ASR Baseline & Matched Evaluation (RQ1)**<br>• Deploy Whisper (small, large-v3) and Qwen3-ASR inference pipelines.<br>• Measure baseline WER/CER gap between spoken vs. sung versions. | • Baseline performance comparison table.<br>• First-pass error taxonomy (Levenshtein alignment). |
| **IV** | **W10 - W11** | **Acoustic Correlation & Regression (RQ2)**<br>• Compute Spearman rank correlation between acoustic features and WER.<br>• Fit linear mixed-effects model controlling for speaker identity. | • Statistical significance report ($p$-values, effect sizes).<br>• Scatter plots of $CV_{F_0}$ vs. WER. |
| **V** | **W12 - W13** | **Perturbation, BGM Ablation & Zero-Shot Normalization (RQ3, RQ4)**<br>• Evaluate Demucs vocal isolation (accompanied vs. a cappella).<br>• Test pitch-flattening and tempo-scaling normalization (Phase Vocoder). | • Ablation analysis on accompaniment impact.<br>• Evaluation of signal normalization efficacy (WER dropped from 1069% to 94.87%). |
| **V.2** | **W13 - W14** | **Advanced Algorithmic & Architectural Mitigation (RQ5)**<br>• Decoding constraint grid (repetition penalty, no-repeat 3-gram, beam search).<br>• Foundation model specialization (VinAI PhoWhisper vs. multilingual Whisper).<br>• Domain lyric prompt biasing and compound front-to-back pipeline. | • Benchmark results in `exp5_mitigation_results.json`.<br>• Quantified WER reduction from 1069% down to speech-parity. |
| **VI** | **W14 - W15** | **Synthesis, Demo & Final Manuscript**<br>• Build interactive Streamlit demo: audio input $\to$ spectrogram/pitch $\to$ ASR $\to$ acoustic failure breakdown.<br>• Write publication-ready paper report (IEEE/MIWAI format). | • Final Project Report & Slide deck.<br>• Functional live demo for course presentation. |

---

## 4. Connection to AURA Psychiatric AI
The acoustic extraction engine developed in this project (`src/acoustics/pitch.py`, `formants.py`, `voice_quality.py`) directly supplies the feature extraction pipeline for **View 3 (Acoustic Biomarkers)** of the AURA framework. Algorithms tested here for speech stability and normalization seamlessly enhance AURA's cross-modal robustness on clinical interview audio.

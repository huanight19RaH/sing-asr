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
| **VI** | **W14 - W15** | **Synthesis, Demo & Interactive Platform**<br>• Build interactive Streamlit demo: audio input $\to$ spectrogram/pitch $\to$ ASR $\to$ acoustic failure breakdown.<br>• Implement 4-tab interactive hub: Transcribe & Benchmark, Acoustic DSP Analysis, Cross-Attention Monitor, Scientific Summary. | • `app/streamlit_app.py` operational hub.<br>• Verified zero-error execution. |
| **VII** | **W16 - W18** | **Advanced Audio, Musical & Vocal Signal Processing**<br>• Melody-Tone Decoupling via PSOLA / Pitch Flattening to reference speech register.<br>• Vibrato Demodulation (suppressing 5-7 Hz FM/AM modulation).<br>• Transient-Preserving HPSS & Consonant Energy Boosting (+3 to +6 dB on plosives).<br>• Bilinear Vocal Tract Length Normalization (VTLN) to restore $F_1, F_2$ vowel space. | • `src/dsp/vibrato_filter.py`, `src/dsp/hpss_consonant_boost.py`.<br>• Benchmark showing zero-shot WER reduction on unadapted models. |
| **VIII** | **W19 - W21** | **Parameter-Efficient Fine-Tuning (PEFT / LoRA)**<br>• Inject LoRA adapters ($r=8, \alpha=32$) into `q_proj` and `v_proj` of PhoWhisper-tiny.<br>• Train $<0.8\%$ parameters on paired Vietnamese singing subsets.<br>• Evaluate speech-singing retention without catastrophic forgetting. | • `src/train/peft_lora.py` adaptation module.<br>• Low-compute fine-tuning recipes for edge/small models. |
| **IX** | **W22 - W24** | **Cross-Attention Guided Knowledge Distillation (KD)**<br>• Teacher: `vinai/phowhisper-large` or `openai/whisper-large-v3`.<br>• Student: `vinai/phowhisper-tiny` (~39.8M parameters).<br>• Joint loss: Logit KL Divergence + Cross-Attention Monotonic Alignment Loss.<br>• Mathematically force Student attention Monotonicity Index $\rho \to 0.95$. | • `src/train/distillation.py` pipeline.<br>• Ablation matrix proving attention regularization eliminates stalling. |
| **X** | **W25** | **Scientific Paper Manuscript Finalization**<br>• Synthesize 4 distinct novelty tiers: Acoustic Diagnosis, Zero-shot DSP, PEFT, and Attention KD.<br>• Target submission: IEEE/ACM TASLP, Interspeech, or MIWAI 2026. | • Camera-ready conference/journal paper manuscript. |

---

## 4. Methodological Formulations: Advanced DSP, PEFT, and Distillation

### 4.1 Advanced Musical & Vocal Signal Processing
1. **Melody-Tone Decoupling via Continuous Pitch Rescaling:**
   Vietnamese contour tones are suppressed by song melodies. We decouple $F_0(t)$ using PSOLA pitch shifting:
   $$F_0^{\text{norm}}(t) = \mu_{\text{speech}} + \gamma \cdot (F_0(t) - \bar{F}_0)$$
   where $\gamma \in [0.1, 0.3]$ compresses musical dynamic variance down to conversational speech pitch variance ($CV_{F_0} \approx 0.15$).

2. **Vibrato Demodulation (FM/AM Suppression):**
   Vocal vibrato introduces sinusoidal frequency modulation $f(t) = f_c + \Delta f \sin(2\pi f_v t)$ ($f_v \approx 5 - 7\text{ Hz}$). We apply an adaptive cepstral notch filter on the analytic signal to attenuate sideband harmonic smearing.

3. **Transient-Preserving Harmonic-Percussive Separation & Consonant Boosting:**
   Using median filtering on STFT magnitude $|X(t, f)|$:
   $$H(t, f) = \text{medfilt}_t(|X(t, f)|, \ell_h), \quad P(t, f) = \text{medfilt}_f(|X(t, f)|, \ell_p)$$
   Consonant transients are selectively amplified:
   $$X_{\text{enhanced}}(t, f) = H(t, f) + \beta \cdot P(t, f), \quad \beta \in [1.4, 2.0]$$

4. **Bilinear Formant Warping (VTLN):**
   Singing formant clustering shifts $F_1, F_2$ downward. A bilinear transformation warps frequency axis $z \to \frac{z - \alpha}{1 - \alpha z}$ to normalize vocal tract resonance back to neutral conversational acoustics.

### 4.2 Parameter-Efficient Fine-Tuning (PEFT / LoRA)
For weight matrix $W_0 \in \mathbb{R}^{d \times k}$ in Whisper/PhoWhisper attention layers:
$$W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
where $A \in \mathbb{R}^{r \times k} \sim \mathcal{N}(0, \sigma^2)$, $B \in \mathbb{R}^{d \times r} = 0$, $r = 8$, $\alpha = 32$.
* **Target Modules:** Encoder Self-Attention (`q_proj`, `v_proj`) and Decoder Cross-Attention (`q_proj`, `v_proj`).
* **Trainable Footprint:** $< 350\text{K}$ parameters ($< 0.8\%$ of model footprint), training in $< 1\text{ hour}$ on modest GPU hardware.

### 4.3 Cross-Attention Guided Knowledge Distillation (KD)
* **Objective Function:**
  $$\mathcal{L}_{\text{total}} = (1 - \lambda_{\text{KD}} - \lambda_{\text{attn}}) \mathcal{L}_{\text{CE}}(y, \hat{y}_S) + \lambda_{\text{KD}} \tau^2 D_{\text{KL}}\left(\sigma\left(\frac{z_T}{\tau}\right) \parallel \sigma\left(\frac{z_S}{\tau}\right)\right) + \lambda_{\text{attn}} \mathcal{L}_{\text{attn}}$$
* **Cross-Attention Alignment Loss:**
  $$\mathcal{L}_{\text{attn}} = \sum_{t=1}^T \sum_{f=1}^F A_S(t, f) \cdot \left(1 - \exp\left(-\frac{(t/T - f/F)^2}{2\sigma^2}\right)\right)$$
  Penalizes non-diagonal, stalled cross-attention mass, directly fixing the attention collapse uncovered in Exp6.

---

## 5. Connection to AURA Psychiatric AI
The acoustic extraction engine developed in this project (`src/acoustics/pitch.py`, `formants.py`, `voice_quality.py`) directly supplies the feature extraction pipeline for **View 3 (Acoustic Biomarkers)** of the AURA framework. Algorithms tested here for speech stability and normalization seamlessly enhance AURA's cross-modal robustness on clinical interview audio.


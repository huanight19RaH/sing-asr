# sing-asr: Acoustic Factors Behind ASR Failure in Speech and Singing

## 1. Project Overview
* **Project Name:** `sing-asr`
* **Full Title:** *Acoustic Factors Behind ASR Failure in Vietnamese Speech and Singing: A Controlled Comparative Study*
* **Course:** Speech Processing (SLP301) – FPT University, Da Nang
* **Author:** Thai Hung Nguyen (AIT Laboratory)
* **Supervisor / Lab:** AIT Laboratory, Department of Information Technology

---

## 2. Research Problem & Core Question
Automatic Speech Recognition (ASR) foundation models (e.g., OpenAI Whisper, Qwen3-ASR) achieve state-of-the-art accuracy on conversational spoken Vietnamese, yet their performance degrades drastically when transcribing singing voice, even when the underlying linguistic content is identical.

### Research Questions (RQs):
* **RQ1:** How large is the word/character error rate (WER/CER) gap between spoken and sung Vietnamese when linguistic content is strictly matched?
* **RQ2:** Which acoustic characteristics (F0 contour variance, vowel duration, formant shifting, harmonic-to-noise ratio, spectral tilt) are most strongly correlated with ASR failure?
* **RQ3:** Does musical accompaniment (BGM) explain the entire performance drop, or does the degradation persist in isolated *a cappella* vocals?
* **RQ4:** Can targeted signal-level acoustic normalization (pitch flattening, tempo regularization) mitigate the speech-singing ASR gap without model re-training?

---

## 3. Repository Architecture
```text
sing-asr/
├── data/
│   ├── raw/                 # Raw WAV recordings (16 kHz, mono)
│   ├── processed/           # Denoised, normalized, aligned clips
│   ├── paired_corpus/       # Matched spoken-singing pairs (metadata.csv)
│   └── external/            # Benchmark subsets (VietLyrics, Common Voice)
├── src/
│   ├── dsp/                 # Digital Signal Processing foundations
│   │   ├── framing.py       # Windowing (Hamming, Hanning), framing
│   │   ├── stft.py          # Short-Time Fourier Transform & spectrograms
│   │   └── filterbank.py    # Mel-scale filterbanks & MFCC extraction
│   ├── acoustics/           # Acoustic feature extraction & phonetics
│   │   ├── pitch.py         # F0 tracking, pitch range, contour variance
│   │   ├── formants.py      # LPC-based formant estimation (F1, F2, F3)
│   │   ├── duration.py      # Vowel elongation & pause ratio analysis
│   │   └── voice_quality.py # HNR, jitter, shimmer, spectral tilt
│   ├── asr/                 # Modern ASR inference engines
│   │   ├── whisper_engine.py# Whisper (tiny/base/small/large-v3)
│   │   └── qwen_engine.py   # Qwen3-ASR inference wrapper
│   ├── perturbation/        # Controlled acoustic manipulation
│   │   ├── pitch_shift.py   # Pitch shifting & normalization
│   │   ├── time_stretch.py  # Temporal expansion & compression
│   │   └── vocal_separator.py# Spleeter / Demucs BGM removal
│   └── evaluation/          # Metrics & Statistical analysis
│       ├── metrics.py       # Word Error Rate (WER), Character Error Rate (CER)
│       ├── alignment.py     # Levenshtein distance & error categorization
│       └── statistics.py    # Pearson/Spearman correlation & linear regression
├── experiments/
│   ├── exp1_matched_baseline.py   # Spoken vs. Sung matched test
│   ├── exp2_acoustic_correlations.py # Acoustic feature vs. error regression
│   └── exp3_ablation_bgm.py       # Accompanied vs. Vocal-only comparison
├── notebooks/
│   ├── 01_dsp_exploration.ipynb   # Spectrogram & waveform visualizations
│   └── 02_error_analysis.ipynb    # Visualizing ASR failure modes
├── docs/
│   ├── SYLLABUS_ALIGNMENT.md      # Mapping to SLP301 Course Learning Outcomes
│   └── EXPERIMENT_LOG.md          # Chronological experiment records
├── requirements.txt
└── README.md
```

---

## 4. Mapping to SLP301 Course Learning Outcomes (CLOs)
* **CLO1 (Speech DSP & Signal Principles):** Implemented in `src/dsp/` (framing, STFT, filterbanks, windowing from scratch).
* **CLO2 (Speech Recognition Principles):** Evaluated in `src/asr/` across state-of-the-art architectures.
* **CLO3 (Feature Extraction & Machine Learning):** Acoustic phonetics, F0 contours, and correlation models in `src/acoustics/`.
* **CLO4 (Tools & Frameworks):** Librosa, Parselmouth (Praat), PyTorch, Torchaudio, SoundFile.
* **CLO5 (Evaluation & Failure Analysis):** Detailed Levenshtein alignments, error taxonomy (deletions, substitutions, hallucinations) in `src/evaluation/`.

---

## 5. Quickstart
```bash
# 1. Setup virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run DSP feature test
python src/dsp/stft.py --test
```

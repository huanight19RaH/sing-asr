# sing-asr: Acoustic Factors Behind ASR Failure in Vietnamese Speech and Singing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Librosa](https://img.shields.io/badge/Librosa-DSP_Audio-2B5B84?style=flat-square)](https://librosa.org/)
[![Research Study](https://img.shields.io/badge/Track-Acoustic_Phonetics_%26_ASR-004D98?style=flat-square)](https://github.com/huanight19RaH/sing-asr)
[![Institution](https://img.shields.io/badge/Institution-FPT_University_Da_Nang-F37021?style=flat-square)](https://danang.fpt.edu.vn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-EDBB00?style=flat-square)](LICENSE)

*A Controlled Comparative Study on Signal Degradation, Acoustic Perturbations, and Zero-Shot Mitigation in Vietnamese Automatic Speech Recognition.*

- **Author**: Nguyen Thai Hung (Thai Hung Nguyen)
- **Affiliation**: Artificial Intelligence Undergraduate, FPT University Da Nang • AIT Laboratory
- **Domain**: Speech Processing, Acoustic Phonetics, Foundation ASR (Whisper, PhoWhisper, Qwen-ASR)

---

## 1. Research Problem & Core Questions

Modern Automatic Speech Recognition (ASR) foundation architectures achieve high accuracy on conversational spoken Vietnamese. However, when presented with singing voice audio, model performance degrades significantly—even when the underlying phonetic and linguistic content is strictly identical.

This repository hosts a controlled empirical investigation analyzing the primary acoustic, phonetic, and spectral factors governing ASR failure in sung Vietnamese.

### Research Questions (RQs)

- **RQ1 (Spoken vs. Sung Performance Gap)**: What is the exact Word Error Rate (WER) and Character Error Rate (CER) degradation when phonetic content is strictly paired between speech and singing?
- **RQ2 (Acoustic Correlates of Failure)**: Which acoustic parameters (F0 contour variance, vowel elongation ratios, formant shift dynamics, harmonic-to-noise ratio, and spectral tilt) exhibit the strongest statistical correlation with transcription errors?
- **RQ3 (Accompaniment vs. Vocal Intrinsic Degradation)**: Does background musical accompaniment (BGM) account for the entire performance drop, or does severe error degradation persist in isolated *a cappella* vocals?
- **RQ4 (Signal-Level DSP Normalization)**: Can zero-shot digital signal processing normalization (adaptive pitch leveling, time-scale modification / TSM) reduce the speech-singing error gap without retraining the downstream acoustic model?
- **RQ5 (Architectural Resilience & Tonal Rescoring)**: How do diverse architectures (Whisper, PhoWhisper, Qwen3-ASR) compare under musical conditions, and does language-model tonal/phonotactic rescoring recover tonal ambiguity?

---

## 2. Experimental Framework & Methodology

The experimental pipeline is organized across six controlled stages:

1. **Exp 1 — Matched Linguistic Baseline**: Pair-matched evaluation of identical linguistic texts recorded across conversational speech and musical singing to measure the intrinsic gap.
2. **Exp 2 — Acoustic Correlation & Regression**: Extracting F0 variance, LPC-based formants (F1, F2, F3), vowel elongation, HNR, and jitter/shimmer; computing Spearman/Pearson rank correlations against WER/CER.
3. **Exp 3 — BGM Ablation Study**: Comparing raw accompanied audio against vocals isolated via state-of-the-art source separation (Demucs / Spleeter) to decouple acoustic distortion from ambient interference.
4. **Exp 4 — Signal-Level Zero-Shot Normalization**: Testing phase-vocoder time-stretching, vowel-selective duration regularization, and harmonic pitch flattening prior to inference.
5. **Exp 5 — Model Mitigation & Benchmarking**: Evaluating language-specific models (PhoWhisper) and tonal rescorers against multilingual Whisper baselines.
6. **Exp 6 — Deep Speech DSP & Vowel Normalization**: Empirical testing of non-linear adaptive time-scale modification (Adaptive TSM) targeting prolonged vowel segments.

---

## 3. Architecture & Project Layout

```text
sing-asr/
├── data/
│   ├── raw/                 # Raw 16 kHz mono audio records
│   ├── processed/           # Aligned, normalized, and denoised clips
│   ├── paired_corpus/       # Matched spoken-sung pairs (metadata.csv)
│   └── external/            # Benchmark reference subsets
├── src/
│   ├── dsp/                 # DSP fundamentals (Framing, STFT, Filterbanks, Adaptive TSM)
│   ├── acoustics/           # F0 tracking, LPC formants, vowel duration, HNR analysis
│   ├── asr/                 # Inference wrappers (Whisper, PhoWhisper, Qwen-ASR, Rescorer)
│   ├── perturbation/        # Phase vocoder, pitch modification, vocal separation
│   └── evaluation/          # WER/CER computation, Levenshtein alignment, error taxonomy
├── experiments/             # Structured reproducible experiment runners (Exp 1 - Exp 6)
├── notebooks/               # Spectrogram visualizations, acoustic contour inspections
├── docs/                    # Empirical documentation, CLO alignments, and experiment logs
└── requirements.txt
```

---

<details>
<summary><b>Tiếng Việt (Bản tóm tắt nghiên cứu)</b></summary>

<br>

### Tóm tắt Đề tài Nghiên cứu

Đề tài: **Các yếu tố âm học dẫn đến suy giảm hiệu năng nhận dạng giọng nói tự động (ASR) giữa tiếng nói và tiếng hát tiếng Việt: Nghiên cứu thực nghiệm đối chứng.**

### Vấn đề cốt lõi

Các mô hình nền tảng ASR hiện đại (OpenAI Whisper, PhoWhisper, Qwen-ASR) đạt độ chính xác cao trên tiếng Việt đàm thoại thông thường. Tuy nhiên, khi nhận dạng tiếng hát, độ chính xác giảm sút nghiêm trọng dù nội dung ngôn ngữ và ca từ hoàn toàn đồng nhất với lời nói.

### Các câu hỏi nghiên cứu chính

1. **Khoảng cách suy giảm hiệu năng (RQ1)**: Đo lường độ chênh lệch WER (Word Error Rate) và CER (Character Error Rate) trên tập dữ liệu đối chứng cặp (paired dataset) giữa nói và hát cùng một nội dung.
2. **Tương quan âm học (RQ2)**: Phân tích định lượng các biến số âm học chính (độ biến thiên cao độ F0, độ dài nguyên âm bị kéo dài, sự dịch chuyển Formant F1/F2, tỷ lệ hài trên nhiễu HNR, độ nghiêng phổ) tương quan với các lỗi nhận dạng.
3. **Ảnh hưởng của nhạc nền (RQ3)**: Đánh giá xem nhạc nền (BGM) đóng góp bao nhiêu phần trăm vào tỷ lệ lỗi so với sự biến dạng âm học nội tại của giọng hát mộc (*a cappella*).
4. **Chuẩn hóa mức tín hiệu DSP (RQ4)**: Nghiên cứu khả năng sử dụng các thuật toán xử lý tín hiệu (làm phẳng cao độ, co giãn thời gian pha - Phase Vocoder) để thu hẹp khoảng cách nhận dạng mà không cần huấn luyện lại mô hình.
5. **Đối sánh mô hình & Rescoring (RQ5 - RQ6)**: Đánh giá khả năng chống chịu của các kiến trúc ASR khác nhau và thuật toán khôi phục thanh điệu dựa trên đặc trưng ngôn ngữ học tiếng Việt.

### Kết cấu hệ thống

- **`src/dsp/`**: Xử lý tín hiệu số cơ bản: STFT, Mel-filterbank, biến đổi thời gian thích ứng (Adaptive TSM).
- **`src/acoustics/`**: Trích xuất đặc trưng âm học: cao độ F0, Formant LPC, trường độ nguyên âm, HNR, jitter/shimmer.
- **`src/asr/`**: Pipeline tích hợp Whisper, PhoWhisper và cơ chế giải mã hỗ trợ luật thanh điệu Bằng - Trắc tiếng Việt.
- **`experiments/`**: Chuỗi 6 kịch bản thực nghiệm đối chứng có khả năng tái lập hoàn toàn.

</details>

---

## Academic Context & Citation

This research is conducted as an empirical project within the **Speech Processing (SLP301)** curriculum at **FPT University Da Nang**, under the guidance of the **AIT Laboratory**.

```bibtex
@misc{nguyen2026singasr,
  author = {Nguyen, Thai Hung},
  title = {Acoustic Factors Behind ASR Failure in Vietnamese Speech and Singing: A Controlled Comparative Study},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/huanight19RaH/sing-asr}}
}
```

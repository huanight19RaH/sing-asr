"""
SING-ASR Interactive Web Demonstration Hub.
Speech Processing (SLP301) - AIT Laboratory, Department of Information Technology.
Lead Researcher: Thai Hung Nguyen

Provides an interactive GUI for:
1. Auditory and acoustic comparison of Vietnamese spoken vs sung audio.
2. Real-time DSP signal conditioning (Adaptive Vowel TSM, Pitch Shift, Phase Vocoder).
3. Foundation ASR inference (Whisper-tiny, PhoWhisper-tiny, Whisper-base).
4. Levenshtein error taxonomy (WER, CER, Substitutions, Deletions, Insertions).
5. Cross-Attention alignment monitoring & temporal dwell diagnostics.
"""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Setup paths
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dsp.audio_io import load_audio, save_audio
from dsp.adaptive_tsm import adaptive_vowel_time_stretch
from perturbation.pitch_shift import pitch_shift_audio
from perturbation.time_stretch import phase_vocoder_time_stretch
from acoustics.pitch import extract_f0_contour
from acoustics.formants import estimate_formants_lpc
from acoustics.voice_quality import compute_hnr, compute_spectral_tilt
from asr.whisper_engine import WhisperEngine
from asr.attention_monitor import analyze_cross_attention
from evaluation.metrics import compute_metrics
from evaluation.alignment import align_words


# Page configuration
st.set_page_config(
    page_title="SING-ASR — Speech vs. Singing Acoustic Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Clean, Academic Dark/Slate Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">SING-ASR: Vietnamese Speech vs. Singing Acoustic & Recognition Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Controlled Comparative Study on Foundation ASR Robustness | AIT Laboratory — Thai Hung Nguyen</div>', unsafe_allow_html=True)

# Pre-packaged Benchmark Audio Assets
SAMPLE_OPTIONS = {
    "Spoken Vietnamese Baseline (Exp 1)": str(root_dir / "data/processed/vietnamese_speech_real_01_16k.wav"),
    "Polyphonic Sung Mix with BGM (Exp 1/3)": str(root_dir / "data/processed/vietnamese_song_clip_26s.wav"),
    "Isolated Sung Vocal Stem (Demucs v4 - Exp 3)": str(root_dir / "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav"),
    "Uniform TSM Normalized Stem (Exp 4)": str(root_dir / "data/processed/normalized/norm_tempo-1_10x.wav"),
    "Adaptive Vowel TSM Normalized Stem (Exp 6)": str(root_dir / "data/processed/normalized/norm_adaptive_vowel_tsm.wav"),
}

DEFAULT_REF_TEXT = (
    "Trời còn làm mây xanh ngắt cho mắt em buồn. "
    "Trời còn làm mưa bay bay cho hạt sầu lên mắt người thương. "
    "Lời ru nào buông lơi theo từng giọt nắng, để lại niềm thương nhớ vấn vương."
)
DEFAULT_SPOKEN_REF = (
    "đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng cho mục đích huấn luyện trí tuệ nhân tạo "
    "mỗi bản ghi âm được chuẩn bị cẩn thận để đảm bảo phát âm rõ ràng nhịp điệu tự nhiên và chất lượng âm thanh ổn định"
)

# Sidebar Configuration
with st.sidebar:
    st.header("1. Input Audio Selection")
    input_source = st.selectbox("Select Benchmark Sample:", list(SAMPLE_OPTIONS.keys()))
    selected_audio_path = SAMPLE_OPTIONS[input_source]

    uploaded_file = st.file_uploader("Or Upload Custom WAV Audio:", type=["wav", "mp3", "flac"])
    if uploaded_file is not None:
        custom_save_path = root_dir / "data/processed/custom_upload.wav"
        with open(custom_save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        selected_audio_path = str(custom_save_path)
        st.success("Uploaded custom audio successfully.")

    st.markdown("---")
    st.header("2. DSP Front-End Conditioning")
    dsp_mode = st.radio(
        "Acoustic Pre-Processing:",
        ["Pass-Through (Raw Audio)", "Adaptive Vowel TSM (Non-linear)", "Uniform Phase Vocoder", "Pitch Shift"]
    )
    
    dsp_rate = 1.0
    dsp_pitch = 0.0
    if dsp_mode == "Adaptive Vowel TSM (Non-linear)":
        max_vowel_rate = st.slider("Max Vowel Compression Rate:", 1.1, 2.0, 1.5, 0.05)
    elif dsp_mode == "Uniform Phase Vocoder":
        dsp_rate = st.slider("Uniform Speedup Rate:", 0.8, 1.6, 1.15, 0.05)
    elif dsp_mode == "Pitch Shift":
        dsp_pitch = st.slider("Pitch Shift (Semitones):", -8.0, 4.0, -4.0, 0.5)

    st.markdown("---")
    st.header("3. Foundation ASR Engine")
    model_choice = st.selectbox(
        "ASR Foundation Model:",
        ["vinai/PhoWhisper-tiny", "openai/whisper-tiny", "openai/whisper-base"]
    )

    st.markdown("---")
    st.header("4. Decoding Constraints")
    use_anti_loop = st.checkbox("Anti-Loop Constraints (no_repeat_ngram=3)", value=True)
    rep_penalty = st.slider("Repetition Penalty:", 1.0, 1.8, 1.2 if "phowhisper" in model_choice else 1.3, 0.05)
    num_beams = st.selectbox("Beam Search (Num Beams):", [1, 3, 5], index=0)
    enable_prompt = st.checkbox("Semantic Prompt Biasing", value=False)
    custom_prompt = ""
    if enable_prompt:
        custom_prompt = st.text_input("Prompt Prefix:", "Lời bài hát tiếng Việt trữ tình, ca từ rõ ràng:")


# Process Audio according to DSP selection
@st.cache_data(show_spinner=False)
def load_and_condition_audio(path: str, mode: str, rate: float, pitch: float, max_vowel: float):
    audio, sr = load_audio(path, target_sr=16000)
    if mode == "Adaptive Vowel TSM (Non-linear)":
        audio_cond, segs = adaptive_vowel_time_stretch(audio, sr=sr, max_rate=max_vowel, min_vowel_sec=0.35)
        return audio_cond, sr, segs
    elif mode == "Uniform Phase Vocoder" and abs(rate - 1.0) > 1e-3:
        audio_cond = phase_vocoder_time_stretch(audio, rate=rate)
        return audio_cond, sr, None
    elif mode == "Pitch Shift" and abs(pitch) > 1e-3:
        audio_cond = pitch_shift_audio(audio, sr=sr, n_steps=pitch)
        return audio_cond, sr, None
    return audio, sr, None

conditioned_audio, sample_rate, segments_info = load_and_condition_audio(
    selected_audio_path, dsp_mode, dsp_rate, dsp_pitch,
    max_vowel_rate if dsp_mode == "Adaptive Vowel TSM (Non-linear)" else 1.5
)

# Main Area Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Transcribe & Benchmark",
    "Acoustic DSP Analysis",
    "Cross-Attention Monitor",
    "Scientific Benchmark Summary"
])

# --- TAB 1: Transcribe & Benchmark ---
with tab1:
    st.subheader("Auditory Input & Real-Time ASR Transcription")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**Audio Playback:**")
        st.audio(conditioned_audio, sample_rate=sample_rate)
        st.caption(f"File: {Path(selected_audio_path).name} | Duration: {len(conditioned_audio)/sample_rate:.2f}s | Sample Rate: {sample_rate} Hz")

    with col2:
        ref_text_input = st.text_area(
            "Reference Ground Truth Text (for WER/CER calculation):",
            value=DEFAULT_SPOKEN_REF if "Speech" in input_source else DEFAULT_REF_TEXT,
            height=100
        )

    run_btn = st.button("Execute ASR Transcription & Alignment", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Running inference with {model_choice} on CPU..."):
            # Save temporary wav
            temp_wav = root_dir / "data/processed/temp_gui_audio.wav"
            save_audio(conditioned_audio, sample_rate, temp_wav)

            # Build generation kwargs
            gen_kwargs = {}
            if rep_penalty > 1.01:
                gen_kwargs["repetition_penalty"] = float(rep_penalty)
            if use_anti_loop:
                gen_kwargs["no_repeat_ngram_size"] = 3
            if num_beams > 1:
                gen_kwargs["num_beams"] = int(num_beams)
            if enable_prompt and custom_prompt.strip():
                gen_kwargs["initial_prompt"] = custom_prompt.strip()

            engine = WhisperEngine(model_id=model_choice, device="cpu")
            res_asr = engine.transcribe(temp_wav, generate_kwargs=gen_kwargs)

            # Metrics
            metrics = compute_metrics(ref_text_input, res_asr.transcription)
            alignment = align_words(ref_text_input, res_asr.transcription)

        st.success("Inference completed successfully.")

        # Metric Displays
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric("Word Error Rate (WER)", f"{metrics.wer * 100:.2f}%")
        mcol2.metric("Char Error Rate (CER)", f"{metrics.cer * 100:.2f}%")
        mcol3.metric("Deletions", f"{alignment.deletions}")
        mcol4.metric("Insertions", f"{alignment.insertions}")
        mcol5.metric("RTF (Speed)", f"{res_asr.real_time_factor:.3f}")

        # Output Text Box
        st.write("**Decoded ASR Output:**")
        st.info(res_asr.transcription if res_asr.transcription else "(No speech detected / Empty transcript)")

        # Detailed Error Alignment Breakdown
        with st.expander("Detailed Levenshtein Word-by-Word Alignment Breakdown"):
            st.write(f"**Total Reference Words:** {alignment.total_ref_words} | **Matches:** {alignment.matches} | **Substitutions:** {alignment.substitutions}")
            diff_table = []
            for op in alignment.operations:
                if op.op_type != "match":
                    diff_table.append({
                        "Operation": op.op_type.upper(),
                        "Reference Word": str(op.ref_word),
                        "Decoded Hypothesis": str(op.hyp_word),
                        "Tone Mismatch": op.is_tone_mismatch
                    })
            if diff_table:
                st.dataframe(diff_table, use_container_width=True)
            else:
                st.write("Perfect alignment! Zero recognition errors.")


# --- TAB 2: Acoustic DSP Analysis ---
with tab2:
    st.subheader("Physical Acoustic Phonetics & Spectral Analysis")
    st.write("Extracts acoustic correlates of speech vs. singing divergence ($F_0$, Formants $F_1/F_2$, HNR, Spectral Tilt).")

    with st.spinner("Extracting physical acoustic parameters..."):
        # Waveform & Spectrogram
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        time_axis = np.linspace(0, len(conditioned_audio) / sample_rate, len(conditioned_audio))

        # 1. Waveform
        ax1.plot(time_axis, conditioned_audio, color="#2563EB", lw=0.8)
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Waveform Time Series")
        ax1.grid(True, alpha=0.3)

        # 2. Spectrogram
        ax2.specgram(conditioned_audio, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="inferno")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_title("Spectrogram (Energy & Harmonic Resonance)")
        ax2.set_ylim(0, 5000)

        # 3. F0 Pitch Contour
        f0_contour, f0_metrics = extract_f0_contour(conditioned_audio, sr=sample_rate)
        f0_time = np.linspace(0, len(conditioned_audio) / sample_rate, len(f0_contour))
        ax3.plot(f0_time, f0_contour, color="#10B981", lw=1.5, label="F0 Contour (Hz)")
        ax3.set_ylabel("F0 Pitch (Hz)")
        ax3.set_xlabel("Time (seconds)")
        ax3.set_title(f"Fundamental Frequency F0 (Mean: {f0_metrics.mean_f0_hz:.1f} Hz | CV_F0: {f0_metrics.cv_f0:.3f})")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right")

        plt.tight_layout()
        st.pyplot(fig)

        # Quantitative Summary
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        qcol1.metric("Pitch Variance (CV_F0)", f"{f0_metrics.cv_f0:.4f}")
        qcol2.metric("Mean F0 Pitch", f"{f0_metrics.mean_f0_hz:.1f} Hz")
        formants = estimate_formants_lpc(conditioned_audio, sr=sample_rate)
        qcol3.metric("Formants (F1 / F2)", f"{formants.f1_mean_hz:.0f} / {formants.f2_mean_hz:.0f} Hz")
        hnr = compute_hnr(conditioned_audio, sr=sample_rate)
        qcol4.metric("Harmonic-to-Noise (HNR)", f"{hnr:.1f} dB")


# --- TAB 3: Cross-Attention Monitor ---
with tab3:
    st.subheader("Cross-Attention Alignment & Temporal Dwell Diagnostics")
    st.write(
        "Mathematically traces the progression of cross-attention weights $A(t, f)$ across decoder steps. "
        "Demonstrates whether the model advances monotonically along audio time or stalls in sustained vowel loops."
    )

    attn_btn = st.button("Run Eager Cross-Attention Diagnostics", use_container_width=True)

    if attn_btn:
        with st.spinner("Extracting eager cross-attention matrices across all layers..."):
            temp_wav = root_dir / "data/processed/temp_gui_audio.wav"
            save_audio(conditioned_audio, sample_rate, temp_wav)

            diag = analyze_cross_attention(temp_wav, model_id=model_choice, device="cpu")

        st.write(f"**Attention Path Analysis for {diag.model_name}:**")
        acol1, acol2, acol3, acol4 = st.columns(4)
        acol1.metric("Monotonicity Index (Spearman rho)", f"{diag.monotonicity_index:.4f}")
        acol2.metric("Stalled Tokens Ratio", f"{diag.stalling_ratio * 100:.1f}%")
        acol3.metric("Stalled Token Count", f"{diag.stalled_token_count} / {diag.total_tokens}")
        acol4.metric("Loop Detected", "YES" if diag.loop_detected else "NO")

        # Plot Alignment Path
        if diag.attention_path:
            fig_attn, ax_a = plt.subplots(figsize=(10, 4))
            token_idx = np.arange(len(diag.attention_path))
            ax_a.plot(token_idx, diag.attention_path, marker="o", color="#8B5CF6", lw=1.5, markersize=3)
            ax_a.set_xlabel("Generated Token Index t")
            ax_a.set_ylabel("Peak Audio Frame Index f")
            ax_a.set_title(f"Cross-Attention Alignment Path: Monotonicity Index = {diag.monotonicity_index:.4f}")
            ax_a.grid(True, alpha=0.3)
            st.pyplot(fig_attn)


# --- TAB 4: Scientific Benchmark Summary ---
with tab4:
    st.subheader("Empirical Research Findings (RQ1 - RQ6 Synthesis)")
    
    # Load Exp 6 results if exists
    exp6_path = root_dir / "data/processed/exp6_deep_benchmark_results.json"
    if exp6_path.exists():
        with open(exp6_path, "r", encoding="utf-8") as f:
            data_exp6 = json.load(f)
        
        st.write("**Controlled Benchmark Matrix Across Mitigation Tiers:**")
        table_rows = []
        for r in data_exp6.get("results", []):
            table_rows.append({
                "ID": r.get("id"),
                "Configuration": r.get("name"),
                "WER (%)": f"{r.get('wer') * 100:.2f}%",
                "CER (%)": f"{r.get('cer') * 100:.2f}%",
                "Deletions": r.get("deletions"),
                "Insertions": r.get("insertions"),
                "RTF": r.get("rtf")
            })
        st.dataframe(table_rows, use_container_width=True)

    st.markdown(r"""
    ### Key Academic Discoveries:
    1. **RQ1 (Speech vs Sung Gap):** Spoken WER $35.40\%$ vs Sung WER $135.90\%$ ($\Delta\text{WER} = +100.50\%$).
    2. **RQ2 (Acoustic Correlates):** Formant F2 shift ($r = -0.9186$) and Tone Neutralization ($CV_{F_0}$ $r = -0.8233$) are the principal mathematical drivers ($R^2 = 0.8838$).
    3. **RQ3 (Accompaniment Paradox):** Vocal isolation removes BGM masking but triggers catastrophic autoregressive repetition loops ($1069.23\%$ WER) on sustained vowels.
    4. **RQ4 (Zero-Shot Normalization):** Phase Vocoder tempo 1.1x breaks the infinite loop, reducing WER to $94.87\%$.
    5. **RQ5 (Decoding & Foundation Model):** Vietnamese PhoWhisper is naturally resilient against loops. `no_repeat_ngram_size = 3` accelerates inference by $5.1\text{x}$.
    6. **RQ6 (Deep Speech DSP):** Non-linear Adaptive Vowel TSM compresses only prolonged vowels, reducing word deletions to **$0$** and dropping accompanied polyphonic mix WER to **$125.64\%$**.
    """)

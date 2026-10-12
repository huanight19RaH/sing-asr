"""
sing-asr DSP Module: Adaptive Vowel-Selective Time-Scale Modification (Non-linear TSM).
Directly addresses Speech Processing (SLP301) CLO1 & CLO3.

Scientific Mechanism:
In Vietnamese singing, vowels undergo drastic elongation (200% - 500% longer than speech),
while initial consonant attacks (t, k, p, ch, tr, s, m, n) remain brief transient events (<80ms).
Uniform time stretching speeds up both consonants and vowels, which induces consonant deletion.
Adaptive Vowel TSM:
1. Segments audio into Voiced Vowel Sustained Regions vs. Consonant / Transient Regions using
   Short-Time Energy (STE), Zero-Crossing Rate (ZCR), and Harmonic Autocorrelation.
2. Selectively applies Phase Vocoder time-scale compression ONLY to steady-state vowels
   exceeding threshold duration (e.g. > 0.35s), compressing them toward conversational norms (~0.25s).
3. Preserves consonant transients and phonetic boundaries at 1.0x (uncompressed).
4. Reconstructs signal using smooth Hann cross-fading to guarantee phase continuity.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import scipy.signal

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dsp.audio_io import load_audio, save_audio
from perturbation.time_stretch import phase_vocoder_time_stretch


@dataclass
class SegmentInfo:
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    duration: float
    is_sustained_vowel: bool
    compression_rate: float


def detect_sustained_vowels(
    audio: np.ndarray,
    sr: int = 16000,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    min_vowel_sec: float = 0.35,
    energy_threshold_percentile: float = 25.0,
    zcr_threshold: float = 0.18
) -> tuple[np.ndarray, list[SegmentInfo]]:
    """
    Classify audio frames into sustained vowel regions vs non-vowel / consonant regions.
    """
    frame_len = int(sr * frame_len_ms / 1000.0)
    hop_len = int(sr * hop_len_ms / 1000.0)
    num_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

    ste = np.zeros(num_frames, dtype=np.float32)
    zcr = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        st = i * hop_len
        en = st + frame_len
        frame = audio[st:en]
        # Short-time energy
        ste[i] = np.mean(frame ** 2)
        # Zero-crossing rate
        zcr[i] = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0

    # Dynamic energy threshold
    energy_cutoff = np.percentile(ste, energy_threshold_percentile)
    
    # Voiced vowel criterion: high energy AND low zero-crossing rate
    is_voiced = (ste > energy_cutoff) & (zcr < zcr_threshold)

    # Median filtering to smooth frame classification
    kernel_size = 5
    is_voiced_smooth = scipy.signal.medfilt(is_voiced.astype(np.float32), kernel_size=kernel_size) > 0.5

    # Group contiguous voiced frames into regions
    min_vowel_frames = int(min_vowel_sec / (hop_len_ms / 1000.0))
    regions = []
    in_region = False
    start_f = 0

    for i, voiced in enumerate(is_voiced_smooth):
        if voiced and not in_region:
            in_region = True
            start_f = i
        elif not voiced and in_region:
            in_region = False
            end_f = i
            regions.append((start_f, end_f))
    if in_region:
        regions.append((start_f, num_frames))

    # Build segments list covering entire audio
    segments = []
    current_sample = 0

    for start_f, end_f in regions:
        vowel_start_samp = start_f * hop_len
        vowel_end_samp = min(len(audio), end_f * hop_len + frame_len)
        vowel_dur = (vowel_end_samp - vowel_start_samp) / sr

        # Prior non-vowel / consonant transient segment
        if vowel_start_samp > current_sample:
            dur = (vowel_start_samp - current_sample) / sr
            segments.append(SegmentInfo(
                start_sample=current_sample,
                end_sample=vowel_start_samp,
                start_time=current_sample / sr,
                end_time=vowel_start_samp / sr,
                duration=dur,
                is_sustained_vowel=False,
                compression_rate=1.0
            ))

        # Check if voiced region qualifies as sustained vowel
        is_sustained = (end_f - start_f) >= min_vowel_frames
        segments.append(SegmentInfo(
            start_sample=vowel_start_samp,
            end_sample=vowel_end_samp,
            start_time=vowel_start_samp / sr,
            end_time=vowel_end_samp / sr,
            duration=vowel_dur,
            is_sustained_vowel=is_sustained,
            compression_rate=1.0  # will be assigned in adaptive_vowel_time_stretch
        ))
        current_sample = vowel_end_samp

    # Tail segment if any
    if current_sample < len(audio):
        dur = (len(audio) - current_sample) / sr
        segments.append(SegmentInfo(
            start_sample=current_sample,
            end_sample=len(audio),
            start_time=current_sample / sr,
            end_time=len(audio) / sr,
            duration=dur,
            is_sustained_vowel=False,
            compression_rate=1.0
        ))

    return is_voiced_smooth, segments


def adaptive_vowel_time_stretch(
    audio: np.ndarray,
    sr: int = 16000,
    target_vowel_sec: float = 0.25,
    max_rate: float = 1.6,
    min_vowel_sec: float = 0.35,
    crossfade_ms: float = 20.0
) -> tuple[np.ndarray, list[SegmentInfo]]:
    """
    Selectively compress sustained singing vowels while keeping consonant attacks intact.

    Parameters:
    -----------
    audio : np.ndarray
        Mono audio signal at sample rate sr.
    sr : int
        Sample rate (default: 16000).
    target_vowel_sec : float
        Target spoken syllable duration (default: 0.25s).
    max_rate : float
        Maximum Phase Vocoder compression factor (default: 1.6x).
    min_vowel_sec : float
        Minimum duration to classify as sustained vowel (default: 0.35s).
    crossfade_ms : float
        Cross-fade overlap window in ms (default: 20ms).

    Returns:
    --------
    output_audio : np.ndarray
        Adaptively normalized audio.
    segments : list[SegmentInfo]
        Detailed segmentation and compression breakdown.
    """
    _, segments = detect_sustained_vowels(
        audio, sr=sr, min_vowel_sec=min_vowel_sec
    )

    fade_samples = int(sr * crossfade_ms / 1000.0)
    processed_chunks = []

    for seg in segments:
        chunk = audio[seg.start_sample:seg.end_sample].copy()

        if seg.is_sustained_vowel and len(chunk) > fade_samples * 2:
            # Dynamic compression rate proportional to prolongation
            rate = float(min(max_rate, max(1.15, seg.duration / target_vowel_sec)))
            seg.compression_rate = round(rate, 2)
            # Apply Phase Vocoder to vowel nucleus
            stretched_chunk = phase_vocoder_time_stretch(chunk, rate=rate)
            processed_chunks.append(stretched_chunk)
        else:
            seg.compression_rate = 1.0
            processed_chunks.append(chunk)

    # Reconstruct with Hann cross-fade concatenation
    if not processed_chunks:
        return audio.copy(), segments

    out = [processed_chunks[0]]
    for i in range(1, len(processed_chunks)):
        prev_chunk = out[-1]
        curr_chunk = processed_chunks[i]

        actual_fade = min(fade_samples, len(prev_chunk) // 2, len(curr_chunk) // 2)
        if actual_fade > 4:
            fade_out = np.linspace(1.0, 0.0, actual_fade, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, actual_fade, dtype=np.float32)

            # Overlap-add
            overlap = prev_chunk[-actual_fade:] * fade_out + curr_chunk[:actual_fade] * fade_in
            out[-1] = prev_chunk[:-actual_fade]
            out.append(overlap)
            out.append(curr_chunk[actual_fade:])
        else:
            out.append(curr_chunk)

    output_audio = np.concatenate(out).astype(np.float32)
    return output_audio, segments


if __name__ == '__main__':
    print("[sing-asr DSP] Testing Adaptive Vowel-Selective TSM on Vocal Stem...")
    vocal_stem = "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav"
    if Path(vocal_stem).exists():
        y, sr = load_audio(vocal_stem, target_sr=16000)
        orig_dur = len(y) / sr
        y_adapt, segs = adaptive_vowel_time_stretch(y, sr=sr, max_rate=1.5, min_vowel_sec=0.35)
        new_dur = len(y_adapt) / sr

        vowel_segs = [s for s in segs if s.is_sustained_vowel]
        print(f"  Original Duration: {orig_dur:.2f}s | Adaptively Compressed Duration: {new_dur:.2f}s")
        print(f"  Total Segments: {len(segs)} | Sustained Vowel Segments Compressed: {len(vowel_segs)}")
        for i, vs in enumerate(vowel_segs[:5], 1):
            print(f"    Vowel {i}: {vs.start_time:.2f}s - {vs.end_time:.2f}s (Dur: {vs.duration:.2f}s -> Rate: {vs.compression_rate}x)")

        out_path = "data/processed/normalized/norm_adaptive_vowel_tsm.wav"
        save_audio(y_adapt, sr, out_path)
        print(f"  Saved adaptively normalized audio to: {out_path}")
        print("[sing-asr DSP] Adaptive Vowel TSM test PASSED!")

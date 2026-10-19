"""
src/acoustics/instrument_collision_profiler.py
Phase XXIII: Sub-Agent 2 (DSP Acoustic Profiler)

Calculates mathematical acoustic collision metrics between musical instruments and singing vocals:
1. Transient Energy Index (TEI): High-frequency percussive crest factor and transient masking (Drums).
2. Formant Disturbance Ratio (FDR): Polyphonic harmonic overlap in vocal formant bands F1/F2 (Piano).
3. Pitch Tracking Deviation (PTD): F0 contour perturbation and vibrato collision rate (Guitar).
4. Spectral Energy Floor (SEF): Low-frequency sub-bass power and dynamic range depression (Synth/808).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import scipy.signal as signal
import librosa

class InstrumentCollisionProfiler:
    def __init__(self, sr=16000, n_fft=1024, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def compute_tei(self, audio):
        """
        Transient Energy Index (TEI):
        Measures high-frequency (>2.5 kHz) crest factor (peak to RMS ratio)
        and high-frequency transient density. Drums and cymbals yield high TEI.
        """
        nyq = 0.5 * self.sr
        cutoff = 2500.0 / nyq
        b, a = signal.butter(4, cutoff, btype='high')
        hi_audio = signal.filtfilt(b, a, audio)

        peak = np.max(np.abs(hi_audio)) + 1e-9
        rms = np.sqrt(np.mean(hi_audio**2)) + 1e-9
        crest_factor_db = 20.0 * np.log10(peak / rms)

        # Onset envelope
        onset_env = librosa.onset.onset_strength(y=hi_audio, sr=self.sr, hop_length=self.hop_length)
        transient_density = np.mean(onset_env > np.percentile(onset_env, 85))

        tei = float(crest_factor_db * (1.0 + transient_density))
        return round(tei, 4)

    def compute_fdr(self, mixture_audio, clean_vocal_audio):
        """
        Formant Disturbance Ratio (FDR):
        Measures relative spectral deviation in the vocal formant region (300 - 2500 Hz).
        Piano chords with rich polyphony produce strong FDR.
        """
        min_len = min(len(mixture_audio), len(clean_vocal_audio))
        mix = mixture_audio[:min_len]
        voc = clean_vocal_audio[:min_len]

        # STFT
        S_mix = np.abs(librosa.stft(mix, n_fft=self.n_fft, hop_length=self.hop_length))
        S_voc = np.abs(librosa.stft(voc, n_fft=self.n_fft, hop_length=self.hop_length))

        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        formant_bins = np.where((freqs >= 300) & (freqs <= 2500))[0]

        band_mix = S_mix[formant_bins, :]
        band_voc = S_voc[formant_bins, :]

        # Normalized Frobenius distance in formant band
        norm_voc = np.linalg.norm(band_voc) + 1e-9
        diff_norm = np.linalg.norm(band_mix - band_voc)
        fdr = float(diff_norm / norm_voc)
        return round(fdr, 4)

    def compute_ptd(self, mixture_audio, clean_vocal_audio):
        """
        Pitch Tracking Deviation (PTD):
        Measures pitch tracking error (octave hopping, vibrato interference)
        caused by competing harmonic sources (Guitar string vibrato / pitch bend).
        """
        min_len = min(len(mixture_audio), len(clean_vocal_audio))
        mix = mixture_audio[:min_len]
        voc = clean_vocal_audio[:min_len]

        # F0 estimation via librosa yin (fast and robust)
        f0_mix = librosa.yin(
            mix, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'),
            sr=self.sr, hop_length=self.hop_length
        )
        f0_voc = librosa.yin(
            voc, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'),
            sr=self.sr, hop_length=self.hop_length
        )

        valid = (~np.isnan(f0_mix) & ~np.isnan(f0_voc) & (f0_mix > 65) & (f0_voc > 65))
        if np.sum(valid) < 10:
            return 0.0

        f0_mix_v = f0_mix[valid]
        f0_voc_v = f0_voc[valid]

        # Convert to semitone differences
        st_diff = 12.0 * np.abs(np.log2(f0_mix_v / f0_voc_v))
        # Rate of deviation > 1 semitone (pitch perturbation)
        ptd = float(np.mean(st_diff > 1.0))
        return round(ptd, 4)

    def compute_sef(self, audio):
        """
        Spectral Energy Floor (SEF):
        Proportion of total spectral energy situated in the sub-bass (<150 Hz) floor.
        Synth / 808 bass yields high SEF, causing filterbank dynamic range depression.
        """
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)) ** 2
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        sub_bins = np.where(freqs <= 150)[0]

        sub_energy = np.sum(S[sub_bins, :])
        total_energy = np.sum(S) + 1e-9

        sef = float(sub_energy / total_energy)
        return round(sef, 4)

    def profile_collision(self, mixture_audio, clean_vocal_audio=None):
        """
        Runs full acoustic collision profile.
        """
        tei = self.compute_tei(mixture_audio)
        sef = self.compute_sef(mixture_audio)

        fdr = 0.0
        ptd = 0.0
        if clean_vocal_audio is not None:
            fdr = self.compute_fdr(mixture_audio, clean_vocal_audio)
            ptd = self.compute_ptd(mixture_audio, clean_vocal_audio)

        return {
            "tei_transient_index": tei,
            "fdr_formant_disturbance": fdr,
            "ptd_pitch_deviation": ptd,
            "sef_subbass_floor": sef
        }

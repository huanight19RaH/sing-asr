"""
sing-asr Dataset & Acoustic Augmentation Module for LoRA Adaptation (Phase VIII).
Directly addresses Phase VIII of the Research Plan (SLP301 - AIT Laboratory).
Lead Researcher: Thai Hung Nguyen

Constructs segmented, paired acoustic training instances from Vietnamese singing
and speech audio, augmented with Phase VII DSP transforms (HPSS, TSM, pitch perturbation).
"""
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dsp.audio_io import load_audio
from dsp.hpss_consonant_boost import boost_consonant_transients
from dsp.adaptive_tsm import adaptive_vowel_time_stretch
from dsp.vibrato_filter import suppress_vibrato_stft


class VietnameseSingingDataset(Dataset):
    """
    Paired Audio-Text Dataset for Vietnamese Singing Voice Adaptation.
    """
    def __init__(
        self,
        processor: WhisperProcessor,
        items: list[dict],
        augment: bool = False,
        normalize_cmvn: bool = False,
        normalize_rms: bool = False
    ):
        self.processor = processor
        self.items = items
        self.augment = augment
        self.normalize_cmvn = normalize_cmvn
        self.normalize_rms = normalize_rms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio = item["audio"].copy()
        sr = item.get("sr", 16000)
        text = item["text"]
        is_speech = item.get("id", "").startswith("spoken")

        # Waveform RMS energy normalization
        if self.normalize_rms:
            from dsp.normalization import normalize_waveform_rms
            audio = normalize_waveform_rms(audio, target_dbfs=-20.0)

        # On-the-fly acoustic augmentation
        if self.augment:
            aug_type = np.random.choice(["none", "hpss_boost", "vibrato_suppress", "noise"])
            if aug_type == "hpss_boost":
                audio, _ = boost_consonant_transients(audio, sr=sr, boost_factor=np.random.uniform(1.3, 1.8))
            elif aug_type == "vibrato_suppress":
                audio, _ = suppress_vibrato_stft(audio, sr=sr)
            elif aug_type == "noise":
                noise = np.random.normal(0, 0.005, size=audio.shape).astype(np.float32)
                audio = audio + noise

        # Extract log-Mel filterbanks
        input_features = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        ).input_features[0]

        # Utterance-level Cepstral Mean & Variance Normalization (CMVN)
        if self.normalize_cmvn:
            from dsp.normalization import apply_spectrogram_cmvn
            input_features = apply_spectrogram_cmvn(input_features)

        # Tokenize labels
        labels = self.processor.tokenizer(text).input_ids

        return {
            "input_features": input_features,
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": text,
            "id": item.get("id", f"sample_{idx}"),
            "is_speech": is_speech
        }


def collate_fn_whisper(batch, processor: WhisperProcessor):
    """
    Pads input features and labels dynamically for Whisper training.
    """
    input_features = torch.stack([item["input_features"] for item in batch])
    label_list = [item["labels"] for item in batch]
    is_speech_flags = torch.tensor([item.get("is_speech", False) for item in batch], dtype=torch.bool)

    # Pad labels to max length in batch
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        label_list,
        batch_first=True,
        padding_value=-100  # -100 is ignored by CrossEntropyLoss
    )

    return {
        "input_features": input_features,
        "labels": padded_labels,
        "is_speech": is_speech_flags
    }


def create_singing_paired_corpus(root_dir: Path) -> list[dict]:
    """
    Extracts segmented, aligned audio instances from the real benchmark assets.
    """
    corpus = []
    vocal_stem_wav = root_dir / "data/processed/separated/vietnamese_song_clip_26s_vocals_16k.wav"
    poly_mix_wav = root_dir / "data/processed/vietnamese_song_clip_26s.wav"
    speech_wav = root_dir / "data/processed/vietnamese_speech_real_01_16k.wav"

    # Segment definitions for the 26s lyric clip
    song_segments = [
        (0.0, 8.5, "Trời còn làm mây xanh ngắt cho mắt em buồn", "seg1"),
        (8.5, 17.5, "Trời còn làm mưa bay bay cho hạt sầu lên mắt người thương", "seg2"),
        (17.5, 26.1, "Lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương", "seg3"),
    ]

    # 1. Extract from Vocal Stem
    if vocal_stem_wav.exists():
        y_vocal, sr = load_audio(vocal_stem_wav, target_sr=16000)
        # Full clip
        corpus.append({
            "id": "vocal_full",
            "audio": y_vocal,
            "sr": sr,
            "text": "Trời còn làm mây xanh ngắt cho mắt em buồn trời còn làm mưa bay bay cho hạt sầu lên mắt người thương lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương"
        })
        # Sub-bars
        for start_t, end_t, text, seg_id in song_segments:
            s_idx = int(start_t * sr)
            e_idx = min(len(y_vocal), int(end_t * sr))
            corpus.append({
                "id": f"vocal_{seg_id}",
                "audio": y_vocal[s_idx:e_idx],
                "sr": sr,
                "text": text
            })

    # 2. Extract from Polyphonic Mix
    if poly_mix_wav.exists():
        y_poly, sr = load_audio(poly_mix_wav, target_sr=16000)
        corpus.append({
            "id": "poly_full",
            "audio": y_poly,
            "sr": sr,
            "text": "Trời còn làm mây xanh ngắt cho mắt em buồn trời còn làm mưa bay bay cho hạt sầu lên mắt người thương lời ru nào buông lơi theo từng giọt nắng để lại niềm thương nhớ vấn vương"
        })
        for start_t, end_t, text, seg_id in song_segments:
            s_idx = int(start_t * sr)
            e_idx = min(len(y_poly), int(end_t * sr))
            corpus.append({
                "id": f"poly_{seg_id}",
                "audio": y_poly[s_idx:e_idx],
                "sr": sr,
                "text": text
            })

    # 3. Extract Spoken Anchor
    if speech_wav.exists():
        y_speech, sr = load_audio(speech_wav, target_sr=16000)
        speech_text = (
            "đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng cho mục đích huấn luyện trí tuệ nhân tạo "
            "mỗi bản ghi âm được chuẩn bị cẩn thận để đảm bảo phát âm rõ ràng nhịp điệu tự nhiên và chất lượng âm thanh ổn định"
        )
        corpus.append({
            "id": "spoken_anchor_full",
            "audio": y_speech,
            "sr": sr,
            "text": speech_text
        })
        # Half split
        mid_idx = len(y_speech) // 2
        corpus.append({
            "id": "spoken_anchor_part1",
            "audio": y_speech[:mid_idx],
            "sr": sr,
            "text": "đây là bộ dữ liệu âm thanh giọng nói mà bạn có thể sử dụng cho mục đích huấn luyện trí tuệ nhân tạo"
        })
        corpus.append({
            "id": "spoken_anchor_part2",
            "audio": y_speech[mid_idx:],
            "sr": sr,
            "text": "mỗi bản ghi âm được chuẩn bị cẩn thận để đảm bảo phát âm rõ ràng nhịp điệu tự nhiên và chất lượng âm thanh ổn định"
        })

    return corpus


def create_expanded_benchmark_corpus(root_dir: Path) -> list[dict]:
    """
    Loads all multi-track singing and multi-speaker VIVOS speech items from benchmark_manifest.json.
    """
    import json
    manifest_file = root_dir / "data/processed/benchmark_manifest.json"
    if not manifest_file.exists():
        return create_singing_paired_corpus(root_dir)

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    corpus = []
    for item in manifest:
        audio_path = root_dir / item["audio_path"]
        if audio_path.exists():
            audio, sr = load_audio(audio_path, target_sr=16000)
            corpus.append({
                "id": item["id"],
                "audio": audio,
                "sr": sr,
                "text": item["text"],
                "modality": item.get("modality", "singing"),
                "category": item.get("category", "general"),
                "is_speech": (item.get("modality") == "speech")
            })
    return corpus

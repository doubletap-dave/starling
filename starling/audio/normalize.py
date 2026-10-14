from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np


TARGET_SR = 16_000


def load_audio_16k_mono(source: bytes | str | Path) -> np.ndarray:
    """Decode audio to float32 mono at 16 kHz for all backends."""
    if isinstance(source, (str, Path)):
        audio, sr = librosa.load(str(source), sr=None, mono=True)
    else:
        audio, sr = librosa.load(io.BytesIO(source), sr=None, mono=True)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return np.asarray(audio, dtype=np.float32)

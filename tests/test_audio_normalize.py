from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from starling.audio.normalize import load_audio_16k_mono


def test_load_wav_bytes():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(8000, dtype=np.float32), 8000, format="WAV")
    audio = load_audio_16k_mono(buf.getvalue(), filename="clip.wav")
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    # 1.0s @ 8kHz → 1.0s @ 16kHz
    assert abs(len(audio) - 16_000) < 32


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_load_m4a_bytes_via_ffmpeg(tmp_path: Path):
    wav = tmp_path / "src.wav"
    m4a = tmp_path / "clip.m4a"
    sf.write(wav, np.sin(np.linspace(0, 20, 16000, dtype=np.float32)), 16000)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav),
            "-c:a",
            "aac",
            str(m4a),
        ],
        check=True,
    )
    raw = m4a.read_bytes()
    audio = load_audio_16k_mono(raw, filename="recording.m4a")
    assert audio.dtype == np.float32
    assert len(audio) > 1000

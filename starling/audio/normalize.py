from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

TARGET_SR = 16_000
logger = logging.getLogger(__name__)

# Formats soundfile can usually handle without ffmpeg
_SOUNDFILE_SUFFIXES = {".wav", ".flac", ".ogg", ".oga"}


def load_audio_16k_mono(
    source: bytes | str | Path,
    *,
    filename: str | None = None,
) -> np.ndarray:
    """Decode audio to float32 mono at 16 kHz for all backends.

    Bytes uploads (e.g. iOS M4A) are written to a temp file with the original
    extension so ffmpeg/audioread can decode them. ``librosa.load(BytesIO)``
    only uses soundfile and cannot decode AAC/M4A even when ffmpeg is installed.
    """
    if isinstance(source, (str, Path)):
        return _load_path(Path(source))

    suffix = _suffix_from_name(filename) or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)
    try:
        return _load_path(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _suffix_from_name(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def _load_path(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    # Prefer ffmpeg for containers that need it (m4a/aac/mp3/…)
    needs_ffmpeg = suffix not in _SOUNDFILE_SUFFIXES or not _try_soundfile(path)
    if needs_ffmpeg:
        audio = _ffmpeg_decode(path)
        if audio is not None:
            return audio
        if suffix not in _SOUNDFILE_SUFFIXES:
            raise ValueError(
                f"Cannot decode {suffix or 'unknown'} audio: ffmpeg is required "
                "(e.g. iOS M4A/AAC). Install ffmpeg in the image/host."
            )

    import librosa

    audio, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return np.asarray(audio, dtype=np.float32)


def _try_soundfile(path: Path) -> bool:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames > 0
    except Exception:
        return False


def _ffmpeg_decode(path: Path) -> np.ndarray | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found on PATH; cannot decode %s via ffmpeg", path.suffix)
        return None

    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        logger.warning("ffmpeg executable missing")
        return None
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise ValueError(f"ffmpeg failed to decode audio ({path.suffix}): {err or exc}") from exc

    if not proc.stdout:
        raise ValueError(f"ffmpeg produced empty audio for {path.suffix}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"ffmpeg produced empty audio for {path.suffix}")
    return audio

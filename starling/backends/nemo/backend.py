from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf

from starling.backends.base import Segment, TranscriptResult, WordStamp
from starling.gpu.placer import Placement
from starling.registry.catalog import ModelEntry

logger = logging.getLogger(__name__)


@contextmanager
def _legacy_checkpoint_load_env() -> Iterator[None]:
    """Allow NeMo's legacy pickle checkpoints to load under torch >= 2.6.

    PyTorch 2.6 flipped ``torch.load``'s ``weights_only`` default to True,
    which rejects the legacy ``.nemo`` checkpoints (e.g. orukeet) that NeMo's
    ``restore_from`` loads without an explicit ``weights_only`` argument.
    ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` is PyTorch/NeMo's prescribed
    override for trusted checkpoints; torch re-reads it on every load, so
    scoping it to ``restore_from`` confines the (unsafe) pickle load to
    registry-pinned NeMo checkpoints instead of the whole process. Only
    enable this for checkpoints from trusted sources -- weights_only=False
    permits arbitrary code execution during unpickling.
    """
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    if previous != "1":
        logger.warning(
            "Loading NeMo checkpoint with TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "
            "(legacy weights_only=False load of a trusted, registry-pinned checkpoint)"
        )
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous


class NemoBackend:
    name = "nemo"

    def load(self, entry: ModelEntry, local_path: str, placement: Placement) -> Any:
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "NeMo backend requires nemo-toolkit[asr]. Install with: uv sync --extra nemo"
            ) from exc

        import torch

        device = placement.device if placement.device != "auto" else "cuda:0"
        logger.info("NeMo restore_from(%s) -> %s", local_path, device)
        with _legacy_checkpoint_load_env():
            model = ASRModel.restore_from(restore_path=local_path, map_location=device)
        model.eval()
        if device.startswith("cuda") and torch.cuda.is_available():
            model = model.to(device)
        return model

    def unload(self, handle: Any) -> None:
        # Manager drops its ref then calls release_cuda_memory()
        del handle

    def transcribe(
        self,
        handle: Any,
        audio: np.ndarray,
        *,
        language: str | None = None,
        word_timestamps: bool = True,
        **kwargs: Any,
    ) -> TranscriptResult:
        import torch

        # NeMo file-path path is most reliable across model classes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, 16000)

        try:
            want_ts = word_timestamps or True
            with torch.inference_mode():
                try:
                    hyps = handle.transcribe([tmp_path], timestamps=want_ts)
                except TypeError:
                    hyps = handle.transcribe([tmp_path])

            hyp = hyps[0]
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            segments: list[Segment] = []
            words: list[WordStamp] = []

            ts = getattr(hyp, "timestamp", None) or {}
            if isinstance(ts, dict):
                for w in ts.get("word") or []:
                    words.append(
                        WordStamp(
                            word=str(w.get("word", w.get("char", ""))),
                            start=float(w.get("start", 0.0)),
                            end=float(w.get("end", 0.0)),
                        )
                    )
                for i, s in enumerate(ts.get("segment") or []):
                    segments.append(
                        Segment(
                            id=i,
                            start=float(s.get("start", 0.0)),
                            end=float(s.get("end", 0.0)),
                            text=str(s.get("segment", s.get("text", ""))).strip(),
                        )
                    )

            if not segments and text:
                duration = float(len(audio) / 16000)
                segments = [Segment(id=0, start=0.0, end=duration, text=text, words=words)]
            elif segments and words:
                segments[0].words = words

            return TranscriptResult(
                text=text.strip() if isinstance(text, str) else str(text),
                language=language,
                duration=float(len(audio) / 16000),
                segments=segments,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

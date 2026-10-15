from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from starling.backends.base import Segment, TranscriptResult, WordStamp
from starling.gpu.placer import Placement
from starling.registry.catalog import ModelEntry

logger = logging.getLogger(__name__)


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

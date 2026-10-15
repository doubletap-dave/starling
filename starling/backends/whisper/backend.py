from __future__ import annotations

import logging
from typing import Any

import numpy as np

from starling.backends.base import Segment, TranscriptResult, WordStamp
from starling.gpu.placer import Placement
from starling.registry.catalog import ModelEntry

logger = logging.getLogger(__name__)


class WhisperBackend:
    name = "whisper"

    def __init__(self, compute_type: str = "default") -> None:
        self.compute_type = compute_type

    def load(self, entry: ModelEntry, local_path: str, placement: Placement) -> Any:
        from faster_whisper import WhisperModel

        device = "cpu" if placement.device == "cpu" else "cuda"
        device_index = placement.device_index if placement.device_index is not None else 0
        compute = self.compute_type
        if compute == "default":
            compute = "float16" if device == "cuda" else "int8"

        # Prefer local snapshot/path; fall back to repo id (HF cache)
        model_ref = local_path or entry.repo
        logger.info("WhisperModel(%s) device=%s index=%s compute=%s", model_ref, device, device_index, compute)
        return WhisperModel(
            model_ref,
            device=device,
            device_index=device_index,
            compute_type=compute,
        )

    def unload(self, handle: Any) -> None:
        # Manager drops its ref then calls release_cuda_memory()
        del handle

    def transcribe(
        self,
        handle: Any,
        audio: np.ndarray,
        *,
        language: str | None = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        word_timestamps: bool = False,
        hotwords: str | None = None,
        prompt: str | None = None,
        **_: Any,
    ) -> TranscriptResult:
        segments_iter, info = handle.transcribe(
            audio,
            language=language,
            task=task,
            temperature=temperature,
            word_timestamps=word_timestamps,
            hotwords=hotwords,
            initial_prompt=prompt,
            vad_filter=True,
        )

        segments: list[Segment] = []
        texts: list[str] = []
        for i, seg in enumerate(segments_iter):
            words = []
            if word_timestamps and seg.words:
                words = [
                    WordStamp(word=w.word, start=float(w.start), end=float(w.end)) for w in seg.words
                ]
            segments.append(
                Segment(
                    id=i,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    words=words,
                )
            )
            texts.append(seg.text.strip())

        return TranscriptResult(
            text=" ".join(t for t in texts if t).strip(),
            language=getattr(info, "language", language),
            duration=float(getattr(info, "duration", 0.0) or 0.0) or None,
            segments=segments,
            task=task,
        )

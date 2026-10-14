from __future__ import annotations

import logging
from typing import Any

import numpy as np

from starling.backends.base import Segment, TranscriptResult
from starling.gpu.placer import Placement
from starling.registry.catalog import ModelEntry

logger = logging.getLogger(__name__)


class TransformersBackend:
    name = "transformers"

    def load(self, entry: ModelEntry, local_path: str, placement: Placement) -> Any:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        dtype = torch.float16 if placement.device != "cpu" and torch.cuda.is_available() else torch.float32
        model_ref = entry.repo
        base_kwargs: dict[str, Any] = {"torch_dtype": dtype}

        def _from_pretrained(**extra: Any):
            return AutoModelForSpeechSeq2Seq.from_pretrained(model_ref, **base_kwargs, **extra)

        if placement.use_device_map:
            logger.info("Transformers load %s device_map=auto", model_ref)
            try:
                model = _from_pretrained(device_map="auto", attn_implementation="sdpa")
            except Exception:
                model = _from_pretrained(device_map="auto")
        else:
            try:
                model = _from_pretrained(attn_implementation="sdpa")
            except Exception:
                model = _from_pretrained()
            model = model.to(placement.device)
            logger.info("Transformers load %s -> %s", model_ref, placement.device)

        model.eval()
        processor = AutoProcessor.from_pretrained(model_ref)
        return {"model": model, "processor": processor, "entry": entry, "placement": placement}

    def unload(self, handle: Any) -> None:
        del handle
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def transcribe(
        self,
        handle: Any,
        audio: np.ndarray,
        *,
        language: str | None = None,
        **_: Any,
    ) -> TranscriptResult:
        import torch

        model = handle["model"]
        processor = handle["processor"]
        entry: ModelEntry = handle["entry"]
        placement: Placement = handle["placement"]

        if entry.requires_language and not language:
            raise ValueError(f"Model '{entry.id}' requires language parameter")

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        if not placement.use_device_map:
            device = placement.device
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        else:
            # Move inputs to first parameter device
            try:
                device = next(model.parameters()).device
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            except StopIteration:
                pass

        gen_kwargs: dict[str, Any] = {}
        if language:
            # Cohere and some models take language= in generate()
            gen_kwargs["language"] = language

        with torch.inference_mode():
            try:
                output = model.generate(**inputs, **gen_kwargs)
            except TypeError:
                # Models that don't accept language kwarg
                output = model.generate(**inputs)

        text = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        duration = float(len(audio) / 16000)
        return TranscriptResult(
            text=text,
            language=language,
            duration=duration,
            segments=[Segment(id=0, start=0.0, end=duration, text=text)],
        )

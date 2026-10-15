from __future__ import annotations

import logging
from typing import Any

import numpy as np

from starling.backends.base import Segment, TranscriptResult
from starling.gpu.placer import Placement
from starling.registry.catalog import ModelEntry

logger = logging.getLogger(__name__)


def _is_qwen3_asr(entry: ModelEntry, processor: Any | None = None) -> bool:
    repo = (entry.repo or "").lower()
    mid = (entry.id or "").lower()
    if "qwen3-asr" in repo or "qwen3_asr" in repo or mid.startswith("qwen3-asr"):
        return True
    return bool(processor is not None and callable(getattr(processor, "apply_transcription_request", None)))


class TransformersBackend:
    name = "transformers"

    def load(self, entry: ModelEntry, local_path: str, placement: Placement) -> Any:
        import torch
        from transformers import AutoProcessor

        dtype = torch.float16 if placement.device != "cpu" and torch.cuda.is_available() else torch.float32
        model_ref = entry.repo
        processor = AutoProcessor.from_pretrained(model_ref)
        qwen3 = _is_qwen3_asr(entry, processor)

        if qwen3:
            model = self._load_qwen3(model_ref, dtype, placement)
        else:
            model = self._load_speech_seq2seq(model_ref, dtype, placement)

        model.eval()
        return {
            "model": model,
            "processor": processor,
            "entry": entry,
            "placement": placement,
            "flavor": "qwen3_asr" if qwen3 else "speech_seq2seq",
        }

    def _load_speech_seq2seq(self, model_ref: str, dtype: Any, placement: Placement) -> Any:
        from transformers import AutoModelForSpeechSeq2Seq

        base_kwargs: dict[str, Any] = {"torch_dtype": dtype}

        def _from_pretrained(**extra: Any):
            return AutoModelForSpeechSeq2Seq.from_pretrained(model_ref, **base_kwargs, **extra)

        if placement.use_device_map:
            logger.info("Transformers SpeechSeq2Seq load %s device_map=auto", model_ref)
            try:
                return _from_pretrained(device_map="auto", attn_implementation="sdpa")
            except Exception:
                return _from_pretrained(device_map="auto")

        try:
            model = _from_pretrained(attn_implementation="sdpa")
        except Exception:
            model = _from_pretrained()
        model = model.to(placement.device)
        logger.info("Transformers SpeechSeq2Seq load %s -> %s", model_ref, placement.device)
        return model

    def _load_qwen3(self, model_ref: str, dtype: Any, placement: Placement) -> Any:
        """Qwen3-ASR is a multimodal LM, not AutoModelForSpeechSeq2Seq."""
        try:
            from transformers import AutoModelForMultimodalLM as ModelCls
        except ImportError:
            try:
                from transformers import Qwen3ASRForConditionalGeneration as ModelCls
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen3-ASR requires transformers>=5.13 with AutoModelForMultimodalLM "
                    "or Qwen3ASRForConditionalGeneration"
                ) from exc

        base_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if placement.use_device_map or placement.device != "cpu":
            # Prefer device_map for multimodal models
            try:
                logger.info("Transformers Qwen3-ASR load %s device_map=auto", model_ref)
                return ModelCls.from_pretrained(model_ref, device_map="auto", **base_kwargs)
            except Exception:
                logger.exception("device_map=auto failed for %s; falling back to .to()", model_ref)

        model = ModelCls.from_pretrained(model_ref, **base_kwargs)
        if placement.device != "cpu":
            model = model.to(placement.device)
        logger.info("Transformers Qwen3-ASR load %s -> %s", model_ref, placement.device)
        return model

    def unload(self, handle: Any) -> None:
        # Clear nested refs so tensors die before manager.empty_cache()
        if isinstance(handle, dict):
            model = handle.pop("model", None)
            handle.clear()
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
                del model
        del handle

    def transcribe(
        self,
        handle: Any,
        audio: np.ndarray,
        *,
        language: str | None = None,
        prompt: str | None = None,
        **_: Any,
    ) -> TranscriptResult:
        entry: ModelEntry = handle["entry"]
        if entry.requires_language and not language:
            raise ValueError(f"Model '{entry.id}' requires language parameter")

        flavor = handle.get("flavor") or (
            "qwen3_asr" if _is_qwen3_asr(entry, handle.get("processor")) else "speech_seq2seq"
        )
        if flavor == "qwen3_asr":
            return self._transcribe_qwen3(handle, audio, language=language, prompt=prompt)
        return self._transcribe_speech_seq2seq(handle, audio, language=language)

    def _move_inputs(self, model: Any, placement: Placement, inputs: Any) -> Any:
        import torch

        if hasattr(inputs, "to"):
            # BatchFeature with .to(device, dtype)
            try:
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
                return inputs.to(device, dtype)
            except StopIteration:
                return inputs

        if isinstance(inputs, dict):
            if placement.use_device_map:
                try:
                    device = next(model.parameters()).device
                except StopIteration:
                    return inputs
            else:
                device = placement.device
            return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs

    def _transcribe_speech_seq2seq(
        self,
        handle: Any,
        audio: np.ndarray,
        *,
        language: str | None = None,
    ) -> TranscriptResult:
        import torch

        model = handle["model"]
        processor = handle["processor"]
        placement: Placement = handle["placement"]

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = self._move_inputs(model, placement, inputs)

        gen_kwargs: dict[str, Any] = {}
        if language:
            gen_kwargs["language"] = language

        with torch.inference_mode():
            try:
                output = model.generate(**inputs, **gen_kwargs)
            except TypeError:
                output = model.generate(**inputs)

        text = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        duration = float(len(audio) / 16000)
        return TranscriptResult(
            text=text,
            language=language,
            duration=duration,
            segments=[Segment(id=0, start=0.0, end=duration, text=text)],
        )

    def _transcribe_qwen3(
        self,
        handle: Any,
        audio: np.ndarray,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptResult:
        import torch

        model = handle["model"]
        processor = handle["processor"]
        placement: Placement = handle["placement"]

        # Multimodal chat-template path — not processor(audio, ...)
        req_kwargs: dict[str, Any] = {"audio": np.asarray(audio, dtype=np.float32)}
        if language:
            req_kwargs["language"] = language
        if prompt:
            req_kwargs["prompt"] = prompt

        try:
            inputs = processor.apply_transcription_request(**req_kwargs)
        except TypeError:
            # Some versions want sampling_rate via chat template kwargs
            inputs = processor.apply_transcription_request(
                audio=np.asarray(audio, dtype=np.float32),
                language=language,
                sampling_rate=16000,
            )

        inputs = self._move_inputs(model, placement, inputs)
        input_len = int(inputs["input_ids"].shape[1])

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=256)

        generated_ids = output_ids[:, input_len:]

        detected_lang = language
        try:
            text = processor.decode(generated_ids, return_format="transcription_only")[0]
            if isinstance(text, dict):
                text = text.get("transcription") or text.get("text") or str(text)
            text = str(text).strip()
            if not language:
                try:
                    parsed = processor.decode(generated_ids, return_format="parsed")[0]
                    if isinstance(parsed, dict) and parsed.get("language"):
                        detected_lang = str(parsed["language"])
                except Exception:
                    pass
        except TypeError:
            # Older decode without return_format
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        duration = float(len(audio) / 16000)
        return TranscriptResult(
            text=text,
            language=detected_lang,
            duration=duration,
            segments=[Segment(id=0, start=0.0, end=duration, text=text)],
        )

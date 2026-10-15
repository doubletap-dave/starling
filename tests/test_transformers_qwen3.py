from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from starling.backends.transformers.backend import TransformersBackend, _is_qwen3_asr
from starling.gpu.placer import Placement
from starling.registry.catalog import ModelEntry


def test_is_qwen3_asr_by_repo():
    entry = ModelEntry(id="qwen3-asr-0.6b", backend="transformers", repo="Qwen/Qwen3-ASR-0.6B-hf")
    assert _is_qwen3_asr(entry) is True


def test_is_qwen3_asr_by_processor():
    entry = ModelEntry(id="custom", backend="transformers", repo="org/other")
    proc = SimpleNamespace(apply_transcription_request=lambda **kw: None)
    assert _is_qwen3_asr(entry, proc) is True


def test_qwen3_transcribe_uses_apply_transcription_request():
    backend = TransformersBackend()
    entry = ModelEntry(id="qwen3-asr-0.6b", backend="transformers", repo="Qwen/Qwen3-ASR-0.6B-hf")
    audio = np.zeros(16000, dtype=np.float32)

    class FakeProcessor:
        def __init__(self):
            self.called_with = None

        def apply_transcription_request(self, **kwargs):
            self.called_with = kwargs
            ids = torch.tensor([[1, 2, 3, 4]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def decode(self, ids, return_format="raw"):
            assert return_format == "transcription_only"
            return ["hello from qwen"]

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            # prompt len 4 + 2 new tokens
            return torch.tensor([[1, 2, 3, 4, 9, 8]])

    processor = FakeProcessor()
    model = FakeModel()
    handle = {
        "model": model,
        "processor": processor,
        "entry": entry,
        "placement": Placement(device="cpu", device_index=None),
        "flavor": "qwen3_asr",
    }

    result = backend.transcribe(handle, audio, language="en")
    assert processor.called_with is not None
    assert "audio" in processor.called_with
    assert processor.called_with.get("language") == "en"
    assert result.text == "hello from qwen"


def test_speech_seq2seq_still_uses_processor_call():
    backend = TransformersBackend()
    entry = ModelEntry(id="moonshine-base", backend="transformers", repo="UsefulSensors/moonshine-base")
    audio = np.zeros(1600, dtype=np.float32)

    class FakeProcessor:
        def __call__(self, audio, sampling_rate=16000, return_tensors="pt"):
            ids = torch.tensor([[1.0, 2.0]])
            return {"input_features": ids}

        def batch_decode(self, output, skip_special_tokens=True):
            return ["classic path"]

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            return torch.tensor([[5, 6]])

    handle = {
        "model": FakeModel(),
        "processor": FakeProcessor(),
        "entry": entry,
        "placement": Placement(device="cpu", device_index=None),
        "flavor": "speech_seq2seq",
    }
    result = backend.transcribe(handle, audio)
    assert result.text == "classic path"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from starling.api.app import create_app
from starling.backends.base import TranscriptResult
from starling.config import Settings
from starling.gpu.placer import Placement
from starling.lifecycle.manager import LoadedModel
from starling.registry.catalog import ModelEntry


class FakeBackend:
    name = "whisper"

    def load(self, entry, local_path, placement):
        return {"ok": True, "id": entry.id}

    def unload(self, handle):
        pass

    def transcribe(self, handle, audio, **kwargs):
        assert isinstance(audio, np.ndarray)
        return TranscriptResult(text="hello from fake", language=kwargs.get("language") or "en", duration=1.0)


@pytest.fixture
def client(tmp_path):
    registry_src = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")
    runtime_path = tmp_path / "runtime.yaml"

    settings = Settings(
        registry_path=registry_path,
        runtime_config_path=runtime_path,
        stt_model_ttl=-1,
        api_key=None,
        preload_models=[],
        metrics_enabled=True,
        enable_ui=True,
        device_override="cpu",
    )
    app = create_app(settings)

    with TestClient(app) as c:
        mgr = c.app.state.manager
        fake = FakeBackend()
        mgr.backends = {"whisper": fake, "transformers": fake, "nemo": fake}

        mgr.registry.download = lambda e: str(tmp_path / "model")  # type: ignore[method-assign]
        mgr.registry.is_downloaded = lambda e: True  # type: ignore[method-assign]
        mgr.registry.local_path = lambda e: str(tmp_path / "model")  # type: ignore[method-assign]

        def load_no_place(model_id: str):
            entry = mgr.registry.require(model_id)
            if model_id in mgr._loaded:
                return mgr._loaded[model_id]
            handle = mgr.backends[entry.backend].load(entry, str(tmp_path), Placement("cpu", None))
            loaded = LoadedModel(entry=entry, handle=handle, placement=Placement("cpu", None))
            mgr._loaded[model_id] = loaded
            return loaded

        mgr.load = load_no_place  # type: ignore[method-assign]
        yield c

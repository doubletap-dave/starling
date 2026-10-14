from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from starling.api.app import create_app
from starling.backends.base import TranscriptResult
from starling.config import Settings
from starling.gpu.placer import Placement
from starling.registry.catalog import ModelEntry


class FakeWhisper:
    name = "whisper"

    def load(self, entry, local_path, placement):
        return {"ok": True}

    def unload(self, handle):
        pass

    def transcribe(self, handle, audio, **kwargs):
        assert isinstance(audio, np.ndarray)
        return TranscriptResult(text="hello from fake", language=kwargs.get("language") or "en", duration=1.0)


@pytest.fixture
def client(monkeypatch, tmp_path):
    settings = Settings(
        registry_path=Path(__file__).resolve().parents[1] / "config" / "models.yaml",
        stt_model_ttl=-1,
        api_key=None,
        preload_models=[],
        metrics_enabled=True,
    )
    app = create_app(settings)

    # Patch manager backends after lifespan would run — use TestClient context
    with TestClient(app) as c:
        # Replace backends with fake; stub download
        mgr = c.app.state.manager
        mgr.backends = {"whisper": FakeWhisper(), "transformers": FakeWhisper(), "nemo": FakeWhisper()}

        def fake_download(entry: ModelEntry) -> str:
            return str(tmp_path / "model")

        def fake_is_downloaded(entry: ModelEntry) -> bool:
            return True

        mgr.registry.download = fake_download  # type: ignore[method-assign]
        mgr.registry.is_downloaded = fake_is_downloaded  # type: ignore[method-assign]
        mgr.registry.local_path = lambda e: str(tmp_path / "model")  # type: ignore[method-assign]

        # Avoid real load path issues
        orig_load = mgr.load

        def load_no_place(model_id: str):
            entry = mgr.registry.require(model_id)
            from starling.lifecycle.manager import LoadedModel

            if model_id in mgr._loaded:
                return mgr._loaded[model_id]
            handle = mgr.backends[entry.backend].load(entry, str(tmp_path), Placement("cpu", None))
            loaded = LoadedModel(entry=entry, handle=handle, placement=Placement("cpu", None))
            mgr._loaded[model_id] = loaded
            return loaded

        mgr.load = load_no_place  # type: ignore[method-assign]
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_registry(client):
    r = client.get("/v1/registry")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "whisper-turbo" in ids
    assert "cohere-transcribe-2b" in ids


def test_cohere_requires_language(client):
    # minimal wav header-ish bytes won't decode — use silence via soundfile
    import io
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    buf.seek(0)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", buf, "audio/wav")},
        data={"model": "cohere-transcribe-2b"},
    )
    assert r.status_code == 400
    assert "language" in r.json()["detail"].lower()


def test_transcription_ok(client):
    import io
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    buf.seek(0)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", buf, "audio/wav")},
        data={"model": "whisper-tiny", "response_format": "json"},
    )
    assert r.status_code == 200
    assert r.json()["text"] == "hello from fake"

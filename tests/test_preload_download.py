from pathlib import Path

from fastapi.testclient import TestClient

from starling.api.app import create_app
from starling.config import Settings, preload_models_from_env


def test_default_preload_is_orukeet():
    assert Settings.model_fields["preload_models"].default_factory() == ["orukeet"]


def test_preload_models_from_env_parses():
    assert preload_models_from_env('["orukeet"]') == ["orukeet"]
    assert preload_models_from_env("orukeet, whisper-tiny") == ["orukeet", "whisper-tiny"]
    assert preload_models_from_env([]) == []


def test_startup_downloads_without_loading(tmp_path):
    registry_src = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")

    downloaded: list[str] = []

    settings = Settings(
        registry_path=registry_path,
        runtime_config_path=tmp_path / "runtime.yaml",
        stt_model_ttl=-1,
        preload_models=["orukeet"],
        device_override="cpu",
        enable_ui=False,
        metrics_enabled=False,
    )
    app = create_app(settings)

    # Patch registry before lifespan runs
    from starling.registry.catalog import ModelRegistry

    orig_init = ModelRegistry.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.is_downloaded = lambda e: False  # type: ignore[method-assign]
        self.download = lambda e, job_id=None: downloaded.append(e.id) or str(tmp_path / e.id)  # type: ignore

    ModelRegistry.__init__ = patched_init  # type: ignore[method-assign]
    try:
        with TestClient(app) as client:
            assert downloaded == ["orukeet"]
            assert client.app.state.manager.loaded_ids() == []
    finally:
        ModelRegistry.__init__ = orig_init  # type: ignore[method-assign]

import time

from starling.backends.base import TranscriptResult
from starling.gpu.placer import GpuPlacer, Placement
from starling.lifecycle.manager import ModelManager
from starling.registry.catalog import ModelEntry, ModelRegistry


class FakeBackend:
    name = "whisper"

    def __init__(self):
        self.loaded = 0
        self.unloaded = 0

    def load(self, entry, local_path, placement):
        self.loaded += 1
        return object()

    def unload(self, handle):
        self.unloaded += 1

    def transcribe(self, handle, audio, **kwargs):
        return TranscriptResult(text="ok")


def test_ttl_unload(tmp_path):
    # Minimal registry yaml
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        """
models:
  m1:
    backend: whisper
    repo: fake/repo
    vram_mb: 100
""",
        encoding="utf-8",
    )
    registry = ModelRegistry(yaml_path)
    registry.is_downloaded = lambda e: True  # type: ignore[method-assign]
    registry.download = lambda e: str(tmp_path)  # type: ignore[method-assign]
    registry.local_path = lambda e: str(tmp_path)  # type: ignore[method-assign]

    backend = FakeBackend()
    placer = GpuPlacer(device_override="cpu")
    mgr = ModelManager(registry, {"whisper": backend}, placer, ttl_seconds=1)

    with mgr.use("m1"):
        assert mgr.is_loaded("m1")
        assert backend.loaded == 1

    # Wait for TTL
    deadline = time.time() + 3
    while time.time() < deadline and mgr.is_loaded("m1"):
        time.sleep(0.1)
    assert not mgr.is_loaded("m1")
    assert backend.unloaded == 1

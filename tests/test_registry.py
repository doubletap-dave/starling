from pathlib import Path

from starling.registry.catalog import ModelRegistry


def test_registry_loads_yaml():
    path = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    reg = ModelRegistry(path)
    assert "whisper-turbo" in reg.ids()
    assert "orukeet" in reg.ids()
    assert "cohere-transcribe-2b" in reg.ids()
    turbo = reg.require("whisper-turbo")
    assert turbo.repo == "deepdml/faster-whisper-large-v3-turbo-ct2"
    assert turbo.supports_translate is False
    orukeet = reg.require("orukeet")
    assert orukeet.revision == "555136b50265a132d4cea0d35560c26fc4f657ab"
    cohere = reg.require("cohere-transcribe-2b")
    assert cohere.requires_language is True
    assert cohere.gated is True

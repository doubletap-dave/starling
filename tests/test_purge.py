from pathlib import Path

import yaml

from starling.registry.catalog import ModelEntry, ModelRegistry


def _write_registry(path: Path, models: dict) -> None:
    path.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")


def test_purge_removes_hub_repo_dir(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    hub = cache / "hub"
    repo_dir = hub / "models--Systran--faster-whisper-tiny"
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "weights").write_bytes(b"x" * 2048)

    reg_path = tmp_path / "models.yaml"
    _write_registry(
        reg_path,
        {
            "whisper-tiny": {
                "backend": "whisper",
                "repo": "Systran/faster-whisper-tiny",
                "vram_mb": 1024,
            }
        },
    )
    registry = ModelRegistry(reg_path, cache_dir=cache)

    # Force fallback path (no usable scan result)
    monkeypatch.setattr(
        "starling.registry.catalog.scan_cache_dir",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no scan")),
    )

    entry = registry.require("whisper-tiny")
    info = registry.purge(entry)
    assert info["method"] == "rmtree"
    assert info["freed_bytes"] >= 2048
    assert not repo_dir.exists()


def test_purge_hf_cache_revisions(tmp_path, monkeypatch):
    class FakeRev:
        def __init__(self, commit_hash: str, refs=()):
            self.commit_hash = commit_hash
            self.refs = frozenset(refs)

    class FakeRepo:
        def __init__(self):
            self.repo_id = "oruk/orukeet"
            self.revisions = [FakeRev("555136b50265a132d4cea0d35560c26fc4f657ab", refs=("main",))]

    class FakeStrategy:
        expected_freed_size = 4096

        def execute(self):
            self.executed = True

    class FakeCache:
        def __init__(self):
            self.repos = [FakeRepo()]
            self.strategy = FakeStrategy()

        def delete_revisions(self, *hashes):
            assert "555136b50265a132d4cea0d35560c26fc4f657ab" in hashes
            return self.strategy

    monkeypatch.setattr("starling.registry.catalog.scan_cache_dir", lambda *_a, **_k: FakeCache())

    reg_path = tmp_path / "models.yaml"
    _write_registry(
        reg_path,
        {
            "orukeet": {
                "backend": "nemo",
                "repo": "oruk/orukeet",
                "file": "orukeet-v0.1.0.nemo",
                "revision": "555136b50265a132d4cea0d35560c26fc4f657ab",
                "vram_mb": 4096,
            }
        },
    )
    registry = ModelRegistry(reg_path, cache_dir=tmp_path / "cache")
    info = registry.purge(registry.require("orukeet"))
    assert info["method"] == "hf_cache"
    assert info["revisions_deleted"] == 1
    assert info["freed_bytes"] == 4096


def test_purge_api_updates_downloaded_flag(client, tmp_path, monkeypatch):
    registry = client.app.state.registry
    entry = ModelEntry(
        id="whisper-tiny",
        backend="whisper",
        repo="Systran/faster-whisper-tiny",
        vram_mb=1024,
    )

    calls = {"n": 0}

    def fake_purge(e):
        calls["n"] += 1
        return {"method": "hf_cache", "freed_bytes": 100, "revisions_deleted": 1, "repo": e.repo}

    # After purge, both still_downloaded and _entry_view should see disk gone
    states = iter([False, False])

    monkeypatch.setattr(registry, "purge", fake_purge)
    monkeypatch.setattr(registry, "is_downloaded", lambda e: next(states, False))
    monkeypatch.setattr(registry, "require", lambda mid: entry)

    r = client.delete("/v1/models/whisper-tiny?purge=true")
    assert r.status_code == 200
    body = r.json()
    assert body["purged"] is True
    assert body["still_downloaded"] is False
    assert body["downloaded"] is False
    assert calls["n"] == 1

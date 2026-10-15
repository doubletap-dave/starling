def test_registry_create_update_delete(client):
    r = client.post(
        "/v1/registry",
        json={
            "id": "custom-tiny",
            "backend": "whisper",
            "repo": "Systran/faster-whisper-tiny",
            "vram_mb": 1024,
            "supports_timestamps": True,
        },
    )
    assert r.status_code == 200
    assert r.json()["id"] == "custom-tiny"

    # Persisted in yaml copy
    yaml_text = client.app.state.registry.path.read_text(encoding="utf-8")
    assert "custom-tiny" in yaml_text

    r = client.put(
        "/v1/registry/custom-tiny",
        json={"vram_mb": 2048, "gated": True},
    )
    assert r.status_code == 200
    assert r.json()["vram_mb"] == 2048
    assert r.json()["gated"] is True

    r = client.delete("/v1/registry/custom-tiny")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    r = client.get("/v1/registry")
    ids = {m["id"] for m in r.json()["data"]}
    assert "custom-tiny" not in ids


def test_registry_nemo_requires_file(client):
    r = client.post(
        "/v1/registry",
        json={
            "id": "bad-nemo",
            "backend": "nemo",
            "repo": "nvidia/parakeet-tdt-0.6b-v3",
        },
    )
    assert r.status_code == 400


def test_cannot_delete_loaded_registry_entry(client):
    client.post("/v1/models/whisper-tiny/load")
    r = client.delete("/v1/registry/whisper-tiny")
    assert r.status_code == 409

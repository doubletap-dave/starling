def test_admin_status(client):
    r = client.get("/v1/admin/status")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body
    assert "whisper" in body["backends"]
    assert "devices" in body
    assert all("uuid" in d for d in body["devices"])


def test_settings_mask_and_persist(client, tmp_path):
    r = client.get("/v1/admin/settings")
    assert r.status_code == 200
    assert "hf_token" not in r.json()
    assert r.json()["hf_token_set"] is False

    r = client.patch(
        "/v1/admin/settings",
        json={
            "stt_model_ttl": 120,
            "device_override": "cuda:0",
            "hf_token": "secret-token",
            "preload_models": ["whisper-tiny"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["stt_model_ttl"] == 120
    assert body["device_override"] == "cuda:0"
    assert body["hf_token_set"] is True
    assert "hf_token" not in body
    assert body["preload_models"] == ["whisper-tiny"]

    # Overlay file written
    runtime = client.app.state.settings.runtime_config_path
    assert runtime.exists()
    text = runtime.read_text(encoding="utf-8")
    assert "stt_model_ttl" in text
    assert "secret-token" in text  # stored on disk; never returned by API

    # Clear token
    r = client.patch("/v1/admin/settings", json={"hf_token": ""})
    assert r.status_code == 200
    assert r.json()["hf_token_set"] is False


def test_model_load_unload_detail(client):
    r = client.post("/v1/models/whisper-tiny/load")
    assert r.status_code == 200
    assert r.json()["loaded"] is True
    assert r.json()["id"] == "whisper-tiny"

    r = client.get("/v1/models/whisper-tiny")
    assert r.status_code == 200
    assert r.json()["loaded"] is True
    assert "placement" in r.json()

    r = client.post("/v1/models/whisper-tiny/unload")
    assert r.status_code == 200
    assert r.json()["unloaded"] is True
    assert r.json()["loaded"] is False

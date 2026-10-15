import io

import numpy as np
import soundfile as sf


def _wav_bytes():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    buf.seek(0)
    return buf


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
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"model": "cohere-transcribe-2b"},
    )
    assert r.status_code == 400
    assert "language" in r.json()["detail"].lower()


def test_transcription_ok(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"model": "whisper-tiny", "response_format": "json"},
    )
    assert r.status_code == 200
    assert r.json()["text"] == "hello from fake"


def test_ui_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Starling" in r.text
    assert "/static/app.js" in r.text
    assert 'id="activity"' in r.text

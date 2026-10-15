# Starling

Universal OpenAI-compatible speech-to-text server.

One endpoint for **Whisper** (faster-whisper), **NeMo** (Orukeet / Parakeet / Canary), and **Transformers** (Cohere Transcribe, Qwen3-ASR, Moonshine). Ollama-style lazy model download, TTL VRAM eviction, VRAM-aware multi-GPU placement.

Replaces `orukeet-server`.

## Quick start

```bash
# Python 3.12 required
uv sync --extra nemo --extra dev
uv run starling
# or: uv run uvicorn starling.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Open the admin UI at **http://localhost:8000/** — models, registry editor, settings, status, and a try-transcription form. The UI only talks to the HTTP API (paste your `API_KEY` in the bar if auth is enabled).

Whisper-only (skip heavy NeMo layer):

```bash
uv sync --extra dev
```

### Transcribe

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-tiny
```

First request downloads the model from HuggingFace. Pre-warm / load:

```bash
curl -X POST http://localhost:8000/v1/models/whisper-tiny
curl -X POST http://localhost:8000/v1/models/whisper-tiny/load
```

### OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-tiny", file=f))
```

## API

| Method | Path | Notes |
|--------|------|-------|
| GET | `/` | Admin UI (public shell; APIs still auth-gated) |
| GET | `/health` | Liveness (public) |
| GET | `/metrics` | Prometheus |
| GET | `/v1/admin/status` | Devices, loaded models, backends |
| GET/PATCH | `/v1/admin/settings` | Runtime settings (`hf_token` write-only / masked) |
| GET | `/v1/registry` | Full catalog + downloaded/loaded flags |
| POST | `/v1/registry` | Create catalog entry → `models.yaml` |
| PUT | `/v1/registry/{id}` | Update entry |
| DELETE | `/v1/registry/{id}` | Remove entry (`?purge=true` also drops disk file when possible) |
| GET | `/v1/models` | Downloaded models (OpenAI list) |
| GET | `/v1/models/{id}` | Model detail + placement if loaded |
| POST | `/v1/models/{id}` | Explicit download |
| POST | `/v1/models/{id}/load` | Load into VRAM |
| POST | `/v1/models/{id}/unload` | Unload from VRAM |
| DELETE | `/v1/models/{id}?purge=true` | Unload (+ optional cache purge) |
| POST | `/v1/audio/transcriptions` | STT (lazy download) |
| POST | `/v1/audio/translations` | Whisper translate-capable models only |

Set `API_KEY` to require `Authorization: Bearer …` on protected routes.

Runtime overrides from the UI/API persist to `config/runtime.yaml` (gitignored). Env remains the bootstrap source. `API_KEY` / bind host-port are not editable via the UI.

## Models

See [`config/models.yaml`](config/models.yaml) (also editable via UI/API). Notable pins:

- `whisper-turbo` → `deepdml/faster-whisper-large-v3-turbo-ct2`
- `orukeet` → revision `555136b50265a132d4cea0d35560c26fc4f657ab`
- `cohere-transcribe-2b` is gated — accept the license and set `HF_TOKEN`; `language` is required

## GPU

Starling discovers CUDA devices and places loads on the GPU with the most free VRAM. Different models can sit on different GPUs concurrently. Transformers models that exceed a single card may use `device_map=auto`. Override with `DEVICE_OVERRIDE=cuda:0` (or `cpu`).

## Config

| Env | Default | Meaning |
|-----|---------|---------|
| `STT_MODEL_TTL` | `300` | Idle unload seconds (`-1` never, `0` immediate) |
| `API_KEY` | unset | Bearer auth |
| `HF_TOKEN` | unset | Gated/private Hub repos |
| `HF_HUB_CACHE` | Hub default | Cache directory |
| `PRELOAD_MODELS` | `[]` | JSON list of model ids |
| `DEVICE_OVERRIDE` | unset | `cpu` / `cuda:N` / `auto` |
| `WHISPER_COMPUTE_TYPE` | `default` | CT2 compute type |
| `RUNTIME_CONFIG_PATH` | `config/runtime.yaml` | Mutable settings overlay |
| `ENABLE_UI` | `true` | Serve admin UI at `/` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind |

## Docker

```bash
docker build -t starling:latest .
docker run --gpus all -p 8000:8000 \
  -v starling-hf:/cache/huggingface \
  -e HF_TOKEN \
  starling:latest
```

Recommend a **50 GB** volume for the full model set (~22 GB + overhead).

## Holocron cutover

1. Deploy Starling alongside orukeet-server.
2. Verify Whisper + NeMo (orukeet/parakeet) + Transformers (cohere) against your clients.
3. Point clients at Starling’s OpenAI base URL.
4. Shut down orukeet-server.

## License notes

Model weights have their own licenses (MIT, CC-BY-4.0, CC-BY-SA-4.0 for Orukeet, Apache-2.0 for Cohere/Qwen). Server code: see repository license.

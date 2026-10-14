#projects/ideas

# Starling — Universal STT Server

**Status:** Implemented (v0.1.0 scaffold + three backends) — deploy/cutover pending on holocron
**Created:** 2026-10-14
**Goal:** A single self-hosted, OpenAI-compatible speech-to-text server supporting everything speaches does PLUS Orukeet, NVIDIA Parakeet, and Cohere Transcribe — with on-demand model downloading (Ollama-style UX).

**Name:** Starling — named after starlings, the scrappy mimic birds that copy car alarms, ringtones, and other birds. (Dave's pick, 2026-10-14.)

---

## Why

Speaches is Whisper-only for STT. It can't run Orukeet (NeMo), Parakeet (NeMo), or Cohere (transformers). Dave uses STT heavily and wants one endpoint for everything — no more juggling separate servers per model family.

---

## Model Matrix

| Model | HF Repo | Size | Runtime | Langs | Notes |
|---|---|---|---|---|---|
| Whisper large-v3-turbo | `Systran/faster-whisper-large-v3-turbo` | 1.6 GB | faster-whisper (CTranslate2) | 99 (+auto-detect) | Sweet spot: near-large-v3 quality, much faster |
| Whisper medium | `Systran/faster-whisper-medium` | 1.5 GB | faster-whisper | 99 (+auto-detect) | Solid default |
| Whisper small/base/tiny | `Systran/faster-whisper-*` | 75MB–470MB | faster-whisper | 99 | Smoke test tier |
| Orukeet v0.1.0 | `oruk/orukeet` | 2.5 GB | NeMo 3.0.0 | 25 (EU) | Our current server; 1.46% WER LibriSpeech clean |
| Parakeet TDT 0.6B v3 | `nvidia/parakeet-tdt-0.6b-v3` | 2.4 GB | NeMo 3.0.0 | 25 (EU, auto-detect) | CC-BY-4.0; built-in timestamps |
| Parakeet TDT 1.1B | `nvidia/parakeet-tdt-1.1b` | ~4 GB | NeMo 3.0.0 | EN | Larger English model |
| Cohere Transcribe 2B | `CohereLabs/cohere-transcribe-03-2026` | 4.1 GB | transformers >=5.4.0 | 14 (must specify, NO auto-detect) | Apache 2.0; 5.42% avg WER (#1 on HF leaderboard) |

**Key insight:** Orukeet and all Parakeet models load through the same NeMo `ASRModel.from_pretrained()` call. One NeMo backend covers both. So we need exactly three backends:
1. **WhisperBackend** (faster-whisper/CTranslate2)
2. **NemoBackend** (NeMo 3.0.0 — covers Orukeet + entire Parakeet family)
3. **CohereBackend** (transformers — covers Cohere Transcribe 2B)

See also: [[STT Models Technical Reference]] for the full deep-dive on every model.

---

## Architecture Recommendation

### Single FastAPI process, backend registry (mirror speaches' executor pattern)

```
POST /v1/audio/transcriptions {model: "orukeet"}
  → registry lookup: "orukeet" → {backend: "nemo", hf_repo: "oruk/orukeet", ...}
  → if not on disk: hf_hub_download() (first request is slow, stream progress)
  → if not in VRAM: backend.load(local_path)
  → transcribe → return JSON
  → TTL timer starts: unload after N idle seconds (frees VRAM, keeps disk cache)
```

**API surface** (OpenAI-compatible + speaches extensions):
- `GET /health` — liveness
- `GET /v1/models` — list downloaded models
- `POST /v1/models/{id}` — explicit download (pre-warming)
- `DELETE /v1/models/{id}` — unload/remove
- `GET /v1/registry` — list all supported remote models
- `POST /v1/audio/transcriptions` — STT (lazy-downloads on first use)

**On-demand downloading:** Copy Ollama's UX. First request for a model triggers download from HuggingFace Hub (resumable, cached in `~/.cache/huggingface/hub`). Subsequent requests are instant. Explicit `POST /v1/models/{id}` for pre-warming. Pin revisions per model for reproducibility.

**TTL eviction:** Models unload from VRAM after idle timeout (default 5 min, configurable). Disk cache persists. This is critical — we can't fit all models in VRAM simultaneously.

### GPU placement (holocron: RTX A2000 12GB + RTX 4060 8GB)

| Backend | Model | VRAM | Placement |
|---|---|---|---|
| NemoBackend | Orukeet / Parakeet | ~4-5 GB each | A2000 (don't co-reside by default) |
| CohereBackend | Cohere 2B | ~5-7 GB | A2000 |
| WhisperBackend | large-v3-turbo | ~6 GB | 4060 or A2000 |

**Rule:** Never load two large models on the 4060 simultaneously. A2000 is the workhorse (12GB). Use `CUDA_VISIBLE_DEVICES` per backend.

### Disk budget

Full model set ≈ 12 GB + HF overhead. Recommend 50 GB volume. 100 GB if adding more Parakeet variants.

---

## Gotchas & Notes

- **Cohere has no auto language detection.** Must pass `language="en"` etc. API should require/expose a `language` param for Cohere models.
- **Cohere needs `AutoModelForSpeechSeq2Seq`**, not plain `AutoModel` (which returns encoder only).
- **NeMo 3.0.0 is dependency-heavy** (~8-10 GB Docker layer). Consider separate Docker stages or a pre-built base image.
- **faster-whisper needs 16 kHz mono input.** Normalize with librosa (we already do this in orukeet-server).
- **Systran HF org is `Systran`**, not `deepdml` (the community mirror is `deepdml/faster-whisper-large-v3-turbo-ct2`).
- **Revision pinning:** Pin `revision=<commit-sha>` per model in the registry (like orukeet-server does with `555136b...`).

---

## Build Plan (if approved)

### Phase 1: Core server + WhisperBackend
- FastAPI skeleton with OpenAI-compatible endpoints
- Model registry (YAML or Python dict)
- WhisperBackend with faster-whisper
- On-demand download + TTL eviction
- Docker image, deploy on holocron

### Phase 2: NemoBackend
- Add NeMo 3.0.0 dependency
- NemoBackend supporting Orukeet + Parakeet via `ASRModel.from_pretrained`
- GPU placement config

### Phase 3: CohereBackend
- Add transformers dependency
- CohereBackend with language param handling
- Test with 14 supported languages

### Phase 4: Polish
- `GET /v1/registry` with full model catalog
- Prometheus metrics (inference latency, VRAM usage, cache hits)
- Kuma monitor + Grafana dashboard
- Vault docs

### Open questions
- [ ] Publish to Docker Hub as public image like orukeet-server? (namespace: topiary)
- [ ] Should this replace orukeet-server, or live alongside it?
- [ ] TTS support? (Dave said no — STT only)

---

## Links

- [[Holocron - LLM Server]]
- [[Orukeet Server]]
- [[STT Models Technical Reference]]
- Speaches: https://github.com/speaches-ai/speaches
- Orukeet: https://huggingface.co/oruk/orukeet
- Parakeet: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Cohere: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026

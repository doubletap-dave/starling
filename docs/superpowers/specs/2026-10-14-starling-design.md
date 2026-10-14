# Starling Universal STT Server — Design Spec

**Date:** 2026-10-14  
**Status:** Approved  
**Goal:** Self-hosted, OpenAI-compatible speech-to-text server supporting Whisper, NeMo (Orukeet/Parakeet/Canary), and Transformers (Cohere/Qwen3-ASR/Moonshine), with Ollama-style lazy download and TTL VRAM eviction. Replaces orukeet-server.

## Decisions

- Greenfield FastAPI (speaches ideas only; no speaches code)
- Single process, three backends, static pinned YAML registry
- Max OpenAI Audio API client compatibility; Starling extras additive
- Flexible multi-GPU: place by free VRAM; concurrent models on different GPUs; Transformers `device_map="auto"` when needed
- STT only — no TTS / Realtime / Gradio in v1
- Tooling: Python 3.12+, uv, latest mutually compatible package versions

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/v1/registry` | Full catalog + flags |
| GET | `/v1/models` | Downloaded models (OpenAI list) |
| POST | `/v1/models/{id}` | Explicit download |
| DELETE | `/v1/models/{id}` | Unload; `?purge=true` deletes cache |
| POST | `/v1/audio/transcriptions` | STT (lazy download) |
| POST | `/v1/audio/translations` | Whisper translate-capable only |

Auth: optional `API_KEY` Bearer. `/health` public.

## Lifecycle

registry → download if missing → VRAM-aware load → transcribe → TTL reset (default 300s).  
Audio normalized once to 16 kHz mono float32.  
`requires_language` models return 400 without `language`.

## Backends

1. **Whisper** — faster-whisper / CTranslate2  
2. **NeMo** — NeMo 3.0 `hf_hub_download` + `ASRModel.restore_from`  
3. **Transformers** — `AutoModelForSpeechSeq2Seq`, transformers ≥5.4  

## Out of v1

TTS, Realtime WS, Gradio, whisper.cpp, Hub-live registry, Docker Hub publish.

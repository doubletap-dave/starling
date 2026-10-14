# Universal STT Server — Deep Technical Reference

**Compiled:** 2026-10-14
**Purpose:** Technical foundation for the "speaches alternative" project — a self-hosted, OpenAI-compatible speech-to-text server supporting Whisper, Orukeet, Parakeet, Cohere, and other models, with on-demand downloading.

**Research sources:** speaches-ai/speaches (master @ 993994f), NVIDIA-NeMo/Speech 3.0, HuggingFace Hub, HF Open ASR Leaderboard, model cards, arXiv papers.

---

## Table of Contents

1. [Speaches — Deep Dive](#1-speaches--deep-dive)
2. [NVIDIA NeMo — Framework Deep Dive](#2-nvidia-nemo--framework-deep-dive)
3. [HuggingFace Transformers for ASR](#3-huggingface-transformers-for-asr)
4. [Model Reference — Whisper Family](#4-model-reference--whisper-family)
5. [Model Reference — Orukeet](#5-model-reference--orukeet)
6. [Model Reference — NVIDIA Parakeet Family](#6-model-reference--nvidia-parakeet-family)
7. [Model Reference — Cohere Transcribe 2B](#7-model-reference--cohere-transcribe-2b)
8. [Other Models Worth Supporting](#8-other-models-worth-supporting)
9. [Technical Comparison Matrix](#9-technical-comparison-matrix)
10. [Recommended Backend Architecture](#10-recommended-backend-architecture)
11. [Gotchas and Warnings](#11-gotchas-and-warnings)

---

## 1. Speaches — Deep Dive

**Repo:** github.com/speaches-ai/speaches | **License:** MIT | **Stars:** 3,716 / **Forks:** 460
**Latest stable:** v0.8.3 (v0.9.0 in RC) | **Last commit:** 2026-04-17 (~6 months quiet as of Oct 2026)
**Stack:** Python, FastAPI + uvicorn, `uv`-managed | **Tagline:** "Ollama, but for TTS/STT models"

### 1.1 Architecture: The Executor Pattern

The core abstraction (introduced Oct 2025):

```python
class Executor(BaseModel):
    name: str                   # "whisper", "parakeet", "kokoro", "piper", ...
    model_manager: ManagerT     # load/inference/unload lifecycle
    model_registry: RegistryT   # HF discovery + download
    task: ModelTask             # "automatic-speech-recognition", "text-to-speech", ...
```

Seven executors registered in a singleton `ExecutorRegistry`:

| Executor | Manager | Task | TTL env var |
|---|---|---|---|
| `whisper` | `WhisperModelManager` | automatic-speech-recognition | `stt_model_ttl` (300s) |
| `parakeet` | `ParakeetModelManager` | automatic-speech-recognition | `stt_model_ttl` (300s) |
| `piper` | `PiperModelManager` | text-to-speech | `tts_model_ttl` (300s) |
| `kokoro` | `KokoroModelManager` | text-to-speech | `tts_model_ttl` (300s) |
| `wespeaker-speaker-embedding` | speaker embedding | — | hardcoded 0 |
| `pyannote-diarization` | speaker diarization | — | hardcoded -1 |
| `vad` | `SileroVADModelManager` | voice-activity-detection | `vad_model_ttl` (-1) |

**Routing:** task-scoped executor tuples + live HF model-card filter matching (`HfModelFilter`: model_name substring / library_name / task / tags). For transcription, whisper is checked before parakeet.

**This executor pattern is directly reusable for our build** — it's clean, proven, and extensible.

### 1.2 STT Models Supported

Exactly **two STT executors**. There is **no static model list** — `GET /v1/registry` queries the HF Hub live.

**faster-whisper executor** (`library_name="ctranslate2"`, task ASR):
- Every CTranslate2-format Whisper/ASR model on the HF Hub
- Official: `Systran/faster-whisper-{tiny,base,small,medium,large-v1,v2,v3,large-v3-turbo}` (+ `.en` variants), `Systran/faster-distil-whisper-*`
- Community fine-tunes: `ivrit-ai/whisper-large-v3-turbo-ct2`, `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, etc.
- Full OpenAI surface: json/verbose_json/text/srt/vtt, word timestamps, SSE streaming, hotwords, translations endpoint

**Parakeet executor** (added May 2025, v2/v3 Sep 2025):
- Filter: `model_name` contains "istupakov/parakeet-tdt" (istupakov's ONNX exports)
- Runtime: **`onnx_asr`** (ONNX Runtime) — **NOT NeMo toolkit**
- Limitations: text/JSON only, no streaming (`NotImplementedError`), no translation, English-only, no quantization

**What speaches does NOT support:** No transformers-based ASR (no Cohere, no Canary), no `.nemo` checkpoints, no Whisper.cpp/GGUF, no Zipformer, no streaming-oriented models.

### 1.3 API Surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness: `200 {"message": "OK"}` (public) |
| `POST` | `/v1/audio/transcriptions` | STT. Multipart: `file`, `model`, `language`, `prompt`, `response_format` (text/json/verbose_json/srt/vtt), `temperature`, `timestamp_granularities[]`, `stream`, `hotwords` |
| `POST` | `/v1/audio/translations` | Whisper → English translation (whisper executor only) |
| `GET` | `/v1/models[?task=]` | Local (downloaded) models, OpenAI list shape |
| `GET/POST/DELETE` | `/v1/models/{model_id}` | Model detail / **download** (200=downloaded, 201=exists, 401=gated, 404=unknown) / delete from cache |
| `GET` | `/v1/registry[?task=]` | Remote (Hub) models across all executors |
| `GET/POST/DELETE` | `/api/ps[/{model_id}]` | (Experimental) Ollama-style loaded-model management |
| `POST` | `/v1/audio/speech` | TTS (Kokoro/Piper) |
| `POST` | `/v1/audio/speech/timestamps` | VAD (Silero) |
| `POST` | `/v1/audio/speech/embedding` | Speaker embedding (WeSpeaker) |
| `POST` | `/v1/audio/diarization` | Diarization (Pyannote, needs HF_TOKEN) |
| `POST` | `/v1/chat/completions` | Proxies to Ollama-compatible backend |
| `WS`/`POST` | `/v1/realtime` | OpenAI Realtime API |

**Auth:** Single global `API_KEY` (Bearer) or nothing. `/health`, `/docs`, `/openapi.json`, UI stay public.

### 1.4 Configuration (Key Env Vars)

| Env var | Default | Meaning |
|---|---|---|
| `STT_MODEL_TTL` | `300` | Seconds idle before STT model unloads. `-1`=never, `0`=immediate |
| `TTS_MODEL_TTL` | `300` | Same for TTS |
| `VAD_MODEL_TTL` | `-1` | Same for Silero VAD (never unload) |
| `WHISPER__INFERENCE_DEVICE` | `auto` | cpu/cuda/auto |
| `WHISPER__DEVICE_INDEX` | `0` | GPU index |
| `WHISPER__COMPUTE_TYPE` | `default` | int8/int8_float16/float16/bfloat16/float32 |
| `WHISPER__CPU_THREADS` | `0` | CTranslate2 CPU threads |
| `PRELOAD_MODELS` | `[]` | JSON list of model IDs to download at startup (exits on failure) |
| `HF_HUB_CACHE` | `~/.cache/huggingface/hub` | HF cache location |
| `HF_TOKEN` | unset | For gated/private repos |
| `API_KEY` | unset | Bearer auth |
| `ENABLE_UI` | `true` | Gradio web UI at `/` |

### 1.5 Model Lifecycle

1. **Download — explicit only.** No lazy download on inference. Transcription of a non-downloaded model → **HTTP 404**: "Model '<id>' is not installed locally. You can download the model using POST /v1/models"
2. **Load — lazy on first use.** `SelfDisposingModel` context manager, ref-counted, thread-safe
3. **Inference.** Silero VAD runs on every request; segments passed as `clip_timestamps` to faster-whisper
4. **Unload — TTL eviction.** `threading.Timer(ttl)` when ref count hits zero. Weights stay in HF cache

### 1.6 Limitations and Gaps (What Our Build Must Fix)

1. **No lazy download** — hard 404 instead of Ollama-style auto-pull
2. **Only 2 STT runtimes** — no transformers, no NeMo toolkit, no GGUF
3. **Parakeet is second-class** — no streaming, no timestamps, no translation, English-only
4. **Hub-dependent registry** — fragile, requires network; local listing needs parseable model cards
5. **Mandatory opinionated VAD** — every request runs Silero VAD with fixed defaults
6. **Single-tenant, in-process** — no multi-GPU placement, no request queueing
7. **Maintenance risk** — 6 months quiet; STT core stable but low-maintenance

### 1.7 Notable Fork

**`godsquantum/speaches-reloaded`** — keeps speaches as the API + lifecycle layer, adds **transcribe.cpp (GGUF STT) as a second executor**. This is a preview of exactly the "second executor" pattern our build needs.

### 1.8 Repo Health

MIT license. 3,716 stars, 460 forks. Very active 2024-05 through 2025-02 (peak 128 commits Jan 2025), steady through 2025, tapering in 2026. Recent work almost entirely Realtime API. STT core unchanged since late 2025.

> **⚠️ HEADS-UP (from research):** `WHISPER__TTL` **never existed** upstream. The field was `TTL` pre-Sep-2025, now `STT_MODEL_TTL`. pydantic-settings silently ignores unknown env vars — Dave's `WHISPER__TTL=-1` had **no effect**. Correct var: `STT_MODEL_TTL=-1`.

---

## 2. NVIDIA NeMo — Framework Deep Dive

### 2.1 What is NeMo?

**NeMo** is NVIDIA's open-source (Apache 2.0) PyTorch toolkit for conversational AI: ASR, TTS, speaker tasks, audio processing, SpeechLM. It is a **library**, not a server — you call it from Python. Models are PyTorch Lightning modules under the hood.

**`nemo.collections.asr`** — the ASR collection. Every model follows encoder–decoder pattern:

| Component | Typical classes |
|---|---|
| Preprocessor | `AudioToMelSpectrogramPreprocessor` (log-mel) |
| Encoder | `FastConformerEncoder`, `ConformerEncoder`, `CitrinetEncoder` |
| Decoder | `ConvASRDecoder` (CTC), `RNNTDecoder`/TDT (transducer), `TransformerDecoder` (AED) |
| Tokenizer | SentencePiece BPE (bundled in checkpoint) |

### 2.2 ASR Model Classes

| Class | Decoder | Used by |
|---|---|---|
| `EncDecCTCModel` / `EncDecCTCModelBPE` | CTC | parakeet-ctc-0.6b/1.1b |
| **`EncDecRNNTBPEModel`** | RNN-Transducer | parakeet-rnnt-0.6b/1.1b, **ALL parakeet-tdt-* models**, orukeet |
| `EncDecHybridRNNTCTCBPEModel` | Transducer + CTC | parakeet-tdt_ctc-1.1b, parakeet-tdt_ctc-110m |
| `EncDecMultiTaskModel` | Encoder-decoder AED | Canary (transcribe + translate) |
| `ASRModel` | Abstract base | Auto-dispatch entry point |

**Key fact:** Parakeet TDT models are `EncDecRNNTBPEModel`. TDT (Token-and-Duration Transducer) is a transducer-decoder variant within the same class. **One NeMo backend class covers Orukeet + the entire Parakeet family.**

### 2.3 Model Loading

**`from_pretrained()`** — downloads from HF Hub automatically:
```python
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
# Auto-dispatches to the correct subclass based on checkpoint config
```

**`restore_from()`** — for local `.nemo` files (recommended for servers):
```python
from huggingface_hub import hf_hub_download
local_path = hf_hub_download(repo_id="nvidia/parakeet-tdt-0.6b-v3",
                             filename="parakeet-tdt-0.6b-v3.nemo")
model = nemo_asr.models.ASRModel.restore_from(restore_path=local_path)
```

> **NeMo 3.0 gotcha:** `from_pretrained` can fail on repos shipping a bare `.nemo` with no model metadata. `hf_hub_download` + `restore_from` always works. **Use this pattern for the server.**

**What is a `.nemo` file?** A tar archive containing:
- `model_config.yaml` — full model config
- `model_weights.ckpt` — PyTorch Lightning checkpoint
- Tokenizer files (SentencePiece `.model`)

### 2.4 Audio Preprocessing

- **Input must be 16 kHz, mono, float32** — NeMo does **NOT** resample or downmix for you
- Accepts: file paths (wav/flac), numpy arrays, torch tensors, or lists thereof
- Internal: STFT → 128-bin (Parakeet) log-mel → 8x conv subsampling → FastConformer

```python
import librosa
audio, sr = librosa.load("input.mp3", sr=None, mono=True)
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
audio = audio.astype(np.float32)
```

### 2.5 Transcription with Timestamps

```python
model.eval().to("cuda")
hyps = model.transcribe(["audio.wav"], timestamps=True)
h = hyps[0]
print(h.text)                                    # punctuation + capitalization baked in
word_stamps = h.timestamp["word"]                # [{'start': 0.12, 'end': 0.45, 'word': 'hello'}]
segment_stamps = h.timestamp["segment"]
```

**Timestamps are native to TDT models** — the decoder predicts token durations. No forced alignment needed. Free — always request them.

### 2.6 GPU Memory

| Model | Params | .nemo on disk | Est. VRAM FP32 | Est. VRAM FP16 |
|---|---|---|---|---|
| parakeet-tdt-0.6b-v3 | 627M | 2.51 GB | ~3.5–4.5 GB | ~2–3 GB |
| parakeet-ctc-0.6b | 609M | 2.44 GB | ~3.5–4.5 GB | ~2–3 GB |
| parakeet-rnnt-0.6b | 617M | 2.47 GB | ~3.5–4.5 GB | ~2–3 GB |
| orukeet | 627M | ~2.5 GB | ~3.5–4.5 GB | ~2–3 GB |
| parakeet-tdt-1.1b | 1.07B | 4.28 GB | ~5.5–6.5 GB | ~3–4 GB |
| parakeet-ctc-1.1b | 1.06B | 4.25 GB | ~5.5–6.5 GB | ~3–4 GB |
| parakeet-tdt_ctc-1.1b | ~1.07B | 4.29 GB | ~5.5–6.5 GB | ~3–4 GB |
| parakeet-tdt_ctc-110m | ~114M | ~0.46 GB | ~1–1.5 GB | <1 GB |

**FP16/BF16:** Use `torch.autocast` + `torch.inference_mode()`. Keep the mel frontend FP32 (autocast handles this). Enable `torch.backends.cuda.matmul.allow_tf32 = True` on Ampere+ for free throughput.

```python
model.eval().to("cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    hyps = model.transcribe([audio_np], batch_size=8)
```

**TTL eviction:**
```python
del model
torch.cuda.empty_cache()   # Clean — no daemon, no shared state
```

### 2.7 NeMo 3.0.0 vs Older

- **Repo split:** Speech-only at `NVIDIA-NeMo/Speech`; LLM/NLP/export moved to separate repos
- **Lighter install:** ~800k deprecated LOC removed; `pip install 'nemo-toolkit[asr]'` (PyTorch ≥ 2.7)
- **ASR inference API unchanged** from 2.x → 3.0: `from_pretrained()` / `restore_from()` / `transcribe()`
- **Export moved out** — `model.export()` may not exist in 3.0 pip install
- **Dependency footprint:** Still heavy — PyTorch + Lightning + librosa + transformers + sentencepiece + omegaconf. Expect **~8–10 GB Docker layer**

### 2.8 Export to ONNX/TensorRT

**Verdict: Not worth it for phase 1.**
- Encoder exports cleanly (static graph), but the transducer decoder is a Python loop — fiddly to trace
- Stock NeMo eager hits **~1,000x real-time** already (RTFx 3380 for v2 at batch 128)
- Community TensorRT path (fastconformer-trt) got 4.5x speedup but required custom encoder reimplementation
- Revisit only with measured latency data

### 2.9 Long Audio

Full attention is O(n²). Parakeet TDT v2/v3: ~24 min single-pass with full attention, ~3 hours with local attention:
```python
model.change_attention_model(self_attention_model="rel_pos_local_attn",
                             att_context_size=[256, 256])
```

---

## 3. HuggingFace Transformers for ASR

### 3.1 ASR Architectures in Transformers

**A. Encoder-decoder (seq2seq, autoregressive)** — accuracy leaders:
- `openai/whisper-{tiny,base,small,medium,large-v3,large-v3-turbo}` (MIT)
- `distil-whisper/distil-large-v2/v3` (MIT, ~6x faster)
- `UsefulSensors/moonshine-tiny` (27M), `moonshine-base` (61.5M) (MIT, RoPE, no 30s limit)
- `CohereLabs/cohere-transcribe-03-2026` (Apache 2.0, native in transformers ≥ 5.4.0)
- `Qwen/Qwen3-ASR-{0.6B,1.7B}-hf` (Apache 2.0, 52 languages)
- `mistralai/Voxtral-Mini-3B-2507` (Apache 2.0, audio LLM)

**B. Encoder-only + CTC** — faster, slightly less accurate:
- `facebook/wav2vec2-large-960h` (315M, MIT)
- `facebook/wav2vec2-xls-r-*` (CC-BY-NC 4.0 — **non-commercial**)
- `facebook/mms-1b-*` (CC-BY-NC 4.0 — **non-commercial**)

**C. Transducer/TDT/NAR — NOT in transformers.** Parakeet, Canary, Granite-NAR are NeMo-only. Don't force them into transformers.

### 3.2 Which Auto Class

| Class | For | Models |
|---|---|---|
| `AutoModelForSpeechSeq2Seq` | Encoder-decoder ASR | Whisper, Moonshine, Cohere, Qwen3-ASR |
| `AutoModelForCTC` | CTC ASR | Wav2Vec2, HuBERT, MMS |

Output via `.generate()` (seq2seq) or logits → argmax (CTC). The `automatic-speech-recognition` pipeline auto-selects but adds overhead — for a server, use `AutoModel` + `AutoProcessor` + manual chunking.

### 3.3 Audio Preprocessing

- **Log-mel models** (Whisper, Moonshine, Cohere): 16 kHz mono → feature extractor (Whisper: n_fft=400, hop=160, 80 mel bins, 30s windows)
- **CTC models**: raw 16 kHz waveform → normalization only
- `AutoProcessor.from_pretrained()` bundles feature extractor + tokenizer
- **Server tip:** normalize to 16 kHz mono once at the HTTP layer; per-model processors are cheap

### 3.4 Performance: Transformers vs Specialized Runtimes

**Rule of thumb: transformers is 2–5x slower than specialized runtimes.**

- faster-whisper (CTranslate2): up to **4x faster** than openai/whisper, same accuracy
- nano-cohere-transcribe (pure PyTorch): **1.5–3.6x faster** than transformers path (same weights, same WER)
- vLLM on Cohere: RTFx ~119 (production concurrent serving)

**Optimization levers:**
1. `attn_implementation="flash_attention_2"` or `"sdpa"` (Ampere+, fp16/bf16)
2. `torch_dtype=torch.float16` (~1.5–2x speedup, halves VRAM)
3. **Batched inference** — single biggest lever
4. `torch.compile` on the encoder
5. Static KV cache (`cache_implementation="static"` in transformers 5.x)
6. Energy-based chunking at quiet points for long-form
7. vLLM for production concurrency

**Bottom line:** transformers is the right universal backend for coverage. Keep faster-whisper (Whisper) and NeMo (Parakeet/Canary) as specialized backends where they win.

### 3.5 Version Requirements

- **Cohere Transcribe needs `transformers >= 5.4.0`** for native `cohere_asr` support (`trust_remote_code=False`)
- **transformers 5.0/5.1 have a weight-loading bug** — exclude them
- 4.52–4.57 works via remote code (`trust_remote_code=True`) but has O(n²) detokenization
- **Cohere weights are gated** — HF license acceptance + token required on every deploy
- **trust_remote_code posture:** prefer native; pin `revision=` for anything needing remote code

---

## 4. Model Reference — Whisper Family

All via **faster-whisper** (CTranslate2). VRAM = FP16 CUDA working set.

| Model | HF Repo ID | Params | Download | VRAM | Languages | WER (test-clean/test-other) |
|---|---|---|---|---|---|---|
| tiny | `Systran/faster-whisper-tiny` | 39M | 78 MB | ~1 GB | 99 | 6.3% / 15.9% |
| tiny.en | `Systran/faster-whisper-tiny.en` | 39M | 78 MB | ~1 GB | EN | — |
| base | `Systran/faster-whisper-base` | 74M | 148 MB | ~1 GB | 99 | 4.9% / 11.0% |
| base.en | `Systran/faster-whisper-base.en` | 74M | 148 MB | ~1 GB | EN | — |
| small | `Systran/faster-whisper-small` | 244M | 486 MB | ~2 GB | 99 | 3.3% / 7.7% |
| small.en | `Systran/faster-whisper-small.en` | 244M | 486 MB | ~2 GB | EN | — |
| medium | `Systran/faster-whisper-medium` | 769M | 1.5 GB | ~5 GB | 99 | 2.8% / 6.5% |
| medium.en | `Systran/faster-whisper-medium.en` | 769M | 1.5 GB | ~5 GB | EN | — |
| large-v1 | `Systran/faster-whisper-large-v1` | 1550M | ~2.9 GB | ~10 GB | 99 | baseline |
| large-v2 | `Systran/faster-whisper-large-v2` | 1550M | 3.1 GB | ~10 GB | 99 | 10–20% better than v1 |
| large-v3 | `Systran/faster-whisper-large-v3` | 1550M | 3.1 GB | ~10 GB | 99 | 10–20% better than v2 |
| **large-v3-turbo** | **`deepdml/faster-whisper-large-v3-turbo-ct2`** | 809M | ~1.6 GB | ~4–6 GB | 99 | ≈ large-v3, **8x faster decode** |
| distil-small.en | `Systran/faster-distil-whisper-small.en` | ~166M | ~330 MB | ~1 GB | EN | 5.8x faster, within 1% WER |
| distil-medium.en | `Systran/faster-distil-whisper-medium.en` | ~394M | ~760 MB | ~2 GB | EN | — |
| distil-large-v2 | `Systran/faster-distil-whisper-large-v2` | ~756M | ~1.5 GB | ~5 GB | EN | — |
| distil-large-v3 | `Systran/faster-distil-whisper-large-v3` | ~756M | ~1.5 GB | ~5 GB | EN | — |
| distil-large-v3.5 | `distil-whisper/distil-large-v3.5-ct2` | 756M | ~1.5 GB | ~5 GB | EN | Short-form OOD 7.08 (best distil) |

### Key Notes

- **large-v3-turbo** = large-v3 encoder + 4-layer decoder (vs 32). Sweet spot for the server. **NOT trained for translation** — returns source language even when `task=translate`.
- **⚠️ Turbo repo gotcha:** `Systran/faster-whisper-large-v3-turbo` **does not exist**. faster-whisper's built-in `turbo` alias resolves to `mobiuslabsgmbh/faster-whisper-large-v3-turbo` — that org was abandoned and re-claimed by an unverified account (2026-07). **Always pin `deepdml/faster-whisper-large-v3-turbo-ct2`.**
- **distil models:** 5.8x faster than large-v2, 51% fewer params, within 1% WER OOD, less prone to hallucination loops. English-only.
- **`.en` variants:** English-only, slightly more accurate on pure English.

### faster-whisper API

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cuda",
    compute_type="float16",  # float16 (Ampere+), int8_float16 (VRAM tight), int8 (CPU)
)

segments, info = model.transcribe(
    audio,                          # file path or np.ndarray float32 16kHz mono
    language="en",                  # None = auto-detect
    task="transcribe",              # or "translate" (not supported by turbo)
    beam_size=5,
    vad_filter=True,                # built-in Silero VAD
    vad_parameters=dict(threshold=0.5, min_silence_duration_ms=500),
    word_timestamps=True,
    hotwords="Orukeet Parakeet",    # boost specific terms
    temperature=0.0,
)
# segments is a generator — consume to drive inference
```

### Notable Community Fine-tunes

| Repo | What | Notes |
|---|---|---|
| `ivrit-ai/whisper-large-v3-turbo-ct2` | Hebrew fine-tune | 7.6% WER on Hebrew (vs 13.4% stock). **Set `language="he"` explicitly** — detection degraded in fine-tuning |
| `NbAiLab/nb-whisper-*` | Norwegian fine-tunes | Used as defaults by Norwegian speech projects |
| `nyrahealth/faster_CrisperWhisper` | Verbatim transcription | Preserves disfluencies, fillers, pause timing |

---

## 5. Model Reference — Orukeet

**HF Repo:** `oruk/orukeet` | **File:** `orukeet-v0.1.0.nemo` (2.5 GB)
**Revision:** `555136b50265a132d4cea0d35560c26fc4f657ab` (pin this) | **SHA256:** `031c8ddab484...`
**License:** **CC BY-SA 4.0** (copyleft — attribution + ShareAlike on redistribution; ships `LICENSE-WEIGHTS` + `NOTICE.md`)

### Architecture

NVIDIA **Parakeet TDT 0.6B v3** fine-tune by Oruk AI (Stanford/Cambridge/OpenWhispr contributors). Replaces half the encoder's temporal depthwise filters with **12,288 fitted, frozen Gabor kernels**; trains remaining parameters on multilingual + multi-accent data. FastConformer encoder + TDT decoder. ~600M params.

### Languages (25)

bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk

### Quality

Beats Parakeet TDT 0.6B v3 on **61 of 74 tested splits**:

| Benchmark | Parakeet v3 | Orukeet |
|---|---|---|
| LibriSpeech test-clean | 1.53% | **1.46%** |
| LibriSpeech test-other | 3.14% | **2.86%** |
| FLEURS English | 4.28% | **3.82%** |
| FLEURS pooled (25 langs) | 11.01% | **9.85%** |

**Standouts:** Czech 8.97 vs 11.12, Estonian 10.44 vs 13.32, Latvian 17.41 vs 21.78, Lithuanian 16.55 vs 20.95, Italian 2.09 vs 2.43.
**Regressions:** **Greek 30.81 vs 21.07** (large — checkpoint selected on English LibriSpeech test-other), French 5.01 vs 4.69 (slight).

### Runtime

NeMo 3.0.0 (`nemo_toolkit[asr]==3.0.0`):
```python
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

checkpoint = hf_hub_download(
    "oruk/orukeet", "orukeet-v0.1.0.nemo",
    revision="555136b50265a132d4cea0d35560c26fc4f657ab",
)
asr = ASRModel.restore_from(checkpoint)
asr.eval()
print(asr.transcribe(["recording.wav"])[0].text)
```

**Alternative:** ONNX int8 export on the repo (`onnx/sherpa-onnx-orukeet-v0.1.0-int8.tar.bz2`, 487 MB) — CPU-friendly, small accuracy cost.

### Gotchas

1. CC BY-SA 4.0 (copyleft) — unlike Whisper's MIT
2. **Greek badly regressed** — keep Parakeet or Whisper as fallback for Greek
3. **Unsupported languages produce confident nonsense**, not errors
4. Model doesn't emit language IDs (detection is separate in OpenWhispr integration)
5. Proper nouns outside the 25 languages transcribe poorly ("Orukeet" → "Oroke")
6. Pin the revision hash — don't resolve "latest"

---

## 6. Model Reference — NVIDIA Parakeet Family

All classic Parakeet models: **CC-BY-4.0** (commercial OK). Disk sizes exact from HF API (2026-10-14).

| HF Repo ID | Params | .nemo Size | Decoder | Langs | Notes |
|---|---|---|---|---|---|
| `nvidia/parakeet-tdt-0.6b-v3` | 627M | 2.51 GB | TDT | 25 (EU) | **Flagship.** Auto-LID, timestamps, PnC. LibriSpeech clean 1.93% |
| `nvidia/parakeet-tdt-0.6b-v2` | 600M | ~2.5 GB | TDT | EN | English predecessor. RTFx 3380 @ batch 128 |
| `nvidia/parakeet-tdt-1.1b` | 1.07B | 4.28 GB | TDT | EN | Best English accuracy of the family |
| `nvidia/parakeet-ctc-0.6b` | 609M | 2.44 GB | CTC | EN | Fastest decode |
| `nvidia/parakeet-ctc-1.1b` | 1.06B | 4.25 GB | CTC | EN | CTC accuracy king |
| `nvidia/parakeet-rnnt-0.6b` | 617M | 2.47 GB | RNNT | EN | Multilingual streaming (Riva) |
| `nvidia/parakeet-rnnt-1.1b` | 1.07B | 4.28 GB | RNNT | EN | RNNT accuracy king |
| `nvidia/parakeet-tdt_ctc-1.1b` | ~1.07B | 4.29 GB | Hybrid | EN | Both heads; pick decode at runtime |
| `nvidia/parakeet-tdt_ctc-110m` | ~114M | ~0.46 GB | Hybrid | EN | Tiny anchor; CTC decode default |
| `nvidia/parakeet-unified-en-0.6b` | ~0.6B | ~2.47 GB | RNNT | EN | Offline + streaming (160ms). **NVIDIA Open Model License, NOT CC-BY** |

**Filenames** for `hf_hub_download`: `parakeet-tdt-0.6b-v3.nemo`, `parakeet-tdt-0.6b-v2.nemo`, `parakeet-tdt-1.1b.nemo`, `parakeet-ctc-0.6b.nemo`, `parakeet-ctc-1.1b.nemo`, `parakeet-rnnt-0.6b.nemo`, `parakeet-rnnt-1.1b.nemo`, `parakeet-tdt_ctc-1.1b.nemo`, `parakeet-tdt_ctc-110m.nemo`, `parakeet-unified-en-0.6b.nemo`

### parakeet-tdt-0.6b-v3 — The Primary NeMo Backend Model

- **Architecture:** FastConformer encoder (24 layers, 8x conv subsampling, ~3x compute / ~4x memory vs vanilla Conformer) + **TDT decoder** (predicts token + duration per step → natural timestamps, faster than RNNT). 2-layer LSTM predictor (640 hidden). 128-bin log-mel frontend.
- **Tokenizer:** Unified multilingual SentencePiece BPE, 8,192 tokens.
- **Languages (25, auto-detected):** bg, hr, cs, da, nl, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, sv, es, en, ru, uk
- **Features:** Auto punctuation + capitalization; native word/segment/char timestamps; up to 24 min single-pass (full attention) or ~3h (local attention)
- **Accuracy:** LibriSpeech test-clean 1.93% / test-other 3.59%; HF Open ASR avg ~6.34% WER multilingual
- **License:** CC-BY-4.0. Also ships `model.safetensors` (for transformers `AutoModelForTDT`) and `q8_0.gguf` (714 MB)
- **Training:** Initialized from multilingual CTC checkpoint; Granary ASR subset (660k h pseudo-labeled) + 10k h human NeMo ASR Set 3.0; stage-2 fine-tune 5k steps on 7.5k h human data. Technical report: arXiv 2509.14128

---

## 7. Model Reference — Cohere Transcribe 2B

**HF Repo:** `CohereLabs/cohere-transcribe-03-2026` (**gated** — accept license + HF token required)
**License:** Apache 2.0 | **transformers ≥ 5.4.0** required for native support

### Architecture

2B parameters — **Conformer encoder + Transformer decoder**, trained from scratch on 500K hours with standard supervised cross-entropy. Same shape family as Parakeet/Canary (Conformer-family encoders are the current SOTA recipe).

### Size and VRAM

~4.1 GB safetensors (fp16). ~5 GB fp16 VRAM + activations/KV cache. Fits on an 8 GB GPU.

### Languages (14 — MUST specify, NO auto-detect)

English, German, French, Italian, Spanish, Portuguese, Greek, Dutch, Polish, Arabic, Vietnamese, Chinese (Mandarin), Japanese, Korean.

Token codes: en, fr, de, es, it, pt, nl, pl, el, ar, ja, zh, vi, ko.

### Benchmarks (HF Open ASR Leaderboard — #1 open model, avg 5.42%)

| Dataset | WER |
|---|---|
| AMI | ~8.1–8.2 |
| Earnings22 | 10.86 |
| GigaSpeech | 9.34 |
| LibriSpeech clean / other | 1.25 / 2.37 |
| SPGISpeech | 3.08 |
| TED-LIUM | 2.49 |
| VoxPopuli | 5.87 |

**Context:** Whisper large-v3 = 7.44 avg; Qwen3-ASR-1.7B = 5.76; Canary Qwen 2.5B = 5.63; Zoom Scribe v1 = 5.47 (closed); IBM Granite 4.0 1B = 5.52.

**Human preference:** 61% avg win rate; 78% vs Granite 4.0 1B, 67% vs Canary Qwen 2.5B, 64% vs Whisper large-v3, 56% vs Zoom Scribe v1. **Trails rivals on Portuguese, German, Spanish** (per Cohere's own disclosure).

**Throughput:** Claimed 525x real-time; 3x higher offline throughput than similarly sized ASR models.

### Inference

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch, soundfile as sf

model_id = "CohereLabs/cohere-transcribe-03-2026"
processor = AutoProcessor.from_pretrained(model_id)  # trust_remote_code not needed on >=5.4
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto")

audio, sr = sf.read("audio.wav")
inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(model.device)
output = model.generate(**inputs, language="en")  # language REQUIRED
print(processor.batch_decode(output, skip_special_tokens=True)[0])
```

### Limitations

1. **No automatic language detection** — `language=` is required
2. **No timestamps** (word or segment)
3. No speaker diarization
4. Hallucinates on silence/noise — pair with VAD for production
5. **Gated weights** — HF token needed on every deploy
6. Native encoder lacks int32-indexing workaround: very long clips (~55 min) can exceed CUDA indexing limit — chunk long audio yourself
7. Punctuation toggle via processor arg

### Other Runtimes

- `onnx-community/cohere-transcribe-03-2026-ONNX` (q4/fp16/int8)
- `handy-computer/cohere-transcribe-03-2026-gguf` (GGUF)
- `beshkenadze/cohere-transcribe-03-2026-mlx-fp16` (Apple Silicon)
- vLLM (production serving)
- nano-cohere-transcribe (minimal pure-PyTorch, 1.5–3.6x faster than transformers)

---

## 8. Other Models Worth Supporting

### 8.1 YES — Include

#### Moonshine (Useful Sensors)
- **Repos:** `UsefulSensors/moonshine-tiny` (27M), `moonshine-base` (61.5M); streaming: `moonshine-ai/moonshine-streaming-{tiny,small,medium}`
- **Arch:** Encoder-decoder transformer, RoPE (no 30s limit), SwiGLU decoder
- **Languages:** English (+ multilingual fine-tunes)
- **License:** MIT | **Runtime:** transformers (native), ONNX
- **Benchmarks:** tiny avg 12.66 vs whisper tiny.en 12.81; base 10.07 vs 10.32; tiny uses ~0.7x FLOPs of whisper tiny
- **Why:** Best accuracy-per-watt tiny model. Ideal as the server's **default cheap/fast English model** and CPU-fallback candidate. Near-instant on GPUs.

#### Qwen3-ASR (Alibaba)
- **Repos:** `Qwen/Qwen3-ASR-1.7B-hf`, `Qwen/Qwen3-ASR-0.6B-hf` (transformers-native); companion `Qwen/Qwen3-ForcedAligner-0.6B` (word timestamps, 11 languages)
- **Size:** 4.08 GB / 1.56 GB bf16 | **Ungated** | **License:** Apache 2.0
- **Arch:** Audio encoder + Qwen3 LLM decoder (audio-LLM style). Offline + streaming.
- **Languages:** **52** — 30 languages + 22 Chinese dialects. Auto language detection + timestamp prediction. Handles speech, singing, songs with BGM.
- **Benchmarks:** 5.76 avg WER; 0.6B hits ~2000x RTF at concurrency 128
- **Why:** Strongest open option for Mandarin/Cantonese/Chinese dialects — nothing else touches 22 dialects. Apache 2.0, ungated. **This is the CJK backend.**

#### NVIDIA Canary
- **Repos:** `nvidia/canary-1b-v2` (978M, **25 European languages**, CC-BY-4.0), `nvidia/canary-1b` (1B, 4 langs), `nvidia/canary-1b-flash` (883M, 1000+ RTFx, experimental timestamps), `nvidia/canary-qwen-2.5b` (2.5B, 5.63 avg WER)
- **Arch:** FastConformer encoder + Transformer decoder; task tokens for source/target language + PnC toggle
- **License:** CC-BY-4.0 | **Runtime:** NeMo `.nemo` only
- **Why:** Best open European-multilingual coverage + speech translation (X↔EN). **No transformers work needed — register in the existing NeMo backend.**

#### Distil-Whisper
- **Repos:** `distil-whisper/distil-large-v3`, `distil-large-v2`, `distil-medium.en`, `distil-small.en` | MIT
- **Why:** 6x faster than Whisper, 49% smaller, within 1% WER OOD. The pragmatic "fast + good + English" default.

#### whisper.cpp (as CPU/offline backend)
- **Repo:** `ggml-org/whisper.cpp` (MIT) | Models: any Whisper as GGUF
- **Why:** Single binary + model file, no Python/PyTorch. **CPU fallback tier** for no-GPU hosts. Also gives access to quantized Cohere/Canary/SenseVoice GGUFs. Complement, not replacement, for faster-whisper.
- **Caveats:** Built-in audio loader doesn't decode Opus/OGG (convert with ffmpeg first). `-l` defaults to `en` — pass `-l auto` explicitly.

### 8.2 MAYBE — Conditional

#### FunASR SenseVoice
- **Repo:** `FunAudioLLM/SenseVoiceSmall` (234M); newer `Fun-ASR-Nano-2512-hf` (800M, transformers-native)
- **Arch:** SANM encoder + CTC (non-autoregressive). Also does language ID, emotion, audio-event detection in one shot.
- **Languages:** zh, yue, en, ja, ko
- **License:** **FunASR Model Open Source License v1.1 — NOT Apache-2.0** (bespoke; read before depending on it)
- **Why maybe:** Very fast CJK CTC; but Qwen3-ASR covers the same languages better under Apache 2.0. Include only for sub-second-latency CJK tier.

### 8.3 WATCH — Not Yet

#### IBM Granite Speech
- **Repos:** `ibm-granite/granite-speech-3.3-*`, 4.0/4.1 series | Apache 2.0
- **Benchmarks:** Granite 4.0 1B = 5.52 avg WER (top-3 open); 4.1 2B-NAR = ~1820 RTFx
- **Why watch:** Top-3 open accuracy + most interesting inference architecture (non-autoregressive). Verify transformers support before committing.

#### Mistral Voxtral
- **Repos:** `mistralai/Voxtral-Mini-3B-2507` (+ Realtime variant) | Apache 2.0
- **Why watch:** 3B audio LLM, 30 min continuous audio, unified transcription/translation/summarization. Interesting for an "audio understanding" tier beyond transcription.

#### Fun-ASR-Nano-2512-hf
- **Repo:** `FunAudioLLM/Fun-ASR-Nano-2512-hf` (800M, transformers-native, 31 languages)
- **Why watch:** The transformers-native FunASR variant — if it works well, it's a fast multilingual CTC option without the FunASR runtime.

### 8.4 SKIP

| Model | Why skip |
|---|---|
| Meta MMS (`facebook/mms-1b-*`) | **CC-BY-NC 4.0** (non-commercial) + weak on major languages |
| Wav2Vec2 XLS-R | **CC-BY-NC 4.0** (non-commercial) |
| Qwen2-Audio / Qwen-Audio | Superseded by Qwen3-ASR |
| Deepgram Nova, AssemblyAI, ElevenLabs Scribe, Zoom Scribe | Closed APIs — out of scope for self-hosted |

### 8.5 Leaderboard State (Late 2026)

| Model | Avg WER | Open? | Notes |
|---|---|---|---|
| Cohere Transcribe 2B | 5.42 | Yes (Apache 2.0) | #1 open |
| Zoom Scribe v1 | 5.47 | **No** (closed) | — |
| IBM Granite 4.0 1B | 5.52 | Yes (Apache 2.0) | Watch |
| NVIDIA Canary Qwen 2.5B | 5.63 | Yes (CC-BY-4.0) | NeMo backend |
| Qwen3-ASR-1.7B | 5.76 | Yes (Apache 2.0) | CJK backend |
| ElevenLabs Scribe v2 | 5.83 | **No** (closed) | — |
| Parakeet TDT 0.6B v3 | 6.32 | Yes (CC-BY-4.0) | **RTFx 3332** (throughput king) |
| Whisper large-v3 | 7.44 | Yes (MIT) | Baseline |

---

## 9. Technical Comparison Matrix

| Model | HF Repo | Size | VRAM (FP16) | Speed (RTFx) | WER (avg) | Languages | License | Runtime |
|---|---|---|---|---|---|---|---|---|
| Whisper tiny | `Systran/faster-whisper-tiny` | 78 MB | ~1 GB | ~32x* | ~11% | 99 | MIT | faster-whisper |
| Whisper base | `Systran/faster-whisper-base` | 148 MB | ~1 GB | ~16x* | ~8% | 99 | MIT | faster-whisper |
| Whisper small | `Systran/faster-whisper-small` | 486 MB | ~2 GB | ~6x* | ~5.5% | 99 | MIT | faster-whisper |
| Whisper medium | `Systran/faster-whisper-medium` | 1.5 GB | ~5 GB | ~2x* | ~4.7% | 99 | MIT | faster-whisper |
| Whisper large-v3 | `Systran/faster-whisper-large-v3` | 3.1 GB | ~10 GB | 1x* | 7.44 | 99 | MIT | faster-whisper |
| **Whisper large-v3-turbo** | **`deepdml/faster-whisper-large-v3-turbo-ct2`** | 1.6 GB | ~4–6 GB | ~8x* | ≈ large-v3 | 99 | MIT | faster-whisper |
| Distil large-v3 | `Systran/faster-distil-whisper-large-v3` | 1.5 GB | ~5 GB | ~5.8x* | within 1% of large-v2 | EN | MIT | faster-whisper |
| **Orukeet** | **`oruk/orukeet`** | 2.5 GB | ~2–3 GB | ~1000x | — | 25 (EU) | **CC BY-SA 4.0** | NeMo 3.0.0 |
| **Parakeet TDT 0.6B v3** | **`nvidia/parakeet-tdt-0.6b-v3`** | 2.51 GB | ~2–3 GB | **3332** | 6.32 | 25 (EU) | CC-BY-4.0 | NeMo 3.0.0 |
| Parakeet TDT 1.1B | `nvidia/parakeet-tdt-1.1b` | 4.28 GB | ~3–4 GB | — | — | EN | CC-BY-4.0 | NeMo 3.0.0 |
| **Cohere Transcribe 2B** | **`CohereLabs/cohere-transcribe-03-2026`** | 4.1 GB | ~5 GB | 525 (claimed) | **5.42** | 14 (specify!) | Apache 2.0 | transformers ≥5.4.0 |
| Moonshine base | `UsefulSensors/moonshine-base` | ~120 MB | <1 GB | very fast | ~10 | EN | MIT | transformers |
| Qwen3-ASR-1.7B | `Qwen/Qwen3-ASR-1.7B-hf` | 4.08 GB | ~4.5 GB | ~2000 (conc.) | 5.76 | **52** | Apache 2.0 | transformers |
| Qwen3-ASR-0.6B | `Qwen/Qwen3-ASR-0.6B-hf` | 1.56 GB | ~2 GB | ~2000 (conc.) | — | 52 | Apache 2.0 | transformers |
| Canary 1b-v2 | `nvidia/canary-1b-v2` | ~4 GB | ~3–4 GB | — | 7.15 (EN) | 25 (EU) | CC-BY-4.0 | NeMo 3.0.0 |
| Granite 4.0 1B | `ibm-granite/granite-speech-4.0-1b` | ~2 GB | ~2–3 GB | — | 5.52 | — | Apache 2.0 | TBD |

*Whisper relative speed vs large-v3 on GPU. RTFx = minutes of audio per minute of processing.

**Bold** = recommended for the initial build.

---

## 10. Recommended Backend Architecture

### Backend Matrix

| Backend | Runtime | Models | GPU |
|---|---|---|---|
| **WhisperBackend** | faster-whisper (CTranslate2) | Whisper turbo/large-v3/medium/small, distil variants | RTX 4060 (8GB) |
| **NemoBackend** | NeMo 3.0.0 | Parakeet TDT 0.6B v3, Orukeet, Canary 1b-v2 | RTX A2000 (12GB) |
| **TransformersBackend** | transformers ≥5.4.0 | Cohere Transcribe 2B, Qwen3-ASR, Moonshine | RTX A2000 (12GB) |
| **CPUBackend** (future) | whisper.cpp | Whisper GGUF (fallback), quantized models | CPU |

### Design Principles (Steal from Speaches)

1. **Executor pattern:** `Executor = (name, model_manager, model_registry, task)` + card-data filter predicates + TTL `SelfDisposingModel` + task-scoped routing
2. **Ollama-style UX:** lazy-download on first inference (speaches 404s — **don't copy this**); explicit `POST /v1/models/{id}` for pre-warming
3. **TTL eviction:** default 300s idle → unload from VRAM; weights stay in HF cache
4. **Per-GPU placement:** config maps backends to `CUDA_VISIBLE_DEVICES`
5. **Revision pinning:** pin `revision=<commit-sha>` per model for reproducibility
6. **Normalize once:** 16 kHz mono float32 at the HTTP layer; all backends get identical input

### Model Registry (Initial)

```yaml
models:
  # WhisperBackend (faster-whisper)
  whisper-turbo:
    backend: whisper
    repo: deepdml/faster-whisper-large-v3-turbo-ct2
  whisper-large-v3:
    backend: whisper
    repo: Systran/faster-whisper-large-v3
  whisper-medium:
    backend: whisper
    repo: Systran/faster-whisper-medium
  whisper-distil-large-v3:
    backend: whisper
    repo: Systran/faster-distil-whisper-large-v3

  # NemoBackend (NeMo 3.0.0)
  parakeet-tdt-0.6b-v3:
    backend: nemo
    repo: nvidia/parakeet-tdt-0.6b-v3
    file: parakeet-tdt-0.6b-v3.nemo
  orukeet:
    backend: nemo
    repo: oruk/orukeet
    file: orukeet-v0.1.0.nemo
    revision: 555136b50265a132d4cea0d35560c26fc4f657ab
  canary-1b-v2:
    backend: nemo
    repo: nvidia/canary-1b-v2
    file: canary-1b-v2.nemo

  # TransformersBackend (transformers >= 5.4.0)
  cohere-transcribe-2b:
    backend: transformers
    repo: CohereLabs/cohere-transcribe-03-2026
    requires_language: true
    gated: true
  qwen3-asr-1.7b:
    backend: transformers
    repo: Qwen/Qwen3-ASR-1.7B-hf
  moonshine-base:
    backend: transformers
    repo: UsefulSensors/moonshine-base
```

### Disk Budget

| Component | Size |
|---|---|
| Whisper turbo + distil-large-v3 | ~3.1 GB |
| Parakeet v3 + Orukeet + Canary | ~9 GB |
| Cohere 2B + Qwen3-ASR-1.7B + Moonshine | ~8.3 GB |
| HF cache overhead (~10%) | ~2 GB |
| **Total** | **~22 GB** |

Recommend **50 GB** volume. 100 GB if adding more variants.

### Build Phases

1. **Phase 1:** Core server + WhisperBackend (faster-whisper). Easiest, proven patterns.
2. **Phase 2:** NemoBackend (NeMo 3.0.0). Covers Orukeet + Parakeet + Canary with one code path.
3. **Phase 3:** TransformersBackend. Cohere (accuracy king) + Qwen3-ASR (CJK) + Moonshine (fast/cheap).
4. **Phase 4:** Polish — Prometheus metrics, Kuma monitor, Grafana dashboard, full registry endpoint.

---

## 11. Gotchas and Warnings

### Critical

1. **`WHISPER__TTL` never existed in speaches.** Dave's `WHISPER__TTL=-1` had no effect (pydantic-settings silently ignores unknown vars). Correct: `STT_MODEL_TTL=-1`.
2. **Turbo repo:** `Systran/faster-whisper-large-v3-turbo` does not exist. faster-whisper's built-in alias → `mobiuslabsgmbh/` (abandoned, re-claimed by unverified account). **Pin `deepdml/faster-whisper-large-v3-turbo-ct2`.**
3. **Cohere needs `transformers >= 5.4.0`** (native support). 5.0/5.1 have a weight-loading bug. **Gated weights** — HF token on every deploy.
4. **Cohere has NO auto language detection.** `language=` is required. No timestamps.

### Important

5. **Orukeet is CC BY-SA 4.0** (copyleft) — unlike Whisper (MIT) and Parakeet (CC-BY-4.0). Matters for redistribution.
6. **Orukeet Greek regression:** 30.81 vs 21.07 WER. Keep Parakeet/Whisper as Greek fallback.
7. **NeMo does NOT resample** — must be 16 kHz mono float32 before `transcribe()`.
8. **NeMo 3.0 `from_pretrained` gotcha** — can fail on bare `.nemo` repos. Use `hf_hub_download` + `restore_from`.
9. **NeMo dependency footprint** — ~8–10 GB Docker layer. Unavoidable for native Parakeet/Orukeet.
10. **Turbo NOT trained for translation** — returns source language even when `task=translate`.
11. **transformers is 2–5x slower** than CT2/NeMo — batch aggressively, fp16, FA2/SDPA.
12. **License landmines:** MMS/XLS-R are CC-BY-NC (skip). SenseVoice is bespoke FunASR license (read it).

### Useful

13. **Timestamps are free on TDT models** — always request them on the NeMo backend.
14. **ivrit-ai Hebrew fine-tune** needs `language="he"` explicitly (detection degraded in fine-tuning).
15. **ONNX/TensorRT export not worth it** for phase 1 — stock NeMo eager is ~1000x real-time.
16. **Parakeet v3 also ships `model.safetensors`** (transformers `AutoModelForTDT`) and `q8_0.gguf` (714 MB) — only `.nemo` matters for the NeMo backend.

---

## Sources

- speaches-ai/speaches (GitHub, master @ 993994f, 2026-04-17)
- NVIDIA-NeMo/Speech (GitHub, v3.0.0)
- HuggingFace Hub (model cards, API blob sizes, 2026-10-14)
- HF Open ASR Leaderboard (late 2026)
- Orukeet paper: arXiv 2609.10054
- Parakeet technical report: arXiv 2509.14128
- Moonshine paper: arXiv 2410.15608
- faster-whisper (SYSTRAN, GitHub)
- ggml-org/whisper.cpp (GitHub)

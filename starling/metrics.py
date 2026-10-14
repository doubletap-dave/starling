from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

INFERENCE_LATENCY = Histogram(
    "starling_inference_seconds",
    "STT inference latency",
    ["model", "backend"],
)
INFERENCE_TOTAL = Counter(
    "starling_inference_total",
    "STT inference count",
    ["model", "backend", "status"],
)
MODELS_LOADED = Gauge(
    "starling_models_loaded",
    "Number of models currently in VRAM",
)
CACHE_HITS = Counter(
    "starling_model_cache_hits_total",
    "Model already downloaded when requested",
)
CACHE_MISSES = Counter(
    "starling_model_cache_misses_total",
    "Model download triggered",
)


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

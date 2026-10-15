from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MUTABLE_SETTING_KEYS = (
    "stt_model_ttl",
    "device_override",
    "whisper_compute_type",
    "preload_models",
    "metrics_enabled",
    "hf_token",
    "hf_hub_cache",
)


def _default_registry_path() -> Path:
    repo_config = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    if repo_config.exists():
        return repo_config
    return Path(__file__).resolve().parent / "data" / "models.yaml"


def _default_runtime_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "runtime.yaml"


def preload_models_from_env(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    import json

    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        data: Any = json.loads(raw)
        return list(data)
    return [p.strip() for p in raw.split(",") if p.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None

    stt_model_ttl: int = Field(default=300, description="Idle seconds before unload; -1=never, 0=immediate")
    registry_path: Path = Field(default_factory=_default_registry_path)
    runtime_config_path: Path = Field(default_factory=_default_runtime_path)
    hf_hub_cache: Path | None = None
    hf_token: str | None = Field(
        default=None,
        description="Optional. Only needed for gated/private Hugging Face repos (e.g. Cohere).",
    )
    # Download on startup only — does not load into VRAM (see app lifespan).
    preload_models: list[str] = Field(
        default_factory=lambda: ["orukeet"],
        description="Model ids to download on startup; kept on disk, loaded on first use",
    )

    # Optional device pinning: "cuda:0", "cpu", or empty for auto
    device_override: str | None = None
    whisper_compute_type: str = "default"

    # Prometheus
    metrics_enabled: bool = True

    enable_ui: bool = True

    @field_validator("preload_models", mode="before")
    @classmethod
    def _parse_preload_models(cls, value: Any) -> list[str]:
        return preload_models_from_env(value)


settings = Settings()


def load_runtime_overlay(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in MUTABLE_SETTING_KEYS}


def save_runtime_overlay(path: Path, overlay: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {k: v for k, v in overlay.items() if k in MUTABLE_SETTING_KEYS}
    tmp = path.with_suffix(".yaml.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cleaned, f, sort_keys=True)
    tmp.replace(path)


def apply_overlay(settings_obj: Settings, overlay: dict[str, Any]) -> Settings:
    """Return settings with overlay applied (mutates settings_obj in place and returns it)."""
    for key, value in overlay.items():
        if key not in MUTABLE_SETTING_KEYS:
            continue
        if key == "hf_hub_cache" and value is not None:
            value = Path(value)
        if key == "preload_models" and value is None:
            value = []
        setattr(settings_obj, key, value)
    return settings_obj


def settings_public_dict(settings_obj: Settings) -> dict[str, Any]:
    """Safe settings for API responses — never expose raw secrets."""
    return {
        "stt_model_ttl": settings_obj.stt_model_ttl,
        "device_override": settings_obj.device_override,
        "whisper_compute_type": settings_obj.whisper_compute_type,
        "preload_models": list(settings_obj.preload_models or []),
        "metrics_enabled": settings_obj.metrics_enabled,
        "hf_hub_cache": str(settings_obj.hf_hub_cache) if settings_obj.hf_hub_cache else None,
        "hf_token_set": bool(settings_obj.hf_token),
        "api_key_set": bool(settings_obj.api_key),
        "enable_ui": settings_obj.enable_ui,
        "registry_path": str(settings_obj.registry_path),
        "runtime_config_path": str(settings_obj.runtime_config_path),
        "host": settings_obj.host,
        "port": settings_obj.port,
    }


def bootstrap_settings(base: Settings | None = None) -> Settings:
    cfg = base or Settings()
    overlay = load_runtime_overlay(cfg.runtime_config_path)
    return apply_overlay(cfg, overlay)

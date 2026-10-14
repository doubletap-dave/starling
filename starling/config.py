from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_registry_path() -> Path:
    repo_config = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    if repo_config.exists():
        return repo_config
    return Path(__file__).resolve().parent / "data" / "models.yaml"


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
    hf_hub_cache: Path | None = None
    hf_token: str | None = None
    preload_models: list[str] = Field(default_factory=list)

    # Optional device pinning: "cuda:0", "cpu", or empty for auto
    device_override: str | None = None
    whisper_compute_type: str = "default"

    # Prometheus
    metrics_enabled: bool = True


settings = Settings()


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

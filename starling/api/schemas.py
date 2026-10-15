from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SettingsPatch(BaseModel):
    stt_model_ttl: int | None = None
    device_override: str | None = None
    whisper_compute_type: str | None = None
    preload_models: list[str] | None = None
    metrics_enabled: bool | None = None
    hf_hub_cache: str | None = None
    # Write-only: omit to leave unchanged; "" clears
    hf_token: str | None = Field(default=None, description="Write-only; empty string clears")


class RegistryEntryCreate(BaseModel):
    id: str
    backend: Literal["whisper", "nemo", "transformers"]
    repo: str
    file: str | None = None
    revision: str | None = None
    vram_mb: int = 4096
    requires_language: bool = False
    gated: bool = False
    languages: Any = "auto"
    supports_translate: bool = False
    supports_timestamps: bool = False
    license: str | None = None


class RegistryEntryUpdate(BaseModel):
    backend: Literal["whisper", "nemo", "transformers"] | None = None
    repo: str | None = None
    file: str | None = None
    revision: str | None = None
    vram_mb: int | None = None
    requires_language: bool | None = None
    gated: bool | None = None
    languages: Any | None = None
    supports_translate: bool | None = None
    supports_timestamps: bool | None = None
    license: str | None = None

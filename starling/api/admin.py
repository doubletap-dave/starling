from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from starling import __version__
from starling.api.deps import get_manager, get_placer, get_registry, get_settings
from starling.api.schemas import SettingsPatch
from starling.config import (
    Settings,
    load_runtime_overlay,
    save_runtime_overlay,
    settings_public_dict,
)
from starling.gpu.placer import GpuPlacer
from starling.lifecycle.manager import ModelManager
from starling.registry.catalog import ModelRegistry

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/status")
async def admin_status(
    manager: Annotated[ModelManager, Depends(get_manager)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    placer: Annotated[GpuPlacer, Depends(get_placer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    devices = [
        {
            "index": d.index,
            "name": d.name,
            "device": d.torch_device,
            "uuid": d.uuid,
            "total_mb": d.total_mb,
            "free_mb": d.free_mb,
        }
        for d in placer.discover()
    ]
    loaded = []
    for mid in manager.loaded_ids():
        lm = manager._loaded.get(mid)
        item: dict[str, Any] = {"id": mid}
        if lm is not None:
            item["backend"] = lm.entry.backend
            item["device"] = lm.placement.device
            item["uuid"] = lm.placement.uuid
            item["vram_mb"] = lm.entry.vram_mb
        loaded.append(item)

    return {
        "version": __version__,
        "backends": sorted(manager.backends.keys()),
        "devices": devices,
        "loaded": loaded,
        "registry_count": len(registry.ids()),
        "downloaded_count": sum(1 for e in registry.all() if registry.is_downloaded(e)),
        "settings": {
            "stt_model_ttl": settings.stt_model_ttl,
            "device_override": settings.device_override,
        },
    }


@router.get("/settings")
async def get_settings_api(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    return settings_public_dict(settings)


@router.patch("/settings")
@router.put("/settings")
async def patch_settings(
    body: SettingsPatch,
    settings: Annotated[Settings, Depends(get_settings)],
    manager: Annotated[ModelManager, Depends(get_manager)],
    placer: Annotated[GpuPlacer, Depends(get_placer)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> dict[str, Any]:
    updates = body.model_dump(exclude_unset=True)
    overlay = load_runtime_overlay(settings.runtime_config_path)

    if "hf_token" in updates:
        token = updates.pop("hf_token")
        if token == "" or token is None:
            overlay["hf_token"] = None
            settings.hf_token = None
            registry.token = None
        else:
            overlay["hf_token"] = token
            settings.hf_token = token
            registry.token = token

    if "hf_hub_cache" in updates:
        cache = updates.pop("hf_hub_cache")
        if cache == "" or cache is None:
            overlay["hf_hub_cache"] = None
            settings.hf_hub_cache = None
            registry.cache_dir = None
        else:
            overlay["hf_hub_cache"] = str(cache)
            settings.hf_hub_cache = Path(cache)
            registry.cache_dir = str(settings.hf_hub_cache)

    if "device_override" in updates:
        dev = updates.pop("device_override")
        if dev == "":
            dev = None
        overlay["device_override"] = dev
        settings.device_override = dev
        placer.device_override = dev

    if "stt_model_ttl" in updates:
        ttl = int(updates.pop("stt_model_ttl"))
        overlay["stt_model_ttl"] = ttl
        settings.stt_model_ttl = ttl
        manager.ttl_seconds = ttl

    if "whisper_compute_type" in updates:
        ct = updates.pop("whisper_compute_type") or "default"
        overlay["whisper_compute_type"] = ct
        settings.whisper_compute_type = ct
        backend = manager.backends.get("whisper")
        if backend is not None and hasattr(backend, "compute_type"):
            backend.compute_type = ct

    if "preload_models" in updates:
        preload = updates.pop("preload_models") or []
        overlay["preload_models"] = list(preload)
        settings.preload_models = list(preload)

    if "metrics_enabled" in updates:
        enabled = bool(updates.pop("metrics_enabled"))
        overlay["metrics_enabled"] = enabled
        settings.metrics_enabled = enabled

    save_runtime_overlay(settings.runtime_config_path, overlay)
    return settings_public_dict(settings)

from __future__ import annotations

import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, Response

from starling import metrics as M
from starling.api.deps import get_manager, get_registry, get_settings
from starling.config import Settings
from starling.api.formats import format_transcript
from starling.api.schemas import RegistryEntryCreate, RegistryEntryUpdate
from starling.audio.normalize import load_audio_16k_mono
from starling.lifecycle.manager import ModelManager
from starling.registry.catalog import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter()


def _entry_view(registry: ModelRegistry, manager: ModelManager, model_id: str) -> dict[str, Any]:
    entry = registry.require(model_id)
    loaded = model_id in manager.loaded_ids()
    data = registry.entry_public_dict(
        entry,
        downloaded=registry.is_downloaded(entry),
        loaded=loaded,
    )
    if loaded:
        lm = manager._loaded.get(model_id)
        if lm is not None:
            data["placement"] = {
                "device": lm.placement.device,
                "device_index": lm.placement.device_index,
                "use_device_map": lm.placement.use_device_map,
            }
    return data


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "message": "OK"}


@router.get("/metrics")
async def prometheus_metrics(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    if not settings.metrics_enabled:
        raise HTTPException(404, "Metrics disabled")
    body, content_type = M.metrics_response()
    return Response(content=body, media_type=content_type)


@router.get("/v1/registry")
async def list_registry(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    manager: Annotated[ModelManager, Depends(get_manager)],
) -> dict[str, Any]:
    items = [
        registry.entry_public_dict(
            entry,
            downloaded=registry.is_downloaded(entry),
            loaded=entry.id in manager.loaded_ids(),
        )
        for entry in registry.all()
    ]
    return {"object": "list", "data": items}


@router.post("/v1/registry")
async def create_registry_entry(
    body: RegistryEntryCreate,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    manager: Annotated[ModelManager, Depends(get_manager)],
) -> dict[str, Any]:
    data = body.model_dump(exclude={"id"})
    try:
        entry = registry.create(body.id, data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return registry.entry_public_dict(entry, downloaded=False, loaded=False)


@router.put("/v1/registry/{model_id}")
async def update_registry_entry(
    model_id: str,
    body: RegistryEntryUpdate,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    manager: Annotated[ModelManager, Depends(get_manager)],
) -> dict[str, Any]:
    if model_id in manager.loaded_ids():
        raise HTTPException(409, f"Model '{model_id}' is loaded; unload before editing")
    try:
        entry = registry.update(model_id, body.model_dump(exclude_unset=True))
    except KeyError:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _entry_view(registry, manager, entry.id)


@router.delete("/v1/registry/{model_id}")
async def delete_registry_entry(
    model_id: str,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    manager: Annotated[ModelManager, Depends(get_manager)],
    purge: bool = Query(False),
) -> dict[str, Any]:
    if model_id in manager.loaded_ids():
        raise HTTPException(409, f"Model '{model_id}' is loaded; unload before deleting from registry")
    try:
        entry = registry.require(model_id)
    except KeyError:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None
    purge_info: dict[str, Any] | None = None
    if purge:
        purge_info = registry.purge(entry)
    try:
        registry.delete(model_id)
    except KeyError:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None
    return {
        "id": model_id,
        "deleted": True,
        "purged": bool(purge and purge_info and purge_info.get("method") != "noop"),
        "purge": purge_info,
    }


@router.get("/v1/models")
async def list_models(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> dict[str, Any]:
    data = []
    for entry in registry.all():
        if registry.is_downloaded(entry):
            data.append(
                {
                    "id": entry.id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "starling",
                    "permission": [],
                    "root": entry.id,
                    "parent": None,
                }
            )
    return {"object": "list", "data": data}


@router.get("/v1/models/{model_id}")
async def get_model(
    model_id: str,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    manager: Annotated[ModelManager, Depends(get_manager)],
) -> dict[str, Any]:
    try:
        return _entry_view(registry, manager, model_id)
    except KeyError:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None


@router.post("/v1/models/{model_id}/load")
async def load_model(
    model_id: str,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    manager: Annotated[ModelManager, Depends(get_manager)],
) -> dict[str, Any]:
    try:
        registry.require(model_id)
    except KeyError:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None
    try:
        manager.load(model_id)
    except Exception as exc:
        logger.exception("Load failed for %s", model_id)
        raise HTTPException(500, f"Load failed: {exc}") from exc
    M.MODELS_LOADED.set(len(manager.loaded_ids()))
    return _entry_view(registry, manager, model_id)


@router.post("/v1/models/{model_id}/unload")
async def unload_model(
    model_id: str,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    manager: Annotated[ModelManager, Depends(get_manager)],
) -> dict[str, Any]:
    try:
        registry.require(model_id)
    except KeyError:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None
    unloaded = manager.unload(model_id)
    M.MODELS_LOADED.set(len(manager.loaded_ids()))
    return {"id": model_id, "unloaded": unloaded, **_entry_view(registry, manager, model_id)}


@router.get("/v1/jobs")
async def list_jobs(active: bool = Query(True)) -> dict[str, Any]:
    from starling.progress import progress_store

    jobs = progress_store.list_active() if active else progress_store.list_recent()
    return {"object": "list", "data": [j.to_dict() for j in jobs]}


@router.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    from starling.progress import progress_store

    job = progress_store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job '{job_id}'")
    return job.to_dict()


@router.post("/v1/models/{model_id}")
async def download_model(
    model_id: str,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    manager: Annotated[ModelManager, Depends(get_manager)],
    wait: bool = Query(
        True,
        description="If false, start download in background and return 202 with job_id for polling",
    ),
) -> dict[str, Any]:
    try:
        registry.require(model_id)
    except KeyError:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None

    try:
        if wait:
            result = manager.download_wait(model_id)
        else:
            result = manager.start_download(model_id)
    except Exception as exc:
        logger.exception("Download failed for %s", model_id)
        detail = str(exc)
        if "401" in detail or "gated" in detail.lower():
            raise HTTPException(401, f"Model '{model_id}' is gated; set HF_TOKEN") from exc
        raise HTTPException(500, f"Download failed: {exc}") from exc

    if result.get("status") == "already_downloaded":
        M.CACHE_HITS.inc()
    elif result.get("status") in ("downloaded", "downloading"):
        M.CACHE_MISSES.inc()
    return result


@router.delete("/v1/models/{model_id}")
async def delete_model(
    model_id: str,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    manager: Annotated[ModelManager, Depends(get_manager)],
    purge: bool = Query(False),
) -> dict[str, Any]:
    try:
        entry = registry.require(model_id)
    except KeyError:
        raise HTTPException(404, f"Unknown model '{model_id}'") from None

    unloaded = manager.unload(model_id)
    purge_info: dict[str, Any] | None = None
    if purge:
        purge_info = registry.purge(entry)
    still_downloaded = registry.is_downloaded(entry)
    M.MODELS_LOADED.set(len(manager.loaded_ids()))
    return {
        "id": model_id,
        "unloaded": unloaded,
        "purged": bool(purge and purge_info and purge_info.get("method") != "noop"),
        "still_downloaded": still_downloaded,
        "purge": purge_info,
        **_entry_view(registry, manager, model_id),
    }


@router.post("/v1/audio/transcriptions")
async def transcriptions(
    manager: Annotated[ModelManager, Depends(get_manager)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: list[str] | None = Form(None),
    hotwords: str | None = Form(None),
) -> Any:
    return await _run_stt(
        manager=manager,
        registry=registry,
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        hotwords=hotwords,
        task="transcribe",
    )


@router.post("/v1/audio/translations")
async def translations(
    manager: Annotated[ModelManager, Depends(get_manager)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
) -> Any:
    try:
        entry = registry.require(model)
    except KeyError:
        raise HTTPException(404, f"Unknown model '{model}'") from None
    if not entry.supports_translate:
        raise HTTPException(
            400,
            f"Model '{model}' does not support translation (e.g. whisper-turbo is not trained for translate)",
        )
    return await _run_stt(
        manager=manager,
        registry=registry,
        file=file,
        model=model,
        language=None,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=None,
        hotwords=None,
        task="translate",
    )


async def _run_stt(
    *,
    manager: ModelManager,
    registry: ModelRegistry,
    file: UploadFile,
    model: str,
    language: str | None,
    prompt: str | None,
    response_format: str,
    temperature: float,
    timestamp_granularities: list[str] | None,
    hotwords: str | None,
    task: str,
) -> Any:
    try:
        entry = registry.require(model)
    except KeyError:
        raise HTTPException(404, f"Unknown model '{model}'. See GET /v1/registry") from None

    if entry.requires_language and not language:
        raise HTTPException(
            400,
            f"Model '{model}' requires a language parameter (no auto-detect)",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty audio file")

    try:
        audio = load_audio_16k_mono(raw, filename=file.filename)
    except Exception as exc:
        raise HTTPException(400, f"Failed to decode audio: {exc}") from exc

    word_ts = bool(timestamp_granularities and "word" in timestamp_granularities)
    if entry.supports_timestamps and response_format == "verbose_json":
        word_ts = True

    t0 = time.perf_counter()
    try:
        if not registry.is_downloaded(entry):
            M.CACHE_MISSES.inc()
        else:
            M.CACHE_HITS.inc()

        result = manager.transcribe(
            model,
            audio,
            language=language,
            task=task,
            temperature=temperature,
            word_timestamps=word_ts and entry.supports_timestamps,
            hotwords=hotwords,
            prompt=prompt,
        )
        M.INFERENCE_TOTAL.labels(model=model, backend=entry.backend, status="ok").inc()
    except ValueError as exc:
        M.INFERENCE_TOTAL.labels(model=model, backend=entry.backend, status="error").inc()
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        M.INFERENCE_TOTAL.labels(model=model, backend=entry.backend, status="error").inc()
        logger.exception("Inference failed for %s", model)
        raise HTTPException(500, f"Inference failed: {exc}") from exc
    finally:
        M.INFERENCE_LATENCY.labels(model=model, backend=entry.backend).observe(time.perf_counter() - t0)
        M.MODELS_LOADED.set(len(manager.loaded_ids()))

    result.task = task
    payload = format_transcript(result, response_format)
    if response_format in ("text", "srt", "vtt"):
        return PlainTextResponse(content=payload if isinstance(payload, str) else str(payload))
    return payload

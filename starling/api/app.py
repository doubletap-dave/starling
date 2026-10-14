from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI

from starling import __version__
from starling.api.auth import make_auth_dependency
from starling.api.routes import router
from starling.backends.nemo import NemoBackend
from starling.backends.transformers import TransformersBackend
from starling.backends.whisper import WhisperBackend
from starling.config import Settings, settings as default_settings
from starling.gpu.placer import GpuPlacer
from starling.lifecycle.manager import ModelManager
from starling.registry.catalog import ModelRegistry

logger = logging.getLogger(__name__)


def _build_backends(settings: Settings) -> dict:
    import importlib.util

    backends: dict = {
        "whisper": WhisperBackend(compute_type=settings.whisper_compute_type),
        "transformers": TransformersBackend(),
    }
    if importlib.util.find_spec("nemo") is not None:
        backends["nemo"] = NemoBackend()
    else:
        logger.warning(
            "nemo-toolkit not installed; NeMo models unavailable until: uv sync --extra nemo"
        )
    return backends


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or default_settings
    auth_dep = make_auth_dependency(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        registry = ModelRegistry(cfg.registry_path, token=cfg.hf_token, cache_dir=cfg.hf_hub_cache)
        placer = GpuPlacer(device_override=cfg.device_override)
        backends = _build_backends(cfg)
        manager = ModelManager(registry, backends, placer, ttl_seconds=cfg.stt_model_ttl)

        app.state.settings = cfg
        app.state.registry = registry
        app.state.manager = manager
        app.state.placer = placer

        devices = placer.discover()
        logger.info(
            "Starling %s starting; devices=%s backends=%s",
            __version__,
            [d.torch_device for d in devices],
            list(backends.keys()),
        )

        for mid in cfg.preload_models:
            try:
                manager.load(mid)
                logger.info("Preloaded model %s", mid)
            except Exception:
                logger.exception("Failed to preload %s", mid)
                raise

        yield

        for mid in list(manager.loaded_ids()):
            manager.unload(mid)

    app = FastAPI(
        title="Starling",
        description="Universal OpenAI-compatible speech-to-text server",
        version=__version__,
        lifespan=lifespan,
        dependencies=[Depends(auth_dep)],
    )
    app.include_router(router)
    return app

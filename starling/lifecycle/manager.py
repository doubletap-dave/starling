from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Protocol

from starling.gpu.memory import release_cuda_memory
from starling.gpu.placer import GpuPlacer, Placement
from starling.progress import progress_store
from starling.registry.catalog import ModelEntry, ModelRegistry

logger = logging.getLogger(__name__)


class Backend(Protocol):
    name: str

    def load(self, entry: ModelEntry, local_path: str, placement: Placement) -> Any: ...

    def unload(self, handle: Any) -> None: ...

    def transcribe(self, handle: Any, audio: Any, **kwargs: Any) -> Any: ...


@dataclass
class LoadedModel:
    entry: ModelEntry
    handle: Any
    placement: Placement
    refcount: int = 0
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)


class ModelManager:
    def __init__(
        self,
        registry: ModelRegistry,
        backends: dict[str, Backend],
        placer: GpuPlacer,
        ttl_seconds: int = 300,
    ) -> None:
        self.registry = registry
        self.backends = backends
        self.placer = placer
        self.ttl_seconds = ttl_seconds
        self._loaded: dict[str, LoadedModel] = {}
        self._global = threading.RLock()
        self._timers: dict[str, threading.Timer] = {}
        self._download_threads: dict[str, threading.Thread] = {}

    def is_loaded(self, model_id: str) -> bool:
        with self._global:
            return model_id in self._loaded

    def loaded_ids(self) -> list[str]:
        with self._global:
            return list(self._loaded.keys())

    def ensure_downloaded(self, entry: ModelEntry, *, job_id: str | None = None) -> str:
        if self.registry.is_downloaded(entry):
            path = self.registry.local_path(entry)
            if path is not None:
                return path
        logger.info("Downloading model %s from %s", entry.id, entry.repo)
        return self.registry.download(entry, job_id=job_id)

    def start_download(self, model_id: str) -> dict[str, Any]:
        """Start or join a background download. Returns job status dict."""
        entry = self.registry.require(model_id)
        if self.registry.is_downloaded(entry):
            return {"id": model_id, "status": "already_downloaded", "code": 201, "job_id": None}

        existing = progress_store.active_for_model(model_id, "download")
        if existing is not None:
            return {"id": model_id, "status": "downloading", "code": 202, "job_id": existing.id}

        job_id = f"download:{model_id}:{time.time_ns()}"
        progress_store.start(job_id, model_id, "download", message=f"Downloading {model_id}")

        def _run() -> None:
            try:
                self.registry.download(entry, job_id=job_id)
                progress_store.finish(job_id)
            except Exception as exc:
                logger.exception("Download failed for %s", model_id)
                progress_store.finish(job_id, error=str(exc))
            finally:
                with self._global:
                    self._download_threads.pop(model_id, None)

        thread = threading.Thread(target=_run, name=f"dl-{model_id}", daemon=True)
        with self._global:
            self._download_threads[model_id] = thread
        thread.start()
        return {"id": model_id, "status": "downloading", "code": 202, "job_id": job_id}

    def download_wait(self, model_id: str) -> dict[str, Any]:
        """Synchronous download with progress reporting (for curl / wait=true)."""
        entry = self.registry.require(model_id)
        if self.registry.is_downloaded(entry):
            return {"id": model_id, "status": "already_downloaded", "code": 201, "job_id": None}

        job_id = f"download:{model_id}:{time.time_ns()}"
        progress_store.start(job_id, model_id, "download", message=f"Downloading {model_id}")
        try:
            self.registry.download(entry, job_id=job_id)
            progress_store.finish(job_id)
            return {"id": model_id, "status": "downloaded", "code": 200, "job_id": job_id}
        except Exception as exc:
            progress_store.finish(job_id, error=str(exc))
            raise

    def _backend_for(self, entry: ModelEntry) -> Backend:
        backend = self.backends.get(entry.backend)
        if backend is None:
            raise RuntimeError(f"Backend '{entry.backend}' is not available")
        return backend

    def _cancel_timer(self, model_id: str) -> None:
        timer = self._timers.pop(model_id, None)
        if timer is not None:
            timer.cancel()

    def _schedule_unload(self, model_id: str) -> None:
        self._cancel_timer(model_id)
        if self.ttl_seconds < 0:
            return
        if self.ttl_seconds == 0:
            self.unload(model_id)
            return

        def _fire() -> None:
            with self._global:
                loaded = self._loaded.get(model_id)
                if loaded is None or loaded.refcount > 0:
                    return
            self.unload(model_id)

        timer = threading.Timer(self.ttl_seconds, _fire)
        timer.daemon = True
        self._timers[model_id] = timer
        timer.start()

    def load(self, model_id: str) -> LoadedModel:
        entry = self.registry.require(model_id)
        with self._global:
            if model_id in self._loaded:
                loaded = self._loaded[model_id]
                loaded.last_used = time.monotonic()
                self._cancel_timer(model_id)
                return loaded

        job_id = f"load:{model_id}:{time.time_ns()}"
        progress_store.start(job_id, model_id, "load", message=f"Preparing {model_id}")
        try:
            if not self.registry.is_downloaded(entry):
                progress_store.update(job_id, message=f"Downloading {model_id}")
            local_path = self.ensure_downloaded(entry, job_id=job_id)
            progress_store.update(job_id, message=f"Loading {model_id} into VRAM")
            backend = self._backend_for(entry)
            allow_map = entry.backend == "transformers"
            placement = self.placer.place(entry.vram_mb, allow_device_map=allow_map)
            logger.info("Loading %s on %s via %s", model_id, placement.device, entry.backend)
            handle = backend.load(entry, local_path, placement)
            self.placer.reserve(placement.device if placement.device != "auto" else "cuda:0", entry.vram_mb)

            loaded = LoadedModel(entry=entry, handle=handle, placement=placement)
            with self._global:
                self._loaded[model_id] = loaded
                self._cancel_timer(model_id)
            progress_store.finish(job_id)
            return loaded
        except Exception as exc:
            progress_store.finish(job_id, error=str(exc))
            raise

    def unload(self, model_id: str) -> bool:
        with self._global:
            loaded = self._loaded.pop(model_id, None)
            self._cancel_timer(model_id)
        if loaded is None:
            return False
        with loaded.lock:
            backend = self._backend_for(loaded.entry)
            handle = loaded.handle
            # Drop manager ref first so backend.unload + empty_cache see no live holders
            loaded.handle = None
            backend.unload(handle)
            del handle
            device = loaded.placement.device if loaded.placement.device != "auto" else "cuda:0"
            self.placer.release(device, loaded.entry.vram_mb)
        release_cuda_memory()
        logger.info("Unloaded model %s (CUDA cache cleared)", model_id)
        return True

    @contextmanager
    def use(self, model_id: str) -> Generator[LoadedModel, None, None]:
        loaded = self.load(model_id)
        with loaded.lock:
            loaded.refcount += 1
            loaded.last_used = time.monotonic()
            self._cancel_timer(model_id)
        try:
            yield loaded
        finally:
            with loaded.lock:
                loaded.refcount -= 1
                loaded.last_used = time.monotonic()
                if loaded.refcount == 0:
                    self._schedule_unload(model_id)

    def transcribe(self, model_id: str, audio: Any, **kwargs: Any) -> Any:
        with self.use(model_id) as loaded:
            backend = self._backend_for(loaded.entry)
            return backend.transcribe(loaded.handle, audio, **kwargs)

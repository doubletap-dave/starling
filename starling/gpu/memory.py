from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def release_cuda_memory() -> None:
    """Return freed tensors to the driver. PyTorch otherwise keeps VRAM in its pool."""
    gc.collect()
    try:
        import torch

        if not torch.cuda.is_available():
            return
        # Synchronize so pending kernels finish before we drop the cache
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        logger.debug("Released CUDA caching allocator memory")
    except Exception as exc:  # pragma: no cover
        logger.warning("CUDA memory release failed: %s", exc)

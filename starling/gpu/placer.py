from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    index: int | None  # None = CPU
    name: str
    total_mb: int
    free_mb: int

    @property
    def torch_device(self) -> str:
        if self.index is None:
            return "cpu"
        return f"cuda:{self.index}"


@dataclass(frozen=True)
class Placement:
    device: str  # "cpu", "cuda:0", or "auto" for device_map
    device_index: int | None
    use_device_map: bool = False


class GpuPlacer:
    """VRAM-aware placement across available CUDA devices."""

    def __init__(self, device_override: str | None = None) -> None:
        self.device_override = device_override
        self._reserved_mb: dict[str, int] = {}

    def discover(self) -> list[DeviceInfo]:
        devices: list[DeviceInfo] = []
        try:
            import torch

            if not torch.cuda.is_available():
                return [DeviceInfo(index=None, name="cpu", total_mb=0, free_mb=0)]

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_b, total_b = torch.cuda.mem_get_info(i)
                reserved = self._reserved_mb.get(f"cuda:{i}", 0)
                free_mb = max(0, int(free_b / (1024 * 1024)) - reserved)
                devices.append(
                    DeviceInfo(
                        index=i,
                        name=props.name,
                        total_mb=int(total_b / (1024 * 1024)),
                        free_mb=free_mb,
                    )
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("CUDA discovery failed: %s", exc)
            return [DeviceInfo(index=None, name="cpu", total_mb=0, free_mb=0)]

        if not devices:
            devices.append(DeviceInfo(index=None, name="cpu", total_mb=0, free_mb=0))
        return devices

    def reserve(self, device: str, vram_mb: int) -> None:
        self._reserved_mb[device] = self._reserved_mb.get(device, 0) + vram_mb

    def release(self, device: str, vram_mb: int) -> None:
        cur = self._reserved_mb.get(device, 0) - vram_mb
        if cur <= 0:
            self._reserved_mb.pop(device, None)
        else:
            self._reserved_mb[device] = cur

    def place(
        self,
        vram_mb: int,
        *,
        allow_device_map: bool = False,
        prefer_single_device: bool = True,
    ) -> Placement:
        if self.device_override:
            override = self.device_override.strip().lower()
            if override == "cpu":
                return Placement(device="cpu", device_index=None)
            if override.startswith("cuda"):
                idx = int(override.split(":")[1]) if ":" in override else 0
                return Placement(device=f"cuda:{idx}", device_index=idx)
            if override == "auto" and allow_device_map:
                return Placement(device="auto", device_index=None, use_device_map=True)

        devices = self.discover()
        gpus = [d for d in devices if d.index is not None]

        if not gpus:
            return Placement(device="cpu", device_index=None)

        # Prefer a single GPU with enough free VRAM (largest free wins on ties)
        fitting = sorted(
            [d for d in gpus if d.free_mb >= vram_mb],
            key=lambda d: d.free_mb,
            reverse=True,
        )
        if fitting and prefer_single_device:
            best = fitting[0]
            return Placement(device=best.torch_device, device_index=best.index)

        # No single GPU fits — shard via device_map when allowed
        if allow_device_map:
            total_free = sum(d.free_mb for d in gpus)
            if total_free >= vram_mb:
                return Placement(device="auto", device_index=None, use_device_map=True)

        # Best effort: largest free GPU (caller may still OOM)
        best = max(gpus, key=lambda d: d.free_mb)
        if best.free_mb < vram_mb:
            logger.warning(
                "No GPU has %s MB free (best %s has %s MB); placing on %s anyway",
                vram_mb,
                best.name,
                best.free_mb,
                best.torch_device,
            )
        return Placement(device=best.torch_device, device_index=best.index)

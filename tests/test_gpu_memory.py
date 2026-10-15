import pytest

from starling.gpu.memory import release_cuda_memory
from starling.gpu.placer import GpuPlacer, normalize_gpu_uuid


def test_normalize_gpu_uuid():
    assert normalize_gpu_uuid("GPU-8ee934f5-a686-0f6f-8ff1-925a48f5e211") == (
        "8ee934f5-a686-0f6f-8ff1-925a48f5e211"
    )
    assert normalize_gpu_uuid("8ee934f5-a686-0f6f-8ff1-925a48f5e211") == (
        "8ee934f5-a686-0f6f-8ff1-925a48f5e211"
    )


def test_resolve_cuda_override():
    placer = GpuPlacer()
    p = placer.resolve_override("cuda:1")
    assert p is not None
    assert p.device == "cuda:1"
    assert p.device_index == 1


def test_resolve_cpu_override():
    placer = GpuPlacer()
    p = placer.resolve_override("cpu")
    assert p is not None
    assert p.device == "cpu"


def test_release_cuda_memory_safe():
    # Should not raise when CUDA is absent / unused
    release_cuda_memory()


def test_uuid_override_missing_raises():
    placer = GpuPlacer()
    with pytest.raises(ValueError, match="not visible"):
        placer.resolve_override("GPU-00000000-0000-0000-0000-000000000000")

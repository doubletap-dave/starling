import pytest

from starling.gpu.placer import DeviceInfo, GpuPlacer, normalize_gpu_uuid


def test_cpu_override():
    placer = GpuPlacer(device_override="cpu")
    p = placer.place(4096)
    assert p.device == "cpu"
    assert p.device_index is None


def test_cuda_override():
    placer = GpuPlacer(device_override="cuda:1")
    p = placer.place(4096)
    assert p.device == "cuda:1"
    assert p.device_index == 1


def test_reserve_and_release():
    placer = GpuPlacer(device_override="cpu")
    placer.reserve("cuda:0", 1000)
    placer.release("cuda:0", 1000)
    assert placer._reserved_mb.get("cuda:0") is None


def test_uuid_override_resolves(monkeypatch):
    placer = GpuPlacer(device_override="GPU-8ee934f5-a686-0f6f-8ff1-925a48f5e211")

    def fake_discover():
        return [
            DeviceInfo(
                index=0,
                name="RTX 4060",
                total_mb=8188,
                free_mb=7000,
                uuid="GPU-8ee934f5-a686-0f6f-8ff1-925a48f5e211",
            )
        ]

    monkeypatch.setattr(placer, "discover", fake_discover)
    p = placer.place(4096)
    assert p.device == "cuda:0"
    assert p.device_index == 0
    assert normalize_gpu_uuid(p.uuid or "") == "8ee934f5-a686-0f6f-8ff1-925a48f5e211"


def test_uuid_override_missing(monkeypatch):
    placer = GpuPlacer(device_override="GPU-00000000-0000-0000-0000-000000000000")
    monkeypatch.setattr(
        placer,
        "discover",
        lambda: [
            DeviceInfo(
                index=0,
                name="RTX 4060",
                total_mb=8188,
                free_mb=7000,
                uuid="GPU-8ee934f5-a686-0f6f-8ff1-925a48f5e211",
            )
        ],
    )
    with pytest.raises(ValueError, match="not visible"):
        placer.place(4096)

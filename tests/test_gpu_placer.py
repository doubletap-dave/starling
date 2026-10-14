from starling.gpu.placer import GpuPlacer


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

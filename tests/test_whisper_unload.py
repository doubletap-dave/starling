from types import SimpleNamespace

from starling.backends.whisper.backend import WhisperBackend


def test_whisper_unload_calls_ctranslate2_unload_model():
    calls: list[bool] = []

    class FakeCt2:
        def unload_model(self, to_cpu: bool = False) -> None:
            calls.append(to_cpu)

    handle = SimpleNamespace(model=FakeCt2())
    WhisperBackend().unload(handle)
    assert calls == [False]

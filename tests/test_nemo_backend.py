"""NeMo backend: legacy .nemo checkpoints must load under torch >= 2.6.

torch 2.6 flipped ``torch.load``'s ``weights_only`` default to True, which
breaks ``ASRModel.restore_from`` on legacy checkpoints (orukeet). The backend
scopes ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` to the restore call.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from starling.backends.nemo.backend import NemoBackend, _legacy_checkpoint_load_env
from starling.gpu.placer import Placement


def test_legacy_env_sets_and_restores_unset(monkeypatch):
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    with _legacy_checkpoint_load_env():
        assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ


def test_legacy_env_restores_previous_value(monkeypatch):
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "0")
    with _legacy_checkpoint_load_env():
        assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "0"


def test_legacy_env_restores_on_error(monkeypatch):
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    with pytest.raises(RuntimeError), _legacy_checkpoint_load_env():
        raise RuntimeError("boom")
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ


def test_load_scopes_legacy_env_to_restore_from(monkeypatch):
    seen: dict[str, str | None] = {}

    class FakeASRModel:
        @staticmethod
        def restore_from(restore_path, map_location=None):
            seen["env"] = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
            seen["path"] = restore_path
            model = types.SimpleNamespace()
            model.eval = lambda: None
            return model

    fake_nemo = types.ModuleType("nemo")
    fake_collections = types.ModuleType("nemo.collections")
    fake_asr = types.ModuleType("nemo.collections.asr")
    fake_models = types.ModuleType("nemo.collections.asr.models")
    fake_models.ASRModel = FakeASRModel
    for name, mod in {
        "nemo": fake_nemo,
        "nemo.collections": fake_collections,
        "nemo.collections.asr": fake_asr,
        "nemo.collections.asr.models": fake_models,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)

    backend = NemoBackend()
    handle = backend.load(
        entry=None,  # unused by NemoBackend.load
        local_path="/cache/orukeet-v0.1.0.nemo",
        placement=Placement(device="cpu", device_index=None, uuid=None),
    )

    assert handle is not None
    assert seen == {"env": "1", "path": "/cache/orukeet-v0.1.0.nemo"}
    # Scoped: not leaked into the process env after load.
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ

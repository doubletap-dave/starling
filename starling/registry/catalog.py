from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError

BackendName = Literal["whisper", "nemo", "transformers"]


@dataclass
class ModelEntry:
    id: str
    backend: BackendName
    repo: str
    file: str | None = None
    revision: str | None = None
    vram_mb: int = 4096
    requires_language: bool = False
    gated: bool = False
    languages: Any = "auto"
    supports_translate: bool = False
    supports_timestamps: bool = False
    license: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_id: str, data: dict[str, Any]) -> ModelEntry:
        known = {
            "backend",
            "repo",
            "file",
            "revision",
            "vram_mb",
            "requires_language",
            "gated",
            "languages",
            "supports_translate",
            "supports_timestamps",
            "license",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            id=model_id,
            backend=data["backend"],
            repo=data["repo"],
            file=data.get("file"),
            revision=data.get("revision"),
            vram_mb=int(data.get("vram_mb", 4096)),
            requires_language=bool(data.get("requires_language", False)),
            gated=bool(data.get("gated", False)),
            languages=data.get("languages", "auto"),
            supports_translate=bool(data.get("supports_translate", False)),
            supports_timestamps=bool(data.get("supports_timestamps", False)),
            license=data.get("license"),
            extra=extra,
        )


class ModelRegistry:
    def __init__(self, path: Path, token: str | None = None, cache_dir: Path | None = None) -> None:
        self.path = path
        self.token = token
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._models: dict[str, ModelEntry] = {}
        self.reload()

    def reload(self) -> None:
        with self.path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        models = data.get("models") or {}
        self._models = {mid: ModelEntry.from_dict(mid, meta) for mid, meta in models.items()}

    def get(self, model_id: str) -> ModelEntry | None:
        return self._models.get(model_id)

    def require(self, model_id: str) -> ModelEntry:
        entry = self.get(model_id)
        if entry is None:
            raise KeyError(model_id)
        return entry

    def all(self) -> list[ModelEntry]:
        return list(self._models.values())

    def ids(self) -> list[str]:
        return list(self._models.keys())

    def is_downloaded(self, entry: ModelEntry) -> bool:
        try:
            if entry.file:
                path = hf_hub_download(
                    repo_id=entry.repo,
                    filename=entry.file,
                    revision=entry.revision,
                    token=self.token,
                    cache_dir=self.cache_dir,
                    local_files_only=True,
                )
                return Path(path).exists()
            # Directory snapshot presence: try local-only snapshot
            snapshot_download(
                repo_id=entry.repo,
                revision=entry.revision,
                token=self.token,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )
            return True
        except Exception:
            return False

    def download(self, entry: ModelEntry) -> str:
        """Download model weights; returns local path (file or snapshot dir)."""
        if entry.file:
            return hf_hub_download(
                repo_id=entry.repo,
                filename=entry.file,
                revision=entry.revision,
                token=self.token,
                cache_dir=self.cache_dir,
            )
        return snapshot_download(
            repo_id=entry.repo,
            revision=entry.revision,
            token=self.token,
            cache_dir=self.cache_dir,
        )

    def local_path(self, entry: ModelEntry) -> str | None:
        try:
            return self.download(entry) if self.is_downloaded(entry) else None
        except EntryNotFoundError:
            return None

    def purge(self, entry: ModelEntry) -> None:
        """Best-effort: HuggingFace cache cleanup is complex; re-download after delete of snapshot is OK.

        We mark purge by removing the specific file when present; otherwise no-op with guidance.
        """
        path = self.local_path(entry)
        if path and entry.file:
            p = Path(path)
            if p.is_file():
                p.unlink(missing_ok=True)

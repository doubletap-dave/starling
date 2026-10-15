from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from huggingface_hub import hf_hub_download, scan_cache_dir, snapshot_download
from huggingface_hub.utils import EntryNotFoundError

logger = logging.getLogger(__name__)

BackendName = Literal["whisper", "nemo", "transformers"]
VALID_BACKENDS = frozenset({"whisper", "nemo", "transformers"})


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
    multi_gpu: bool = False
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
            "multi_gpu",
            "license",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        backend = data.get("backend")
        if backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {sorted(VALID_BACKENDS)}")
        repo = data.get("repo")
        if not repo:
            raise ValueError("repo is required")
        if backend == "nemo" and not data.get("file"):
            raise ValueError("file is required for nemo backend")
        return cls(
            id=model_id,
            backend=backend,
            repo=repo,
            file=data.get("file"),
            revision=data.get("revision"),
            vram_mb=int(data.get("vram_mb", 4096)),
            requires_language=bool(data.get("requires_language", False)),
            gated=bool(data.get("gated", False)),
            languages=data.get("languages", "auto"),
            supports_translate=bool(data.get("supports_translate", False)),
            supports_timestamps=bool(data.get("supports_timestamps", False)),
            multi_gpu=bool(data.get("multi_gpu", False)),
            license=data.get("license"),
            extra=extra,
        )

    def to_yaml_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "backend": self.backend,
            "repo": self.repo,
            "vram_mb": self.vram_mb,
            "requires_language": self.requires_language,
            "gated": self.gated,
            "languages": self.languages,
            "supports_translate": self.supports_translate,
            "supports_timestamps": self.supports_timestamps,
            "multi_gpu": self.multi_gpu,
        }
        if self.file:
            d["file"] = self.file
        if self.revision:
            d["revision"] = self.revision
        if self.license:
            d["license"] = self.license
        d.update(self.extra)
        return d


class ModelRegistry:
    def __init__(self, path: Path, token: str | None = None, cache_dir: Path | None = None) -> None:
        self.path = path
        self.token = token
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._models: dict[str, ModelEntry] = {}
        self.reload()

    def reload(self) -> None:
        if not self.path.exists():
            self._models = {}
            return
        with self.path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        models = data.get("models") or {}
        self._models = {mid: ModelEntry.from_dict(mid, meta) for mid, meta in models.items()}

    def save(self) -> None:
        payload = {"models": {mid: e.to_yaml_dict() for mid, e in self._models.items()}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
        tmp.replace(self.path)

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

    def create(self, model_id: str, data: dict[str, Any]) -> ModelEntry:
        if not model_id or "/" in model_id or "\\" in model_id:
            raise ValueError("invalid model id")
        if model_id in self._models:
            raise ValueError(f"model '{model_id}' already exists")
        entry = ModelEntry.from_dict(model_id, data)
        self._models[model_id] = entry
        self.save()
        return entry

    def update(self, model_id: str, data: dict[str, Any]) -> ModelEntry:
        if model_id not in self._models:
            raise KeyError(model_id)
        existing = self._models[model_id].to_yaml_dict()
        for k, v in data.items():
            if v is None and k in ("file", "revision", "license"):
                existing.pop(k, None)
            else:
                existing[k] = v
        entry = ModelEntry.from_dict(model_id, existing)
        self._models[model_id] = entry
        self.save()
        return entry

    def delete(self, model_id: str) -> None:
        if model_id not in self._models:
            raise KeyError(model_id)
        del self._models[model_id]
        self.save()

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

    def download(self, entry: ModelEntry, *, job_id: str | None = None) -> str:
        """Download model weights; returns local path (file or snapshot dir)."""
        tqdm_kwargs: dict[str, Any] = {}
        if job_id:
            from starling.hf_progress import make_tqdm_class

            tqdm_kwargs["tqdm_class"] = make_tqdm_class(job_id)

        if entry.file:
            try:
                return hf_hub_download(
                    repo_id=entry.repo,
                    filename=entry.file,
                    revision=entry.revision,
                    token=self.token,
                    cache_dir=self.cache_dir,
                    **tqdm_kwargs,
                )
            except TypeError:
                # Older huggingface_hub: no tqdm_class on hf_hub_download
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
            **tqdm_kwargs,
        )

    def local_path(self, entry: ModelEntry) -> str | None:
        try:
            return self.download(entry) if self.is_downloaded(entry) else None
        except EntryNotFoundError:
            return None

    def purge(self, entry: ModelEntry) -> dict[str, Any]:
        """Remove cached Hub files for this entry from disk.

        Keeps the registry catalog entry. Uses huggingface_hub's cache manager so
        snapshot models (Whisper/Transformers) and single-file NeMo weights are both
        removed — the previous implementation only unlinked ``entry.file`` and was a
        no-op for snapshot downloads.
        """
        result: dict[str, Any] = {
            "repo": entry.repo,
            "freed_bytes": 0,
            "revisions_deleted": 0,
            "method": None,
        }

        try:
            cache_info = scan_cache_dir(self.cache_dir)
        except Exception as exc:
            logger.warning("scan_cache_dir failed: %s", exc)
            cache_info = None

        if cache_info is not None:
            hashes: list[str] = []
            for repo in cache_info.repos:
                if repo.repo_id != entry.repo:
                    continue
                for rev in repo.revisions:
                    if self._revision_matches(rev, entry.revision):
                        hashes.append(rev.commit_hash)
            if hashes:
                strategy = cache_info.delete_revisions(*hashes)
                freed = int(getattr(strategy, "expected_freed_size", 0) or 0)
                strategy.execute()
                result.update(
                    {
                        "freed_bytes": freed,
                        "revisions_deleted": len(hashes),
                        "method": "hf_cache",
                    }
                )
                return result

        # Fallback: remove known local file or whole hub repo folder if present
        path = None
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
        except Exception:
            path = None

        if path:
            p = Path(path)
            if p.is_file():
                p.unlink(missing_ok=True)
                result.update({"freed_bytes": 0, "method": "unlink_file"})
                return result

        repo_dir = self._hub_repo_dir(entry.repo)
        if repo_dir is not None and repo_dir.is_dir():
            size = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
            shutil.rmtree(repo_dir, ignore_errors=True)
            result.update({"freed_bytes": size, "method": "rmtree", "path": str(repo_dir)})
            return result

        result["method"] = "noop"
        return result

    @staticmethod
    def _revision_matches(rev: Any, wanted: str | None) -> bool:
        if not wanted:
            return True
        commit = getattr(rev, "commit_hash", "") or ""
        if commit.startswith(wanted) or wanted.startswith(commit):
            return True
        refs = getattr(rev, "refs", None) or ()
        return wanted in refs

    def _hub_repo_dir(self, repo_id: str) -> Path | None:
        """Best-effort path to models--org--name under the Hub cache."""
        base = Path(self.cache_dir) if self.cache_dir else Path.home() / ".cache" / "huggingface" / "hub"
        # Newer layouts nest under hub/; older may already be the hub root
        candidates = [base, base / "hub"] if base.name != "hub" else [base]
        slug = "models--" + repo_id.replace("/", "--")
        for root in candidates:
            path = root / slug
            if path.is_dir():
                return path
        return None

    def entry_public_dict(self, entry: ModelEntry, *, downloaded: bool, loaded: bool) -> dict[str, Any]:
        return {
            "id": entry.id,
            "backend": entry.backend,
            "repo": entry.repo,
            "file": entry.file,
            "revision": entry.revision,
            "gated": entry.gated,
            "requires_language": entry.requires_language,
            "supports_translate": entry.supports_translate,
            "supports_timestamps": entry.supports_timestamps,
            "languages": entry.languages,
            "vram_mb": entry.vram_mb,
            "license": entry.license,
            "downloaded": downloaded,
            "loaded": loaded,
        }

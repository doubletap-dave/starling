from __future__ import annotations

import threading
from typing import Any, Iterable

from starling.progress import progress_store


def make_tqdm_class(job_id: str) -> type:
    """tqdm-compatible class that reports byte progress into ProgressStore."""

    class JobTqdm:
        def __init__(
            self,
            iterable: Iterable[Any] | None = None,
            *args: Any,
            total: int | float | None = None,
            desc: str | None = None,
            **kwargs: Any,
        ) -> None:
            self.iterable = iterable
            self.total = int(total or 0)
            self.n = 0
            self.desc = (desc or "").strip() or None
            self.disable = bool(kwargs.get("disable", False))
            if self.total > 0:
                job = progress_store.get(job_id)
                prior = job.bytes_total if job else 0
                progress_store.update(
                    job_id,
                    bytes_total=prior + self.total,
                    file_name=self.desc,
                    message=f"Downloading {self.desc}" if self.desc else "Downloading",
                )
            elif self.desc:
                progress_store.update(job_id, file_name=self.desc, message=f"Downloading {self.desc}")

        def update(self, n: int | float = 1) -> None:
            n_int = int(n)
            self.n += n_int
            progress_store.add_bytes(job_id, n_int, file_name=self.desc)

        def close(self) -> None:
            return None

        def __enter__(self) -> JobTqdm:
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

        def __iter__(self) -> Iterable[Any]:
            if self.iterable is None:
                return iter(())
            for item in self.iterable:
                yield item
                self.update(1)

        # Methods huggingface_hub / tqdm may call
        def set_description(self, desc: str | None = None, *args: Any, **kwargs: Any) -> None:
            if desc is not None:
                self.desc = str(desc)
            elif args:
                self.desc = str(args[0])
            if self.desc:
                progress_store.update(job_id, file_name=self.desc)

        def set_description_str(self, desc: str | None = None, *args: Any, **kwargs: Any) -> None:
            # huggingface_hub calls this (tqdm API) during multi-file downloads
            self.set_description(desc, *args, **kwargs)

        def set_postfix(self, *args: Any, **kwargs: Any) -> None:
            return None

        def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
            return None

        def refresh(self, *args: Any, **kwargs: Any) -> None:
            return None

        def clear(self, *args: Any, **kwargs: Any) -> None:
            return None

        def reset(self, total: int | float | None = None) -> None:
            if total is not None:
                self.total = int(total)
            self.n = 0

        def display(self, *args: Any, **kwargs: Any) -> None:
            return None

    return JobTqdm

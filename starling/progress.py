from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JobProgress:
    id: str
    model_id: str
    op: str  # download | load | unload | purge | other
    status: str = "running"  # running | done | error
    bytes_done: int = 0
    bytes_total: int = 0
    file_name: str | None = None
    message: str = ""
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    @property
    def percent(self) -> float | None:
        if self.bytes_total and self.bytes_total > 0:
            return max(0.0, min(100.0, 100.0 * self.bytes_done / self.bytes_total))
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "op": self.op,
            "status": self.status,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "percent": self.percent,
            "file_name": self.file_name,
            "message": self.message,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
            "elapsed_s": (self.finished or time.time()) - self.started,
        }


class ProgressStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, JobProgress] = {}

    def start(self, job_id: str, model_id: str, op: str, message: str = "") -> JobProgress:
        with self._lock:
            job = JobProgress(id=job_id, model_id=model_id, op=op, message=message)
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> JobProgress | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active_for_model(self, model_id: str, op: str | None = None) -> JobProgress | None:
        with self._lock:
            for job in self._jobs.values():
                if job.model_id != model_id or job.status != "running":
                    continue
                if op is None or job.op == op:
                    return job
            return None

    def list_active(self) -> list[JobProgress]:
        with self._lock:
            return [j for j in self._jobs.values() if j.status == "running"]

    def list_recent(self, limit: int = 20) -> list[JobProgress]:
        with self._lock:
            items = sorted(self._jobs.values(), key=lambda j: j.started, reverse=True)
            return items[:limit]

    def update(
        self,
        job_id: str,
        *,
        bytes_done: int | None = None,
        bytes_total: int | None = None,
        file_name: str | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if bytes_done is not None:
                job.bytes_done = bytes_done
            if bytes_total is not None and bytes_total > 0:
                job.bytes_total = bytes_total
            if file_name is not None:
                job.file_name = file_name
            if message is not None:
                job.message = message

    def add_bytes(self, job_id: str, n: int, total: int | None = None, file_name: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.bytes_done += n
            if total is not None and total > 0:
                # Multi-file downloads: keep a running total of the largest known file set
                job.bytes_total = max(job.bytes_total, total) if job.bytes_total else total
            if file_name is not None:
                job.file_name = file_name

    def finish(self, job_id: str, *, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.finished = time.time()
            if error:
                job.status = "error"
                job.error = error
                job.message = error
            else:
                job.status = "done"
                if job.bytes_total and job.bytes_done < job.bytes_total:
                    job.bytes_done = job.bytes_total
                job.message = job.message or "done"


progress_store = ProgressStore()

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WordStamp:
    word: str
    start: float
    end: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[WordStamp] = field(default_factory=list)


@dataclass
class TranscriptResult:
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[Segment] = field(default_factory=list)
    task: str = "transcribe"

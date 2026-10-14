from __future__ import annotations

from typing import Any

from starling.backends.base import TranscriptResult


def format_transcript(
    result: TranscriptResult,
    response_format: str = "json",
) -> Any:
    fmt = (response_format or "json").lower()
    if fmt == "text":
        return result.text
    if fmt == "srt":
        return _to_srt(result)
    if fmt == "vtt":
        return _to_vtt(result)
    if fmt == "verbose_json":
        return {
            "task": result.task,
            "language": result.language,
            "duration": result.duration,
            "text": result.text,
            "segments": [
                {
                    "id": s.id,
                    "seek": 0,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end} for w in s.words
                    ]
                    if s.words
                    else None,
                }
                for s in result.segments
            ],
        }
    # json (OpenAI default)
    return {"text": result.text}


def _ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    return _ts(seconds).replace(",", ".")


def _to_srt(result: TranscriptResult) -> str:
    lines: list[str] = []
    segs = result.segments or []
    if not segs and result.text:
        segs_data = [(0, 0.0, result.duration or 0.0, result.text)]
    else:
        segs_data = [(s.id, s.start, s.end, s.text) for s in segs]
    for i, start, end, text in segs_data:
        lines.append(str(i + 1))
        lines.append(f"{_ts(start)} --> {_ts(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _to_vtt(result: TranscriptResult) -> str:
    lines = ["WEBVTT", ""]
    segs = result.segments or []
    if not segs and result.text:
        segs_data = [(0.0, result.duration or 0.0, result.text)]
    else:
        segs_data = [(s.start, s.end, s.text) for s in segs]
    for start, end, text in segs_data:
        lines.append(f"{_ts_vtt(start)} --> {_ts_vtt(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)

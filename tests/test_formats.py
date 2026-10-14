from starling.api.formats import format_transcript
from starling.backends.base import Segment, TranscriptResult, WordStamp


def test_json_format():
    r = TranscriptResult(text="hello world")
    assert format_transcript(r, "json") == {"text": "hello world"}


def test_text_format():
    r = TranscriptResult(text="hello")
    assert format_transcript(r, "text") == "hello"


def test_verbose_json_with_words():
    r = TranscriptResult(
        text="hi",
        language="en",
        duration=1.0,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=1.0,
                text="hi",
                words=[WordStamp(word="hi", start=0.0, end=1.0)],
            )
        ],
    )
    out = format_transcript(r, "verbose_json")
    assert out["text"] == "hi"
    assert out["segments"][0]["words"][0]["word"] == "hi"


def test_srt_and_vtt():
    r = TranscriptResult(
        text="hi",
        duration=1.0,
        segments=[Segment(id=0, start=0.0, end=1.0, text="hi")],
    )
    srt = format_transcript(r, "srt")
    assert "00:00:00,000 --> 00:00:01,000" in srt
    vtt = format_transcript(r, "vtt")
    assert vtt.startswith("WEBVTT")

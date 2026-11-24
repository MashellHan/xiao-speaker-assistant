import json

from xiao_speaker_assistant.voice.audio_pipeline import extract_record_pcm_from_binary


def test_extract_record_pcm_from_binary_decodes_record_stream() -> None:
    frame = json.dumps(
        {
            "id": "s-1",
            "tag": "record",
            "bytes": [1, 0, 255, 127],
        }
    ).encode("utf-8")
    assert extract_record_pcm_from_binary(frame) == b"\x01\x00\xff\x7f"


def test_extract_record_pcm_from_binary_ignores_non_record_tags() -> None:
    frame = json.dumps({"id": "s-2", "tag": "play", "bytes": [1, 2]}).encode("utf-8")
    assert extract_record_pcm_from_binary(frame) is None
    assert extract_record_pcm_from_binary(b"not-json") is None

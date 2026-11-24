import json
from pathlib import Path
from unittest.mock import patch

from xiao_speaker_assistant.voice.audio_pipeline import extract_record_pcm_from_binary
from xiao_speaker_assistant.voice.models_setup import generate_keywords_file


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


def test_generate_keywords_file_replaces_spaces_in_tag(tmp_path: Path) -> None:
    (tmp_path / "tokens.txt").write_text("dummy\n", encoding="utf-8")
    (tmp_path / "bpe.model").write_text("dummy\n", encoding="utf-8")
    with patch("sherpa_onnx.text2token", return_value=[["▁HI", "▁NO", "M", "I"], ["你", "好", "管", "家"]]):
        out = generate_keywords_file(tmp_path, ["hi nomi", "你好管家"])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == ["▁HI ▁NO M I @hi_nomi", "你 好 管 家 @你好管家"]

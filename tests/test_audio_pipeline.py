import json
import struct
from pathlib import Path
from unittest.mock import patch

import numpy as np

from xiao_speaker_assistant.voice.audio_pipeline import (
    extract_record_pcm_from_binary,
    pcm_to_float32,
)
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


def test_pcm_to_float32_scales_a113_s32_lower_24_bits() -> None:
    raw = struct.pack("<ii", 4456448, -2883584)
    samples = pcm_to_float32(raw, bits_per_sample=32)
    assert np.allclose(samples, [17408 / 32768.0, -11264 / 32768.0], atol=1e-5)

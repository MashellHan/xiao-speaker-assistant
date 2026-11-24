import json
import struct
from pathlib import Path
from unittest.mock import patch

import numpy as np

from xiao_speaker_assistant.voice.audio_pipeline import (
    apply_highpass_filter,
    extract_record_pcm_from_binary,
    match_wakeup_phrase,
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


def test_pcm_to_float32_scales_16bit_and_32bit_cleanly() -> None:
    raw16 = struct.pack("<hh", 16384, -8192)
    samples16 = pcm_to_float32(raw16, bits_per_sample=16)
    assert np.allclose(samples16, [0.5, -0.25], atol=1e-5)

    raw32 = struct.pack("<ii", 16384 << 16, (-8192) << 16)
    samples32 = pcm_to_float32(raw32, bits_per_sample=32, s32_shift=16)
    assert np.allclose(samples32, [0.5, -0.25], atol=1e-5)


def test_match_wakeup_phrase_handles_hi_nomi_variants_and_inline_queries() -> None:
    assert match_wakeup_phrase("Hi, Nomi.") == ("hi nomi", "")
    assert match_wakeup_phrase("H nomi.") == ("hi nomi", "")
    assert match_wakeup_phrase("Hnomi.") == ("hi nomi", "")
    assert match_wakeup_phrase("Hi Nami.") == ("hi nomi", "")
    assert match_wakeup_phrase("i know me") == ("hi nomi", "")
    assert match_wakeup_phrase("Can know me.") == ("hi nomi", "")
    assert match_wakeup_phrase("蔡纳米三句话介绍一下太阳系") == ("hi nomi", "三句话介绍一下太阳系")
    assert match_wakeup_phrase("陈老明三句话介绍一下叹阳系。") == ("hi nomi", "三句话介绍一下叹阳系")
    assert match_wakeup_phrase("陈老师你好") is None
    assert match_wakeup_phrase("海诺米，帮我查一下") == ("hi nomi", "帮我查一下")
    assert match_wakeup_phrase("今天天气怎么样") is None


def test_apply_highpass_filter_removes_dc_offset() -> None:
    dc = np.full(1600, 0.25, dtype=np.float32)
    filtered, _, _ = apply_highpass_filter(dc)
    assert abs(float(filtered[-1])) < 1e-4
    assert match_wakeup_phrase("Hi, Normi.") == ("hi nomi", "")
    assert match_wakeup_phrase("Hi, knowmi. Tell me a joke.") == ("hi nomi", "Tell me a joke")
    assert match_wakeup_phrase("Hi Nomi，帮我查一下邮件") == ("hi nomi", "帮我查一下邮件")
    assert match_wakeup_phrase("你好管家", ["你好管家"]) == ("你好管家", "")
    assert match_wakeup_phrase("打开电视", ["你好管家"]) is None

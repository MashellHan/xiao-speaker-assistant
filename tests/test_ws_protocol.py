import json

from xiao_speaker_assistant.voice.ws_server import SpeakerGateway


def test_extract_final_asr_parses_open_xiaoai_newline_event() -> None:
    line = json.dumps(
        {
            "header": {
                "namespace": "SpeechRecognizer",
                "name": "RecognizeResult",
                "dialog_id": "dlg-123",
                "id": "msg-1",
            },
            "payload": {
                "is_final": True,
                "is_vad_begin": False,
                "results": [{"text": "帮我介绍一下量子计算", "confidence": 0.99}],
            },
        },
        ensure_ascii=False,
    )
    event = {
        "Event": {
            "id": "evt-1",
            "event": "instruction",
            "data": {"NewLine": line},
        }
    }
    assert SpeakerGateway._extract_final_asr(event) == ("dlg-123", "帮我介绍一下量子计算")


def test_extract_final_asr_ignores_partial_results() -> None:
    line = json.dumps(
        {
            "header": {
                "namespace": "SpeechRecognizer",
                "name": "RecognizeResult",
                "dialog_id": "dlg-123",
                "id": "msg-1",
            },
            "payload": {
                "is_final": False,
                "is_vad_begin": False,
                "results": [{"text": "帮我"}],
            },
        },
        ensure_ascii=False,
    )
    event = {"Event": {"id": "evt-1", "event": "instruction", "data": {"NewLine": line}}}
    assert SpeakerGateway._extract_final_asr(event) is None


def test_should_ignore_bare_wake_word() -> None:
    assert SpeakerGateway.should_ignore_utterance("小爱同学") is True
    assert SpeakerGateway.should_ignore_utterance("小爱同学。") is True
    assert SpeakerGateway.should_ignore_utterance("介绍一下自己") is False


def test_detects_native_speak_event() -> None:
    line = json.dumps({"header": {"namespace": "SpeechSynthesizer", "name": "Speak"}})
    event = {"Event": {"id": "evt-2", "event": "instruction", "data": {"NewLine": line}}}
    assert SpeakerGateway._is_native_speak_event(event) is True

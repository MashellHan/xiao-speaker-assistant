import json
from unittest.mock import patch

from xiao_speaker_assistant import cli
from xiao_speaker_assistant.llm.router import parse_opencode_jsonl


def test_parse_opencode_jsonl_collects_text_parts() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "step_start", "sessionID": "ses_1"}),
            json.dumps({"type": "text", "part": {"type": "text", "text": "你好，"}}),
            json.dumps({"type": "text", "part": {"type": "text", "text": "我是 Nomi。"}}),
            json.dumps({"type": "step_finish", "reason": "stop"}),
        ]
    )
    assert parse_opencode_jsonl(stdout) == "你好，\n我是 Nomi。"


def test_cli_serve_wires_relay_and_asr_model_flags() -> None:
    captured = {}

    async def fake_run_with_relay(config, on_query, relay_url, **kwargs):
        captured["asr"] = config.speech.asr_provider
        captured["asr_model"] = config.speech.asr_model
        captured["relay_url"] = relay_url
        captured["also_serve_local"] = kwargs.get("also_serve_local")

    with patch.object(cli, "run_with_relay", fake_run_with_relay):
        rc = cli.main(
            [
                "serve",
                "--asr",
                "sense_voice",
                "--asr-model",
                "paraformer",
                "--relay",
                "wss://relay.example.com",
            ]
        )
    assert rc == 0
    assert captured == {
        "asr": "sense_voice",
        "asr_model": "paraformer",
        "relay_url": "wss://relay.example.com",
        "also_serve_local": True,
    }

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


def test_llm_router_tracks_opencode_session_and_resets() -> None:
    import asyncio
    from xiao_speaker_assistant.config import LLMConfig
    from xiao_speaker_assistant.llm.router import LLMRouter

    router = LLMRouter(LLMConfig())
    calls = []

    class FakeProc:
        def __init__(self, sid: str, reply: str) -> None:
            lines = [
                json.dumps({"type": "step_start", "sessionID": sid}).encode() + b"\n",
                json.dumps({"type": "text", "part": {"type": "text", "text": reply}}).encode() + b"\n",
                b"",
            ]
            self.stdout = self
            self.stderr = self
            self._lines = lines
            self.returncode = 0

        async def readline(self) -> bytes:
            return self._lines.pop(0)

        async def wait(self) -> int:
            return 0

        async def read(self) -> bytes:
            return b""

    async def fake_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        return FakeProc("ses_abc123", "北京今天晴，12度。")

    async def run() -> None:
        with patch("xiao_speaker_assistant.llm.router.resolve_binary", return_value="/usr/local/bin/opencode"), patch(
            "asyncio.create_subprocess_exec", side_effect=fake_exec
        ):
            r1 = await router.ask("查一下天气")
            r2 = await router.ask("北京")
            assert "12度" in r1
            assert "12度" in r2
            assert "--archived" in calls[0]
            assert "--session" in calls[1] and "ses_abc123" in calls[1]
            router.reset_session()
            assert router.opencode_session_id is None
            assert router.history == []

    asyncio.run(run())

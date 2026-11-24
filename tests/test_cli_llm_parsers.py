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

    class FakeRunner:
        async def cleanup(self) -> None:
            pass

    async def fake_start_http(self, host="127.0.0.1", port=4397):
        return FakeRunner(), 14397

    async def fake_router_start(self, *, mcp_url=None) -> None:
        captured["mcp_url"] = mcp_url

    async def fake_router_close(self) -> None:
        pass

    async def fake_run_with_relay(config, on_query, relay_url, **kwargs):
        captured["asr"] = config.speech.asr_provider
        captured["asr_model"] = config.speech.asr_model
        captured["relay_url"] = relay_url
        captured["also_serve_local"] = kwargs.get("also_serve_local")

    with (
        patch.object(cli.McpServer, "start_http_server", fake_start_http),
        patch.object(cli.LLMRouter, "start", fake_router_start),
        patch.object(cli.LLMRouter, "close", fake_router_close),
        patch.object(cli, "run_with_relay", fake_run_with_relay),
    ):
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
        "mcp_url": "http://127.0.0.1:14397/mcp",
        "asr": "sense_voice",
        "asr_model": "paraformer",
        "relay_url": "wss://relay.example.com",
        "also_serve_local": True,
    }


def test_llm_router_tracks_opencode_session_and_resets() -> None:
    import asyncio
    from xiao_speaker_assistant.config import LLMConfig, LLMProvidersConfig, OpencodeCliConfig
    from xiao_speaker_assistant.llm.router import LLMRouter

    cfg = LLMConfig(providers=LLMProvidersConfig(opencode=OpencodeCliConfig(server_url="")))
    router = LLMRouter(cfg)
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

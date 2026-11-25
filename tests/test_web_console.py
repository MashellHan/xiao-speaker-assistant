from __future__ import annotations

import asyncio
import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml
from aiohttp.test_utils import make_mocked_request

from xiao_speaker_assistant import cli
from xiao_speaker_assistant.config import (
    AppConfig,
    McpEndpointConfig,
    OutlookToolConfig,
    SpeechConfig,
    ToolsConfig,
    load_config,
)
from xiao_speaker_assistant.tools import ConversationHistoryStore, McpServer, MemoryStore


def test_conversation_history_store_permissions_stats_and_cap(tmp_path: Path) -> None:
    hist_path = tmp_path / "data" / "history.json"
    store = ConversationHistoryStore(hist_path, max_entries=4)
    assert store.stats() == {
        "total_turns": 0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "turns_today": 0,
        "speaker_breakdown": {},
        "model_breakdown": {},
    }

    now = time.time()
    yesterday = now - 86400 * 2
    store.record(
        "今天天气",
        "北京晴",
        latency_ms=100.0,
        model="raven/gemini-4-hydrogen",
        voice="zh-CN-XiaoxiaoNeural",
        speaker_name="大雄",
        speaker_role="爸爸",
        source="voice",
        timestamp=yesterday,
    )
    store.record(
        "明天呢",
        "多云",
        latency_ms=200.0,
        model="raven/gemini-4-hydrogen",
        voice="zh-CN-XiaoxiaoNeural",
        speaker_name="大雄",
        speaker_role="爸爸",
        source="voice",
        timestamp=now,
    )
    store.record(
        "讲个故事",
        "从前有座山",
        latency_ms=300.0,
        model="raven/gpt-6.1-sol",
        voice="zh-CN-XiaoyiNeural",
        speaker_name="拳拳",
        speaker_role="宝宝",
        source="voice",
        timestamp=now,
    )
    store.record(
        "再讲一个",
        "小兔子乖乖",
        latency_ms=400.0,
        model="raven/gpt-6.1-sol",
        voice="zh-CN-XiaoyiNeural",
        speaker_name="拳拳",
        speaker_role="宝宝",
        source="bridge",
        timestamp=now,
    )
    # 5th entry evicts 1st (max_entries=4)
    store.record(
        "晚安",
        "晚安好梦",
        latency_ms=500.0,
        model="raven/gpt-5.4-nano",
        voice="zh-CN-YunxiNeural",
        speaker_name="大雄",
        speaker_role="爸爸",
        source="cli",
        timestamp=now,
    )

    assert stat.S_IMODE(hist_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(hist_path.parent.stat().st_mode) == 0o700

    reloaded = ConversationHistoryStore(hist_path, max_entries=4)
    recent = reloaded.list(limit=10)
    assert len(recent) == 4
    assert recent[0].query == "晚安"
    assert recent[-1].query == "明天呢"

    filtered = reloaded.list(limit=10, speaker="拳拳")
    assert len(filtered) == 2
    assert all(e.speaker_name == "拳拳" for e in filtered)

    stats = reloaded.stats()
    assert stats["total_turns"] == 4
    assert stats["turns_today"] == 4
    assert stats["avg_latency_ms"] == 350.0
    assert stats["p95_latency_ms"] == 500.0
    assert stats["speaker_breakdown"] == {"大雄": 2, "拳拳": 2}
    assert stats["model_breakdown"] == {
        "raven/gemini-4-hydrogen": 1,
        "raven/gpt-6.1-sol": 2,
        "raven/gpt-5.4-nano": 1,
    }

    cleared = reloaded.clear()
    assert cleared == 4
    assert reloaded.list() == []
    assert reloaded.stats()["total_turns"] == 0
    assert stat.S_IMODE(hist_path.stat().st_mode) == 0o600


def test_web_console_endpoints_settings_memories_and_mcp_auth(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "speaker": {"edge_voice": "zh-CN-XiaoxiaoNeural", "edge_rate": "+8%"},
                "speech": {"aec_enabled": True, "capture_channels": 2, "asr_model": "paraformer"},
                "tools": {"xiaohongshu": {"enabled": True, "mcp_url": "http://127.0.0.1:18060/mcp"}},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)

    # Ensure secrets in env are never exposed by any API endpoint
    monkeypatch.setenv("OUTLOOK_CLIENT_ID", "super-secret-outlook-client-id")
    bearer_secret = "mcp-bearer-secret-token-xyz"

    mem_store = MemoryStore(tmp_path / "memory.json")
    hist_store = ConversationHistoryStore(tmp_path / "history.json")
    hist_store.record(
        "今天天气怎么样",
        "北京今天晴朗，气温12度。",
        latency_ms=245.5,
        model=cfg.llm.providers.opencode.model,
        voice=cfg.speaker.edge_voice,
        speaker_name="大雄",
        speaker_role="主人",
        source="voice",
    )

    speakers_path = tmp_path / "speakers.json"
    speakers_path.write_text(
        json.dumps(
            [
                {"id": "spk_1", "name": "大雄", "role": "爸爸", "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
                {"id": "spk_2", "name": "拳拳", "role": "宝宝", "sample_count": 3},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reminders_path = tmp_path / "reminders.json"
    reminders_path.write_text(
        json.dumps(
            [{"id": "rem_1", "message": "晚上八点喝牛奶", "trigger_at": "20:00", "enabled": True}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    voice_callbacks: list[tuple[str, str]] = []
    model_callbacks: list[str] = []
    memory_callbacks = 0

    def on_mem() -> None:
        nonlocal memory_callbacks
        memory_callbacks += 1

    server = McpServer(
        cfg,
        memory_store=mem_store,
        history_store=hist_store,
        on_memory_changed=on_mem,
        on_voice_changed=lambda v, r: voice_callbacks.append((v, r)),
        on_model_changed=lambda m: model_callbacks.append(m),
        config_path=cfg_path,
        speakers_path=speakers_path,
        reminders_path=reminders_path,
        include_local_tools=True,
    )

    async def run() -> None:
        runner, port = await server.start_http_server(
            host="127.0.0.1",
            port=0,
            auth_token=bearer_secret,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient() as client:
                # 1. GET / serves console HTML with Furrow tokens & no auth required
                idx = await client.get(f"{base}/")
                assert idx.status_code == 200
                assert "text/html" in idx.headers["content-type"]
                assert "Nomi 控制台" in idx.text
                assert "--color-paper" in idx.text
                assert "tabular-nums" in idx.text
                assert "prefers-reduced-motion" in idx.text

                # 2. /mcp strictly enforces Bearer auth & rejects Origin
                unauth_mcp = await client.post(
                    f"{base}/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                )
                assert unauth_mcp.status_code == 401
                bad_mcp = await client.post(
                    f"{base}/mcp",
                    headers={"Authorization": "Bearer wrong-token"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                )
                assert bad_mcp.status_code == 401
                origin_mcp = await client.post(
                    f"{base}/mcp",
                    headers={"Origin": "https://evil.example", "Authorization": f"Bearer {bearer_secret}"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                )
                assert origin_mcp.status_code == 403
                ok_mcp = await client.post(
                    f"{base}/mcp",
                    headers={"Authorization": f"Bearer {bearer_secret}"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                )
                assert ok_mcp.status_code == 200

                # 3. GET /api/overview returns real runtime state & zero secrets
                ov = await client.get(f"{base}/api/overview")
                assert ov.status_code == 200
                ov_data = ov.json()
                assert ov_data["model"] == "raven/gemini-4-hydrogen"
                assert ov_data["voice"] == "zh-CN-XiaoxiaoNeural"
                assert ov_data["asr_model"] == "paraformer"
                assert ov_data["aec_enabled"] is True
                assert ov_data["stats"]["total_turns"] == 1
                cap_ids = {c["id"] for c in ov_data["capabilities"]}
                assert cap_ids == {
                    "weather",
                    "web_search",
                    "memory",
                    "voice_switching",
                    "xiaohongshu",
                    "browser",
                    "outlook",
                    "codex",
                    "mac_local",
                }
                xhs_cap = next(c for c in ov_data["capabilities"] if c["id"] == "xiaohongshu")
                assert xhs_cap["enabled"] is True
                assert "super-secret-outlook-client-id" not in ov.text
                assert bearer_secret not in ov.text

                # 4. GET /api/history & filter & DELETE /api/history
                hist_resp = await client.get(f"{base}/api/history?limit=10&speaker=大雄")
                assert hist_resp.status_code == 200
                hist_json = hist_resp.json()
                assert hist_json["count"] == 1
                assert hist_json["turns"][0]["query"] == "今天天气怎么样"
                assert hist_json["turns"][0]["latency_ms"] == 245.5

                # Cross-origin write must be rejected
                evil_del = await client.delete(
                    f"{base}/api/history",
                    headers={"Origin": "https://evil.example"},
                )
                assert evil_del.status_code == 403

                # 5. Memories CRUD via /api/memories
                add_fact = await client.post(
                    f"{base}/api/memories",
                    json={"fact": "常住城市：上海"},
                )
                assert add_fact.status_code == 201
                mem_item = add_fact.json()["memory"]
                assert mem_item["key"] == "常住城市"
                assert mem_item["value"] == "上海"
                assert memory_callbacks == 1

                rejected_secret = await client.post(
                    f"{base}/api/memories",
                    json={"fact": "password: my-secret-value"},
                )
                assert rejected_secret.status_code == 400

                list_mem = await client.get(f"{base}/api/memories")
                assert list_mem.status_code == 200
                assert list_mem.json()["count"] == 1

                del_mem = await client.delete(f"{base}/api/memories/{mem_item['id']}")
                assert del_mem.status_code == 200
                assert del_mem.json()["removed"] is True
                assert memory_callbacks == 2

                del_missing = await client.delete(f"{base}/api/memories/nonexistent")
                assert del_missing.status_code == 404

                # 6. POST /api/settings updates model, voice, rate, persists atomically, fires callbacks
                upd = await client.post(
                    f"{base}/api/settings",
                    json={
                        "model": "raven/gemini-4-hydrogen#medium",
                        "edge_voice": "晓伊",
                        "edge_rate": "+15%",
                    },
                )
                assert upd.status_code == 200
                upd_json = upd.json()
                assert upd_json["model"] == "raven/gemini-4-hydrogen#medium"
                assert upd_json["edge_voice"] == "zh-CN-XiaoyiNeural"
                assert upd_json["edge_rate"] == "+15%"
                assert upd_json["persisted"] is True
                assert voice_callbacks == [("zh-CN-XiaoyiNeural", "+15%")]
                assert model_callbacks == ["raven/gemini-4-hydrogen#medium"]

                persisted_cfg = load_config(cfg_path)
                assert persisted_cfg.llm.providers.opencode.model == "raven/gemini-4-hydrogen#medium"
                assert persisted_cfg.speaker.edge_voice == "zh-CN-XiaoyiNeural"
                assert persisted_cfg.speaker.edge_rate == "+15%"
                assert persisted_cfg.speech.aec_enabled is True
                assert persisted_cfg.speech.asr_model == "paraformer"

                # Invalid settings rejected cleanly
                bad_model = await client.post(
                    f"{base}/api/settings",
                    json={"model": "not valid model!"},
                )
                assert bad_model.status_code == 400

                # 7. Voiceprints & reminders endpoints
                vp_resp = await client.get(f"{base}/api/voiceprints")
                assert vp_resp.status_code == 200
                vp_list = vp_resp.json()["voiceprints"]
                assert len(vp_list) == 2
                assert "embeddings" not in vp_list[0]
                assert vp_list[0]["sample_count"] == 2

                del_vp = await client.delete(f"{base}/api/voiceprints/spk_1")
                assert del_vp.status_code == 200
                assert stat.S_IMODE(speakers_path.stat().st_mode) == 0o600
                vp_after = await client.get(f"{base}/api/voiceprints")
                assert vp_after.json()["count"] == 1

                rem_resp = await client.get(f"{base}/api/reminders")
                assert rem_resp.status_code == 200
                assert rem_resp.json()["count"] == 1
                assert rem_resp.json()["reminders"][0]["message"] == "晚上八点喝牛奶"

                # Clear history via DELETE /api/history
                del_hist = await client.delete(f"{base}/api/history")
                assert del_hist.status_code == 200
                assert del_hist.json()["cleared"] == 1
                assert del_hist.json()["stats"]["total_turns"] == 0
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_web_console_graceful_missing_voiceprints_and_reminders(tmp_path: Path) -> None:
    server = McpServer(
        AppConfig(),
        memory_store=MemoryStore(tmp_path / "memory.json"),
        history_store=ConversationHistoryStore(tmp_path / "history.json"),
        speakers_path=tmp_path / "missing_speakers.json",
        reminders_path=tmp_path / "missing_reminders.json",
    )

    async def run() -> None:
        runner, port = await server.start_http_server(host="127.0.0.1", port=0, auth_token="tok")
        try:
            async with httpx.AsyncClient() as client:
                vp = await client.get(f"http://127.0.0.1:{port}/api/voiceprints")
                assert vp.status_code == 200
                assert vp.json() == {"voiceprints": [], "count": 0}

                rem = await client.get(f"http://127.0.0.1:{port}/api/reminders")
                assert rem.status_code == 200
                assert rem.json() == {"reminders": [], "count": 0}
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_write_api_rejects_non_loopback_without_bearer(tmp_path: Path) -> None:
    server = McpServer(
        AppConfig(),
        memory_store=MemoryStore(tmp_path / "memory.json"),
        history_store=ConversationHistoryStore(tmp_path / "history.json"),
    )

    async def run() -> None:
        runner, port = await server.start_http_server(host="127.0.0.1", port=0, auth_token="secret-tok")
        # Extract the aiohttp app directly to test non-loopback request.remote
        app = runner.app
        handler = None
        for route in app.router.routes():
            if route.method == "POST" and getattr(route.resource, "canonical", None) == "/api/settings":
                handler = route.handler
                break
        assert handler is not None

        remote_req = make_mocked_request("POST", "/api/settings", headers={})
        # Override transport/remote simulation via clone/mocked peername
        class RemoteReqProxy:
            def __init__(self, inner) -> None:
                self._inner = inner
                self.headers = inner.headers
                self.remote = "192.168.1.50"

            async def json(self):
                return {"model": "raven/gpt-5.4-nano"}

        resp_denied = await handler(RemoteReqProxy(remote_req))
        assert resp_denied.status == 403

        auth_req = make_mocked_request(
            "POST",
            "/api/settings",
            headers={"Authorization": "Bearer secret-tok"},
        )

        class AuthRemoteReqProxy:
            def __init__(self, inner) -> None:
                self._inner = inner
                self.headers = inner.headers
                self.remote = "192.168.1.50"

            async def json(self):
                return {"model": "raven/gpt-5.4-nano"}

        resp_ok = await handler(AuthRemoteReqProxy(auth_req))
        assert resp_ok.status == 200
        await runner.cleanup()

    asyncio.run(run())


def test_cli_serve_records_history_on_handle_query_and_stream_query(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "llm": {"providers": {"opencode": {"agent_home": str(tmp_path / "agent-home")}}},
                "memory": {"file_path": str(tmp_path / "memory.json")},
            }
        ),
        encoding="utf-8",
    )
    hist_path = tmp_path / "history.json"
    recorded_store = ConversationHistoryStore(hist_path)

    class FakeRunner:
        async def cleanup(self) -> None:
            pass

    captured_handlers: dict[str, object] = {}

    async def fake_start_http(self, host="127.0.0.1", port=4397, *, auth_token=None):
        captured_handlers["mcp_server"] = self
        return FakeRunner(), 14397

    async def fake_router_start(self, *, mcp_url=None, mcp_auth_token=None) -> None:
        pass

    async def fake_router_close(self) -> None:
        pass

    async def fake_ask(self, query: str, **kwargs) -> str:
        return f"回答：{query}"

    async def fake_stream_ask(self, query: str, **kwargs):
        yield "第一句。"
        yield "第二句。"

    async def fake_run_with_relay(config, on_query, relay_url, **kwargs):
        r1 = await on_query("你好")
        assert r1 == "回答：你好"
        stream_fn = kwargs["on_stream_query"]
        parts = [s async for s in stream_fn("讲故事")]
        assert parts == ["第一句。", "第二句。"]

    with (
        patch.object(cli, "ConversationHistoryStore", return_value=recorded_store),
        patch.object(cli.McpServer, "start_http_server", fake_start_http),
        patch.object(cli.LLMRouter, "start", fake_router_start),
        patch.object(cli.LLMRouter, "close", fake_router_close),
        patch.object(cli.LLMRouter, "ask", fake_ask),
        patch.object(cli.LLMRouter, "stream_ask", fake_stream_ask),
        patch.object(cli, "run_with_relay", fake_run_with_relay),
    ):
        rc = cli.main(["--config", str(cfg_path), "serve", "--relay", "wss://relay.example.com"])

    assert rc == 0
    entries = recorded_store.list(limit=10)
    assert len(entries) == 2
    assert entries[0].query == "讲故事"
    assert entries[0].reply == "第一句。第二句。"
    assert entries[0].source == "voice"
    assert entries[1].query == "你好"
    assert entries[1].reply == "回答：你好"
    assert captured_handlers["mcp_server"].history_store is recorded_store  # type: ignore[attr-defined]

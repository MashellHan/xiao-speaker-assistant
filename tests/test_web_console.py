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
                assert ov_data["model"] == "raven/gemini-4-hydrogen#low"
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
                rem_json = rem.json()
                assert rem_json["reminders"] == []
                assert rem_json["count"] == 0
                assert rem_json["enabled"] is True
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
        assert resp_denied.status in (401, 403)

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
    from xiao_speaker_assistant.tools import TurnTraceStore

    hist_path = tmp_path / "history.json"
    recorded_store = ConversationHistoryStore(hist_path)
    recorded_traces = TurnTraceStore(tmp_path / "traces.json")

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
        patch.object(cli, "TurnTraceStore", return_value=recorded_traces),
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


def test_voice_model_switching_and_remote_tunnel_login_gate(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "llm:\n  providers:\n    opencode:\n      model: raven/gemini-4-hydrogen#low\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.llm.providers.opencode.model == "raven/gemini-4-hydrogen#low"

    model_changes: list[str] = []
    srv = McpServer(
        cfg,
        config_path=cfg_path,
        on_model_changed=lambda m: model_changes.append(m),
        include_local_tools=True,
    )

    # 1. Voice MCP tools: speaker_list_models + speaker_set_model("gpt 6.2")
    list_res = asyncio.run(
        srv.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "speaker_list_models", "arguments": {}}}
        )
    )
    assert list_res is not None
    list_data = json.loads(list_res["result"]["content"][0]["text"])
    assert list_data["current_model"] == "raven/gemini-4-hydrogen#low"

    set_res = asyncio.run(
        srv.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "speaker_set_model", "arguments": {"model": "gpt 6.2"}},
            }
        )
    )
    assert set_res is not None and set_res["result"]["isError"] is False
    set_data = json.loads(set_res["result"]["content"][0]["text"])
    assert set_data["model"] == "raven/gpt-6.2-sol#low"
    assert model_changes == ["raven/gpt-6.2-sol#low"]
    assert load_config(cfg_path).llm.providers.opencode.model == "raven/gpt-6.2-sol#low"

    # 2. Remote proxy / Cloudflare Tunnel auth gate + /mcp isolation
    async def run_http() -> None:
        runner, port = await srv.start_http_server(
            host="127.0.0.1",
            port=0,
            auth_token="mcp-secret-token",
            console_password="remote-console-password",
        )
        base = f"http://127.0.0.1:{port}"
        tunnel_headers = {"Host": "nomi.mxhan.com", "CF-Connecting-IP": "198.51.100.2"}
        try:
            async with httpx.AsyncClient() as client:
                # /mcp rejects any proxied request even with valid Bearer token
                mcp_proxied = await client.post(
                    f"{base}/mcp",
                    headers={**tunnel_headers, "Authorization": "Bearer mcp-secret-token"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                )
                assert mcp_proxied.status_code == 403

                # /api/overview requires login when accessed through Cloudflare / proxy headers
                auth_status = await client.get(f"{base}/api/auth/status", headers=tunnel_headers)
                assert auth_status.status_code == 200
                assert auth_status.json()["authenticated"] is False
                assert auth_status.json()["auth_required"] is True

                unauth_ov = await client.get(f"{base}/api/overview", headers=tunnel_headers)
                assert unauth_ov.status_code == 401

                # Wrong password fails with 401, then valid login unlocks with HttpOnly cookie
                bad_login = await client.post(
                    f"{base}/api/login",
                    headers=tunnel_headers,
                    json={"password": "wrong"},
                )
                assert bad_login.status_code == 401

                ok_login = await client.post(
                    f"{base}/api/login",
                    headers=tunnel_headers,
                    json={"password": "remote-console-password"},
                )
                assert ok_login.status_code == 200
                assert "xsa_console_session" in ok_login.cookies

                authed_ov = await client.get(
                    f"{base}/api/overview",
                    headers=tunnel_headers,
                )
                assert authed_ov.status_code == 200
                assert authed_ov.json()["model"] == "raven/gpt-6.2-sol#low"

                # Logout clears session cookie
                await client.post(f"{base}/api/logout", headers=tunnel_headers)
                after_logout = await client.get(f"{base}/api/overview", headers=tunnel_headers)
                assert after_logout.status_code == 401

                # HTTPS proxy sets Secure cookie flag
                https_login = await client.post(
                    f"{base}/api/login",
                    headers={**tunnel_headers, "X-Forwarded-Proto": "https"},
                    json={"password": "remote-console-password"},
                )
                assert https_login.status_code == 200
                assert "Secure" in https_login.headers.get("set-cookie", "")
        finally:
            await runner.cleanup()

    asyncio.run(run_http())


def test_web_console_reminders_crud_and_capability_toggles(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "reminders": {
                    "enabled": True,
                    "file_path": str(tmp_path / "reminders.json"),
                    "quiet_start_hhmm": "23:00",
                    "quiet_end_hhmm": "07:00",
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    reminder_changes: list[tuple[bool, str, str]] = []
    cap_changes = 0

    def on_cap() -> None:
        nonlocal cap_changes
        cap_changes += 1

    srv = McpServer(
        cfg,
        config_path=cfg_path,
        on_reminders_config_changed=lambda en, s, e: reminder_changes.append((en, s, e)),
        on_capabilities_changed=on_cap,
        include_local_tools=True,
    )

    async def run() -> None:
        runner, port = await srv.start_http_server(host="127.0.0.1", port=0, auth_token="tok")
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient() as client:
                # 1. Create countdown and daily morning briefing via POST /api/reminders
                r1 = await client.post(
                    f"{base}/api/reminders",
                    json={"message": "提醒喝水", "delay_seconds": 600, "kind": "reminder"},
                )
                assert r1.status_code == 201
                rem1 = r1.json()["reminder"]
                assert rem1["message"] == "提醒喝水"
                assert rem1["trigger_at"] is not None

                r2 = await client.post(
                    f"{base}/api/reminders",
                    json={"message": "每日早报", "daily_hhmm": "08:30", "kind": "morning_briefing"},
                )
                assert r2.status_code == 201
                rem2 = r2.json()["reminder"]
                assert rem2["daily_hhmm"] == "08:30"
                assert rem2["kind"] == "morning_briefing"

                listed = await client.get(f"{base}/api/reminders")
                assert listed.status_code == 200
                assert listed.json()["count"] == 2

                # 2. Delete one reminder
                deleted = await client.delete(f"{base}/api/reminders/{rem1['id']}")
                assert deleted.status_code == 200
                assert deleted.json()["removed"] is True
                assert (await client.get(f"{base}/api/reminders")).json()["count"] == 1

                # 3. Update quiet hours & capability toggles via POST /api/capabilities
                cap_resp = await client.post(
                    f"{base}/api/capabilities",
                    json={
                        "quiet_start_hhmm": "22:30",
                        "quiet_end_hhmm": "07:30",
                        "browser_enabled": True,
                        "xiaohongshu_enabled": True,
                    },
                )
                assert cap_resp.status_code == 200
                cap_data = cap_resp.json()
                assert cap_data["persisted"] is True
                assert cap_data["reminders_config"] == {
                    "enabled": True,
                    "quiet_start_hhmm": "22:30",
                    "quiet_end_hhmm": "07:30",
                }
                assert cap_data["toggles"]["browser"] is True
                assert cap_data["toggles"]["xiaohongshu"] is True
                assert reminder_changes == [(True, "22:30", "07:30")]
                assert cap_changes == 1

                reloaded = load_config(cfg_path)
                assert reloaded.reminders.quiet_start_hhmm == "22:30"
                assert reloaded.reminders.quiet_end_hhmm == "07:30"
                assert reloaded.tools.browser.enabled is True
                assert reloaded.tools.xiaohongshu.enabled is True

                # Invalid HH:MM rejected cleanly
                bad_q = await client.post(
                    f"{base}/api/capabilities",
                    json={"quiet_start_hhmm": "25:99"},
                )
                assert bad_q.status_code == 400

                # Boolean delay_seconds rejected cleanly
                bool_rem = await client.post(
                    f"{base}/api/reminders",
                    json={"message": "测试布尔值", "delay_seconds": True},
                )
                assert bool_rem.status_code == 400

                # Exact-ID deletion must NOT delete another reminder whose message contains the ID
                r_keep = await client.post(
                    f"{base}/api/reminders",
                    json={"message": f"包含关键字 {rem2['id']} 的提醒", "delay_seconds": 900},
                )
                assert r_keep.status_code == 201
                del_exact = await client.delete(f"{base}/api/reminders/{rem2['id']}")
                assert del_exact.status_code == 200
                remaining = (await client.get(f"{base}/api/reminders")).json()["reminders"]
                assert len(remaining) == 1
                assert remaining[0]["id"] == r_keep.json()["reminder"]["id"]

                # Nonexistent ID returns 404
                assert (await client.delete(f"{base}/api/reminders/nonexistent")).status_code == 404

                # Cross-origin and proxied unauthenticated requests rejected on new write endpoints
                assert (
                    await client.post(
                        f"{base}/api/reminders",
                        headers={"Origin": "https://evil.example"},
                        json={"message": "evil", "delay_seconds": 60},
                    )
                ).status_code == 403
                assert (
                    await client.post(
                        f"{base}/api/capabilities",
                        headers={"Host": "nomi.mxhan.com", "CF-Connecting-IP": "198.51.100.2"},
                        json={"browser_enabled": False},
                    )
                ).status_code == 401

                # Disabling reminders blocks new creation via POST /api/reminders
                dis = await client.post(
                    f"{base}/api/capabilities",
                    json={"reminders_enabled": False},
                )
                assert dis.status_code == 200
                blocked = await client.post(
                    f"{base}/api/reminders",
                    json={"message": "禁用后添加", "delay_seconds": 60},
                )
                assert blocked.status_code == 400
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_web_console_daily_trends_opencode_usage_and_wakewords_update(tmp_path: Path) -> None:
    import sqlite3

    agent_home = tmp_path / "agent-home"
    agent_home.mkdir(parents=True)
    db_path = agent_home / "opencode.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE session_message (seq INTEGER, type TEXT, time_created INTEGER, data TEXT)"
    )
    now_ms = int(time.time() * 1000)
    yesterday_ms = now_ms - 86400 * 1000
    conn.execute(
        "INSERT INTO session_message VALUES (1, 'assistant', ?, ?)",
        (
            now_ms,
            json.dumps(
                {
                    "cost": 0.0125,
                    "tokens": {"input": 1200, "output": 80, "reasoning": 20, "cache": {"read": 500}},
                }
            ),
        ),
    )
    conn.execute(
        "INSERT INTO session_message VALUES (2, 'assistant', ?, ?)",
        (
            yesterday_ms,
            json.dumps(
                {
                    "cost": 0.0050,
                    "tokens": {"input": 600, "output": 40, "reasoning": 10, "cache": {"read": 200}},
                }
            ),
        ),
    )
    conn.commit()
    conn.close()

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"llm": {"providers": {"opencode": {"agent_home": str(agent_home)}}}},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    # Verify expanded English default wakewords include HI NOMI, HI NUOMI, HI NAOMI, NI NAOMI
    for phrase in ("HI NOMI", "HI NUOMI", "HI NAOMI", "NI NAOMI"):
        assert phrase in cfg.speech.en_keywords

    wake_updates: list[tuple[list[str], list[str], bool]] = []
    hist_store = ConversationHistoryStore(tmp_path / "history.json")
    hist_store.record("你好", "在呢", latency_ms=150.0, model="raven/gemini-4-hydrogen#low")

    srv = McpServer(
        cfg,
        history_store=hist_store,
        config_path=cfg_path,
        on_wakewords_changed=lambda loc, en, en_on: wake_updates.append((loc, en, en_on)),
        include_local_tools=True,
    )

    async def run() -> None:
        runner, port = await srv.start_http_server(host="127.0.0.1", port=0, auth_token="tok")
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient() as client:
                ov = (await client.get(f"{base}/api/overview")).json()
                assert len(ov["daily_turns"]) == 14
                assert ov["daily_turns"][-1]["turns"] == 1
                usage = ov["usage"]
                assert usage["available"] is True
                assert usage["total_calls"] == 2
                assert usage["total_tokens"] == (1200 + 80 + 20 + 500) + (600 + 40 + 10 + 200)
                assert usage["today_tokens"] == 1800
                assert abs(usage["total_cost_usd"] - 0.0175) < 1e-6
                assert len(usage["daily"]) == 14

                # Update wakewords + rate via POST /api/settings
                upd = await client.post(
                    f"{base}/api/settings",
                    json={
                        "edge_rate": "+20%",
                        "local_keywords": "糯米糯米, 小糯米",
                        "en_keywords": "hi nomi, hi nuomi, ni naomi",
                        "en_kws_enabled": True,
                    },
                )
                assert upd.status_code == 200
                body = upd.json()
                assert body["edge_rate"] == "+20%"
                assert body["wakewords"] == {
                    "local": ["糯米糯米", "小糯米"],
                    "english": ["HI NOMI", "HI NUOMI", "NI NAOMI"],
                    "en_kws_enabled": True,
                }
                assert wake_updates == [
                    (["糯米糯米", "小糯米"], ["HI NOMI", "HI NUOMI", "NI NAOMI"], True)
                ]
                reloaded = load_config(cfg_path)
                assert reloaded.speech.local_keywords == ["糯米糯米", "小糯米"]
                assert reloaded.speech.en_keywords == ["HI NOMI", "HI NUOMI", "NI NAOMI"]

                # Empty local_keywords rejected cleanly
                bad = await client.post(f"{base}/api/settings", json={"local_keywords": " , "})
                assert bad.status_code == 400

                # Empty en_keywords rejected when en_kws_enabled=True, allowed when disabled
                bad_en = await client.post(
                    f"{base}/api/settings",
                    json={"en_keywords": "", "en_kws_enabled": True},
                )
                assert bad_en.status_code == 400
                ok_en_off = await client.post(
                    f"{base}/api/settings",
                    json={"en_keywords": "", "en_kws_enabled": False},
                )
                assert ok_en_off.status_code == 200
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_read_local_opencode_usage_handles_malformed_rows_and_missing_db(tmp_path: Path) -> None:
    import sqlite3
    from xiao_speaker_assistant.tools.history import read_local_opencode_usage

    # Missing DB returns available=False cleanly
    missing = read_local_opencode_usage(tmp_path / "does-not-exist")
    assert missing["available"] is False
    assert len(missing["daily"]) == 14

    # Malformed rows (bad JSON, non-dict tokens, non-numeric cost, huge timestamp) never crash
    ah = tmp_path / "weird#dir?1"
    ah.mkdir(parents=True)
    conn = sqlite3.connect(ah / "opencode.db")
    conn.execute("CREATE TABLE session_message (seq INTEGER, type TEXT, time_created INTEGER, data TEXT)")
    now_sec = int(time.time())
    conn.execute("INSERT INTO session_message VALUES (1, 'assistant', ?, ?)", (now_sec, "{not json"))
    conn.execute(
        "INSERT INTO session_message VALUES (2, 'assistant', ?, ?)",
        (now_sec, json.dumps({"cost": "N/A", "tokens": ["not", "a", "dict"]})),
    )
    conn.execute(
        "INSERT INTO session_message VALUES (3, 'assistant', ?, ?)",
        (99999999999999999, json.dumps({"cost": 0.01, "tokens": {"input": 100, "output": 20}})),
    )
    conn.execute(
        "INSERT INTO session_message VALUES (4, 'assistant', ?, ?)",
        (now_sec, json.dumps({"cost": 0.02, "tokens": {"input": 300, "output": 50}})),
    )
    conn.commit()
    conn.close()

    usage = read_local_opencode_usage(ah, days=7)
    assert usage["available"] is True
    assert usage["total_tokens"] == 470
    assert usage["today_tokens"] == 350
    assert len(usage["daily"]) == 7


def test_turn_trace_store_strict_ten_cap_permissions_and_console_api(tmp_path: Path) -> None:
    from xiao_speaker_assistant.tools import TurnTraceStore

    trace_path = tmp_path / "data" / "traces.json"
    # Even if caller passes max_entries=50, TurnTraceStore strictly caps retention at 10.
    store = TurnTraceStore(trace_path, max_entries=50)
    for i in range(15):
        store.record(
            query=f"第 {i} 个问题",
            reply=f"第 {i} 个回答",
            breakdown_ms={
                "asr_ms": 48.0 + i,
                "speaker_id_ms": 14.0,
                "llm_ttft_ms": 920.0 + i * 10,
                "llm_total_ms": 1350.0 + i * 10,
                "start_play_ms": 125.0,
                "tts_first_chunk_ms": 840.0,
                "tts_total_ms": 2100.0,
                "ttfa_ms": 1822.0 + i * 10,
                "total_ms": 3450.0 + i * 10,
                "ignored_extra_field": 999.0,
            },
            model="raven/gemini-4-hydrogen#low",
            voice="zh-CN-XiaoyiNeural",
            speaker_name="大雄",
            speaker_role="host",
            transport="wan_relay",
            sample_rate=16000,
            sentence_count=2,
            barge_in=(i == 14),
            status="ok",
        )

    assert stat.S_IMODE(trace_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(trace_path.parent.stat().st_mode) == 0o700

    reloaded = TurnTraceStore(trace_path)
    items = reloaded.list(limit=20)
    assert len(items) == 10
    assert items[0].query == "第 14 个问题"
    assert items[0].barge_in is True
    assert items[-1].query == "第 5 个问题"
    assert "ignored_extra_field" not in items[0].breakdown_ms
    assert items[0].breakdown_ms["ttfa_ms"] == 1962.0

    hist_store = ConversationHistoryStore(tmp_path / "data" / "history.json")
    now = time.time()
    hist_store.record("你好", "你好呀", latency_ms=1100.0, timestamp=now)
    hist_store.record("天气", "晴朗", latency_ms=2600.0, timestamp=now)
    daily = hist_store.daily_turns(days=14)
    assert daily[-1]["avg_latency_ms"] == 1850.0
    assert daily[-1]["p95_latency_ms"] == 2600.0

    cfg = AppConfig()
    server = McpServer(
        cfg,
        history_store=hist_store,
        trace_store=reloaded,
        memory_store=MemoryStore(tmp_path / "memory.json"),
        console_password="remote-secret-password",
    )

    async def run() -> None:
        runner, port = await server.start_http_server(host="127.0.0.1", port=0)
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient() as client:
                html_resp = await client.get(f"{base}/")
                assert html_resp.status_code == 200
                assert 'data-trend-metric="latency"' in html_resp.text
                assert "recentTracesContainer" in html_resp.text
                assert "最近 10 次对话 Trace" in html_resp.text

                ov = await client.get(f"{base}/api/overview")
                assert ov.status_code == 200
                ov_data = ov.json()
                assert len(ov_data["recent_traces"]) == 10
                assert ov_data["recent_traces"][0]["query"] == "第 14 个问题"
                assert ov_data["daily_turns"][-1]["p95_latency_ms"] == 2600.0

                # Remote forwarded request without auth must get 401 on GET/DELETE /api/traces,
                # cross-origin DELETE gets 403, and bad limit gets 400.
                unauth = await client.get(
                    f"{base}/api/traces",
                    headers={"X-Forwarded-For": "203.0.113.9"},
                )
                assert unauth.status_code == 401
                unauth_del = await client.delete(
                    f"{base}/api/traces",
                    headers={"X-Forwarded-For": "203.0.113.9"},
                )
                assert unauth_del.status_code == 401
                csrf_del = await client.delete(
                    f"{base}/api/traces",
                    headers={"Origin": "https://evil.example"},
                )
                assert csrf_del.status_code == 403
                bad_limit = await client.get(f"{base}/api/traces?limit=abc")
                assert bad_limit.status_code == 400

                tr = await client.get(f"{base}/api/traces?limit=5")
                assert tr.status_code == 200
                tr_data = tr.json()
                assert tr_data["count"] == 5
                assert tr_data["max_traces"] == 10

                del_resp = await client.delete(f"{base}/api/traces")
                assert del_resp.status_code == 200
                assert del_resp.json()["cleared"] == 10
                assert (await client.get(f"{base}/api/traces")).json()["count"] == 0
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_speaker_gateway_records_turn_trace_with_stage_breakdown(tmp_path: Path) -> None:
    from xiao_speaker_assistant.config import SpeakerConfig
    from xiao_speaker_assistant.tools import TurnTraceStore
    from xiao_speaker_assistant.voice.ws_server import SpeakerGateway

    trace_store = TurnTraceStore(tmp_path / "traces.json")

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw) -> None:
            if isinstance(raw, (bytes, bytearray)):
                return
            msg = json.loads(raw)
            req = msg.get("Request") or {}
            cmd = str(req.get("command") or "")
            if cmd in {"start_play", "stop_play"} and isinstance(req.get("id"), str):
                rid = req["id"]
                asyncio.get_running_loop().call_soon(
                    lambda r=rid: asyncio.create_task(
                        gw._handle_text_message(
                            self,
                            json.dumps({"Response": {"id": r, "code": 0}}),
                        )
                    )
                )

    async def fake_tts(text: str, cfg: SpeakerConfig, **kwargs):
        await asyncio.sleep(0.02)
        yield b"\x02\x00" * 3200

    async def fake_stream_query(query: str):
        await asyncio.sleep(0.02)
        yield "今天北京天气晴朗，微风舒适。"

    cfg = AppConfig()
    cfg.speaker.tts_provider = "edge"
    gw = SpeakerGateway(cfg, on_stream_query=fake_stream_query, trace_store=trace_store)

    async def run() -> None:
        conn = FakeConn()
        gw._relay_connections.add(conn)
        with (
            patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_tts),
            patch("xiao_speaker_assistant.voice.ws_server.synthesize_tts_pcm_stream", fake_tts),
        ):
            await gw._run_query_and_speak(
                conn,
                "今天天气怎么样",
                asr_metrics={"vad_end_mono": time.monotonic() - 0.08, "asr_ms": 52.3, "speaker_id_ms": 14.1},
            )

    asyncio.run(run())
    traces = trace_store.list()
    assert len(traces) == 1
    tr = traces[0]
    assert tr.query == "今天天气怎么样"
    assert tr.reply == "今天北京天气晴朗，微风舒适。"
    assert tr.transport == "wan_relay"
    assert tr.sample_rate == 16000
    assert tr.sentence_count == 1
    assert tr.status == "ok"
    assert tr.breakdown_ms["asr_ms"] == 52.3
    assert tr.breakdown_ms["speaker_id_ms"] == 14.1
    assert tr.breakdown_ms["llm_ttft_ms"] >= 15.0
    assert tr.breakdown_ms["start_play_ms"] >= 0.0
    assert tr.breakdown_ms["tts_first_chunk_ms"] >= 15.0
    assert tr.breakdown_ms["ttfa_ms"] >= 80.0
    assert tr.breakdown_ms["total_ms"] >= tr.breakdown_ms["ttfa_ms"]

    # Error during streaming must record status="error" without polluting TTS/TTFA metrics from error notice.
    async def failing_stream_query(query: str):
        if False:
            yield ""
        raise RuntimeError("upstream timeout")

    gw_err = SpeakerGateway(cfg, on_stream_query=failing_stream_query, trace_store=trace_store)

    async def run_err() -> None:
        conn = FakeConn()
        gw_err._relay_connections.add(conn)
        with (
            patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_tts),
            patch("xiao_speaker_assistant.voice.ws_server.synthesize_tts_pcm_stream", fake_tts),
        ):
            await gw_err._run_query_and_speak(conn, "出错测试")

    asyncio.run(run_err())
    err_tr = trace_store.list()[0]
    assert err_tr.query == "出错测试"
    assert err_tr.status == "error"
    assert "tts_first_chunk_ms" not in err_tr.breakdown_ms
    assert "start_play_ms" not in err_tr.breakdown_ms
    assert "ttfa_ms" not in err_tr.breakdown_ms


def test_p0_oss_readiness_init_doctor_openai_tools_and_speaker_status(tmp_path: Path, monkeypatch) -> None:
    from xiao_speaker_assistant.llm.router import LLMRouter
    from xiao_speaker_assistant.tools import TurnTraceStore
    from xiao_speaker_assistant.voice.speaker_id import default_style_hint_for_role

    # 1. Generic host role hint does not hardcode personal nicknames
    host_hint = default_style_hint_for_role("Alice", "host")
    assert "主人" in host_hint
    assert "大雄" not in host_hint

    # 2. xsa init generates config.yaml without overwriting unless --force
    init_cfg = tmp_path / "config.yaml"
    assert cli.main(["--config", str(init_cfg), "init"]) == 0
    assert init_cfg.is_file()
    loaded = load_config(init_cfg)
    assert loaded.llm.default_provider == "openai_compatible"
    assert "HI NOMI" in loaded.speech.en_keywords
    assert loaded.speech.vad_min_silence_ms == 450

    # 3. xsa doctor runs cleanly for openai_compatible without requiring opencode CLI
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert cli.main(["--config", str(init_cfg), "doctor"]) == 0

    # 4. In-process MCP tool bridge exposes full tool catalog to openai_compatible / gemini_api
    #    and overview exposes real speaker_status + custom available_models + provider-aware model update
    cfg = AppConfig()
    cfg.llm.default_provider = "openai_compatible"
    cfg.llm.providers.codex_api.model = "gpt-4o-mini"
    cfg.llm.available_models = [
        {"id": "gpt-4o-mini", "label": "GPT-4o Mini"},
        {"id": "deepseek-chat", "label": "DeepSeek Chat"},
    ]
    srv = McpServer(
        cfg,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        history_store=ConversationHistoryStore(tmp_path / "hist.json"),
        trace_store=TurnTraceStore(tmp_path / "traces.json"),
        config_path=init_cfg,
        get_speaker_status=lambda: {
            "speaker_connected": True,
            "active_connections": 1,
            "relay_connections": 1,
            "lan_connections": 0,
            "pipeline_awake": False,
        },
        include_local_tools=True,
    )

    async def run() -> None:
        tools_list = await srv.get_tool_definitions_for_llm()
        tool_names = {t["name"] for t in tools_list}
        assert {"weather_query", "memory_save", "reminder_set", "speaker_list_voices"}.issubset(tool_names)

        # In-process execution works identically to /mcp
        res_text = await srv.execute_tool_in_process("speaker_list_voices", {})
        assert "zh-CN-XiaoyiNeural" in res_text

        runner, port = await srv.start_http_server(host="127.0.0.1", port=0)
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient() as client:
                ov = (await client.get(f"{base}/api/overview")).json()
                assert ov["speaker_status"]["speaker_connected"] is True
                assert ov["speaker_status"]["relay_connections"] == 1
                assert ov["model"] == "gpt-4o-mini"
                assert [m["id"] for m in ov["available_models"]] == ["gpt-4o-mini", "deepseek-chat"]

                upd = await client.post(f"{base}/api/settings", json={"model": "deepseek-chat"})
                assert upd.status_code == 200
                assert cfg.llm.providers.codex_api.model == "deepseek-chat"
                assert load_config(init_cfg).llm.providers.codex_api.model == "deepseek-chat"
        finally:
            await runner.cleanup()

    asyncio.run(run())

    # 5. LLMRouter logger is defined and never raises NameError on startup prewarm failure
    import xiao_speaker_assistant.llm.router as router_mod

    assert hasattr(router_mod, "logger")
    assert isinstance(LLMRouter(cfg.llm)._build_system_prompt(), str)

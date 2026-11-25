import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import httpx

from xiao_speaker_assistant import cli
from xiao_speaker_assistant.config import (
    AppConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpencodeCliConfig,
    OutlookToolConfig,
    ToolsConfig,
)
from xiao_speaker_assistant.llm.opencode_server import (
    OpenCodeServerManager,
    build_opencode_v2_config,
    ensure_agent_home,
    load_or_create_server_password,
    parse_model_ref,
)
from xiao_speaker_assistant.llm.router import LLMRouter
from xiao_speaker_assistant.tools import (
    ConfirmationStore,
    HttpMcpClient,
    McpServer,
    PendingEmailDraft,
)


def test_parse_model_ref_and_ensure_agent_home(tmp_path: Path) -> None:
    assert parse_model_ref("raven/gemini-4-hydrogen") == {
        "providerID": "raven",
        "id": "gemini-4-hydrogen",
    }
    assert parse_model_ref("raven/gemini-4-carbon#high") == {
        "providerID": "raven",
        "id": "gemini-4-carbon",
        "variant": "high",
    }

    cfg = OpencodeCliConfig(agent_home=str(tmp_path / "agent-home"), model="raven/gemini-4-hydrogen")
    home = ensure_agent_home(
        cfg,
        mcp_url="http://127.0.0.1:4397/mcp",
        system_prompt="你是语音助手 Nomi",
        mcp_auth_token="secret-mcp-tok",
    )
    assert home.is_dir()
    written = json.loads((home / "opencode.jsonc").read_text(encoding="utf-8"))
    assert written["default_agent"] == "nomi"
    assert written["snapshots"] is False
    assert written["mcp"]["servers"]["xsa_tools"]["url"] == "http://127.0.0.1:4397/mcp"
    assert written["mcp"]["servers"]["xsa_tools"]["headers"] == {
        "Authorization": "Bearer secret-mcp-tok"
    }
    assert written["mcp"]["servers"]["xsa_tools"]["codemode"] is False
    assert written["mcp"]["servers"]["xsa_tools"]["oauth"] is False
    perms = {(p["action"], p["effect"]) for p in written["agents"]["nomi"]["permissions"]}
    assert ("xsa_tools_*", "allow") in perms
    assert ("websearch", "deny") in perms
    assert ("shell", "deny") in perms
    assert ("edit", "deny") in perms
    assert ("write", "deny") in perms

    pw1 = load_or_create_server_password(home, "XSA_TEST_UNSET_PW")
    pw2 = load_or_create_server_password(home, "XSA_TEST_UNSET_PW")
    assert pw1 == pw2 and len(pw1) >= 16
    assert ((home / "server.token").stat().st_mode & 0o777) == 0o600


def test_llm_router_opencode_v2_streams_sse_and_recovers_stale_session(tmp_path: Path) -> None:
    created_sessions: list[dict] = []
    prompts_sent: list[tuple[str, str]] = []
    interrupted_sids: list[str] = []
    session_counter = 0
    prompt_attempt = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal session_counter, prompt_attempt
        path = request.url.path

        if request.method == "POST" and path == "/api/session":
            session_counter += 1
            created_sessions.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"data": {"id": f"ses_{session_counter}"}},
                request=request,
            )

        if request.method == "POST" and path.endswith("/prompt"):
            sid = path.split("/")[3]
            body = json.loads(request.content)
            prompts_sent.append((sid, body["text"]))
            prompt_attempt += 1
            if prompt_attempt == 2 and sid == "ses_1":
                return httpx.Response(404, json={"error": "not found"}, request=request)
            return httpx.Response(200, json={"data": {"id": "msg_1"}}, request=request)

        if request.method == "POST" and path.endswith("/interrupt"):
            sid = path.split("/")[3]
            interrupted_sids.append(sid)
            return httpx.Response(200, json={"interrupted": True}, request=request)

        if request.method == "GET" and path == "/api/event":
            # Determine which session will be prompted next
            target_sid = f"ses_{session_counter}"
            events = [
                {"type": "server.connected", "data": {}},
                {
                    "type": "session.text.delta",
                    "data": {
                        "sessionID": target_sid,
                        "assistantMessageID": "msg_1",
                        "ordinal": 0,
                        "delta": "今天北京晴，",
                    },
                },
                {
                    "type": "session.text.delta",
                    "data": {
                        "sessionID": target_sid,
                        "assistantMessageID": "msg_1",
                        "ordinal": 1,
                        "delta": "气温 15 度。",
                    },
                },
                {
                    "type": "session.execution.succeeded",
                    "data": {"sessionID": target_sid},
                },
            ]
            sse_text = "".join(f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events)
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=sse_text,
                request=request,
            )

        return httpx.Response(404, request=request)

    cfg = LLMConfig(
        providers=LLMProvidersConfig(
            opencode=OpencodeCliConfig(
                server_url="http://127.0.0.1:14396",
                agent_home=str(tmp_path / "home"),
            )
        )
    )

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:14396",
            transport=httpx.MockTransport(handler),
        ) as client:
            router = LLMRouter(cfg, client=client)
            await router.start()

            sentences = [s async for s in router.stream_ask("今天天气怎么样")]
            assert "".join(sentences) == "今天北京晴，气温 15 度。"
            assert router.opencode_session_id == "ses_1"
            assert created_sessions[0]["agent"] == "nomi"
            assert created_sessions[0]["model"] == {
                "providerID": "raven",
                "id": "gemini-4-hydrogen",
                "variant": "low",
            }

            # Second turn triggers simulated 404 on ses_1 and transparently recreates ses_2
            sentences2 = [s async for s in router.stream_ask("明天呢")]
            assert "".join(sentences2) == "今天北京晴，气温 15 度。"
            assert router.opencode_session_id == "ses_2"

            await router.interrupt_active()
            assert interrupted_sids == ["ses_2"]

    asyncio.run(run())


def test_http_mcp_server_shares_confirmation_store_and_exposes_codex_and_disk(tmp_path: Path) -> None:
    class FakeOutlook:
        async def list_recent_emails(self, limit: int = 5, unread_only: bool = False):
            return []

        async def send_email(self, draft: PendingEmailDraft) -> str:
            if not draft.confirmed:
                return json.dumps(
                    {"status": "confirmation_required", "message": f"请确认发送给 {draft.to}"},
                    ensure_ascii=False,
                )
            return json.dumps({"status": "sent", "to": draft.to}, ensure_ascii=False)

    cfg = AppConfig(
        tools=ToolsConfig(
            outlook=OutlookToolConfig(
                enabled=True,
                token_cache_file=str(tmp_path / "cache.json"),
                require_voice_confirmation_for_send=True,
            )
        )
    )
    store = ConfirmationStore()
    server = McpServer(
        cfg,
        outlook_client=FakeOutlook(),  # type: ignore[arg-type]
        confirmation_store=store,
        include_local_tools=True,
    )

    class FakeCodexProc:
        def __init__(self, out_path: str) -> None:
            self._out_path = out_path
            self.returncode = 0

        async def communicate(self):
            Path(self._out_path).write_text("代码结构清晰，未发现严重问题。", encoding="utf-8")
            return b"", b""

    codex_cmds: list[list[str]] = []

    async def fake_codex_exec(*cmd, **kwargs):
        codex_cmds.append(list(cmd))
        out_idx = list(cmd).index("-o") + 1
        return FakeCodexProc(cmd[out_idx])

    async def run() -> None:
        runner, port = await server.start_http_server(
            host="127.0.0.1", port=0, auth_token="mcp-bearer-secret"
        )
        try:
            # Unauthenticated and cross-origin requests must be rejected
            async with httpx.AsyncClient() as raw_http:
                unauth = await raw_http.post(
                    f"http://127.0.0.1:{port}/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                )
                assert unauth.status_code == 401
                csrf = await raw_http.post(
                    f"http://127.0.0.1:{port}/mcp",
                    headers={"Origin": "https://evil.example", "Authorization": "Bearer mcp-bearer-secret"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                )
                assert csrf.status_code == 403

            client = HttpMcpClient(
                f"http://127.0.0.1:{port}/mcp",
                auth_token="mcp-bearer-secret",
            )
            tools = await client.list_tools()
            names = {t["name"] for t in tools}
            assert "outlook_prepare_email" in names
            assert "outlook_send_email" not in names
            assert "codex_analyze" in names
            assert "mac_disk_usage" in names

            # Check outlook_prepare_email schema has no `confirmed` arg
            prep_schema = next(t["inputSchema"] for t in tools if t["name"] == "outlook_prepare_email")
            assert "confirmed" not in prep_schema["properties"]

            # Call outlook_prepare_email over HTTP MCP and verify shared ConfirmationStore is populated
            prep_res = await client.call_tool(
                "outlook_prepare_email",
                {"to": "alice@example.com", "subject": "Hi", "body": "Hello"},
            )
            prep_data = json.loads(prep_res["content"][0]["text"])
            assert prep_data["status"] == "confirmation_required"
            assert store.peek() == PendingEmailDraft(
                to="alice@example.com", subject="Hi", body="Hello", confirmed=False
            )

            # Call mac_disk_usage
            disk_res = await client.call_tool("mac_disk_usage", {"path": str(tmp_path)})
            disk_data = json.loads(disk_res["content"][0]["text"])
            assert disk_data["total_gb"] > 0

            # Call codex_analyze
            with (
                patch("xiao_speaker_assistant.tools.mcp_server.resolve_binary", return_value="/usr/local/bin/codex"),
                patch("asyncio.create_subprocess_exec", side_effect=fake_codex_exec),
            ):
                codex_res = await client.call_tool(
                    "codex_analyze",
                    {"prompt": "检查一下仓库", "workdir": str(tmp_path)},
                )
            codex_data = json.loads(codex_res["content"][0]["text"])
            assert codex_data["status"] == "ok"
            assert "代码结构清晰" in codex_data["analysis"]
            assert "-s" in codex_cmds[0] and "read-only" in codex_cmds[0]
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_doctor_reports_opencode_v2_and_mcp_status(capsys) -> None:
    rc = cli.cmd_doctor(AppConfig())
    assert rc == 0
    out = capsys.readouterr().out
    assert "OpenCode V2 URL:" in out
    assert "Unified MCP HTTP:" in out
    assert "ffmpeg" in out


def test_mcp_weather_query_and_web_search_with_mock_http() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "wttr.in" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "current_condition": [
                        {
                            "temp_C": "8",
                            "FeelsLikeC": "6",
                            "humidity": "30",
                            "windspeedKmph": "12",
                            "lang_zh": [{"value": "晴"}],
                        }
                    ],
                    "weather": [
                        {"date": "2026-11-25", "mintempC": "-1", "maxtempC": "10", "hourly": [{"lang_zh": [{"value": "晴"}]}]},
                        {"date": "2026-11-26", "mintempC": "0", "maxtempC": "11", "hourly": [{"lang_zh": [{"value": "多云"}]}]},
                        {"date": "2026-11-27", "mintempC": "-2", "maxtempC": "6", "hourly": [{"lang_zh": [{"value": "晴"}]}]},
                    ],
                },
                request=request,
            )
        if "geocoding-api.open-meteo.com" in str(request.url):
            return httpx.Response(200, json={"results": [{"latitude": 40.46, "longitude": 115.97}]}, request=request)
        if "api.open-meteo.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "daily": {
                        "time": ["2026-11-25", "2026-11-26", "2026-11-27", "2026-11-28"],
                        "temperature_2m_min": [-2.0, 1.0, -3.0, -4.0],
                        "temperature_2m_max": [8.0, 10.0, 5.0, 6.0],
                    }
                },
                request=request,
            )
        if "www.bing.com/search" in str(request.url):
            html_body = (
                '<ul><li class="b_algo"><ul><li>nested</li></ul><h2><a href="https://example.com/news">北京今日天气</a></h2>'
                '<p>晴朗微风，最高气温10度。</p></li></ul>'
            )
            return httpx.Response(200, text=html_body, request=request)
        return httpx.Response(404, request=request)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            srv = McpServer(AppConfig(), http_client=http_client, include_local_tools=True)
            w_resp = await srv.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "weather_query", "arguments": {"city": "延庆"}},
                }
            )
            assert w_resp is not None and w_resp["result"]["isError"] is False
            w_data = json.loads(w_resp["result"]["content"][0]["text"])
            assert w_data["current_temp_c"] == "8"
            assert w_data["condition"] == "晴"
            assert len(w_data["forecast"]) == 4
            assert w_data["forecast"][2]["date"] == "2026-11-27"
            assert w_data["forecast"][2]["condition"] == "晴"

            s_resp = await srv.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "web_search", "arguments": {"query": "延庆天气"}},
                }
            )
            assert s_resp is not None and s_resp["result"]["isError"] is False
            s_data = json.loads(s_resp["result"]["content"][0]["text"])
            assert s_data["count"] == 1
            assert s_data["results"][0]["title"] == "北京今日天气"

    asyncio.run(run())

import asyncio
import io
import json
from typing import Any

import httpx

from xiao_speaker_assistant import cli
from xiao_speaker_assistant.config import AppConfig, McpEndpointConfig, OutlookToolConfig, ToolsConfig
from xiao_speaker_assistant.tools.confirmation import (
    ConfirmationStore,
    is_cancel_utterance,
    is_confirm_utterance,
)
from xiao_speaker_assistant.tools.mcp_client import parse_mcp_http_response
from xiao_speaker_assistant.tools.mcp_server import McpServer, extract_html_summary
from xiao_speaker_assistant.tools.outlook import OutlookClient, PendingEmailDraft


class FakeOutlookClient:
    def __init__(self, require_confirmation: bool = True) -> None:
        self.require_confirmation = require_confirmation
        self.list_calls: list[tuple[int, bool]] = []
        self.sent_drafts: list[PendingEmailDraft] = []

    async def list_recent_emails(
        self, limit: int = 5, unread_only: bool = False
    ) -> list[dict[str, Any]]:
        self.list_calls.append((limit, unread_only))
        return [
            {
                "subject": "周报提醒",
                "from": {"emailAddress": {"address": "boss@example.com"}},
                "bodyPreview": "请于周五前提交周报。",
                "isRead": False,
            }
        ]

    async def send_email(self, draft: PendingEmailDraft) -> str:
        if self.require_confirmation and not draft.confirmed:
            return json.dumps(
                {
                    "status": "confirmation_required",
                    "message": f"即将发送邮件给 {draft.to}，主题是 {draft.subject}。请先语音确认后再发送。",
                },
                ensure_ascii=False,
            )
        self.sent_drafts.append(draft)
        return json.dumps(
            {"status": "sent", "to": draft.to, "subject": draft.subject},
            ensure_ascii=False,
        )


class FakeHttpMcpClient:
    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses = responses or {}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        return self.responses.get(name, {"ok": True, "tool": name, "arguments": arguments})


def test_confirmation_store_lifecycle_and_ttl() -> None:
    now = 100.0

    def clock() -> float:
        return now

    store = ConfirmationStore(ttl_sec=30.0, clock=clock)
    draft = PendingEmailDraft(to="alice@example.com", subject="问候", body="晚上好")

    stored = store.set_pending(draft)
    assert stored.confirmed is False
    assert store.peek() == PendingEmailDraft(
        to="alice@example.com", subject="问候", body="晚上好", confirmed=False
    )

    now = 129.0
    popped = store.confirm_and_pop()
    assert popped == PendingEmailDraft(
        to="alice@example.com", subject="问候", body="晚上好", confirmed=True
    )
    assert store.peek() is None
    assert store.confirm_and_pop() is None

    now = 200.0
    store.set_pending(draft)
    now = 229.9
    assert store.peek() is not None
    now = 230.0
    assert store.peek() is None
    assert store.confirm_and_pop() is None

    store.set_pending(draft)
    assert store.peek() is not None
    store.clear()
    assert store.peek() is None


def test_confirm_and_cancel_utterance_helpers() -> None:
    for text in ("确认", "确认发送。", "好的，发吧！", "Yes, send it!", "OK", "confirm"):
        assert is_confirm_utterance(text) is True
        assert is_cancel_utterance(text) is False

    for text in ("取消", "取消发送", "别发了", "不要发送", "算了，不用发", "不确认", "No", "cancel"):
        assert is_cancel_utterance(text) is True
        assert is_confirm_utterance(text) is False

    for text in ("", "   ", "今天天气怎么样", "帮我查一下小红书"):
        assert is_confirm_utterance(text) is False
        assert is_cancel_utterance(text) is False


def test_mcp_server_initialize_and_tools_list() -> None:
    server = McpServer(AppConfig())

    init_resp = asyncio.run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
        )
    )
    assert init_resp is not None
    assert init_resp["id"] == 1
    assert init_resp["result"]["protocolVersion"] == "2024-11-05"
    assert "tools" in init_resp["result"]["capabilities"]

    notif_resp = asyncio.run(
        server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    )
    assert notif_resp is None

    list_resp = asyncio.run(
        server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    )
    assert list_resp is not None
    names = {t["name"] for t in list_resp["result"]["tools"]}
    assert names == {
        "outlook_list_emails",
        "outlook_prepare_email",
        "xiaohongshu_search",
        "xiaohongshu_detail",
        "browser_fetch_summary",
    }

    bad_method = asyncio.run(
        server.handle_message({"jsonrpc": "2.0", "id": 3, "method": "unknown/method"})
    )
    assert bad_method is not None
    assert bad_method["error"]["code"] == -32601

    bad_tool = asyncio.run(
        server.handle_message(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope"}}
        )
    )
    assert bad_tool is not None
    assert bad_tool["error"]["code"] == -32602


def test_mcp_server_outlook_list_and_send_confirmation_guard(tmp_path) -> None:
    cfg = AppConfig(
        tools=ToolsConfig(
            outlook=OutlookToolConfig(
                enabled=True,
                token_cache_file=str(tmp_path / "cache.json"),
                require_voice_confirmation_for_send=True,
            )
        )
    )
    fake_outlook = FakeOutlookClient(require_confirmation=True)
    server = McpServer(cfg, outlook_client=fake_outlook)  # type: ignore[arg-type]

    list_resp = asyncio.run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "outlook_list_emails",
                    "arguments": {"limit": 3, "unread_only": True},
                },
            }
        )
    )
    assert list_resp is not None
    assert list_resp["result"]["isError"] is False
    list_data = json.loads(list_resp["result"]["content"][0]["text"])
    assert list_data["count"] == 1
    assert list_data["emails"][0]["subject"] == "周报提醒"
    assert fake_outlook.list_calls == [(3, True)]

    # Calling outlook_prepare_email queues in ConfirmationStore even if confirmed=True is smuggled.
    real_outlook = OutlookClient(cfg.tools.outlook)
    guarded_server = McpServer(cfg, outlook_client=real_outlook)
    unconfirmed_resp = asyncio.run(
        guarded_server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "outlook_prepare_email",
                    "arguments": {
                        "to": "bob@example.com",
                        "subject": "会议",
                        "body": "三点开会",
                        "confirmed": True,
                    },
                },
            }
        )
    )
    assert unconfirmed_resp is not None
    assert unconfirmed_resp["result"]["isError"] is False
    payload = json.loads(unconfirmed_resp["result"]["content"][0]["text"])
    assert payload["status"] == "confirmation_required"
    pending = guarded_server.confirmation_store.peek()
    assert pending == PendingEmailDraft(
        to="bob@example.com", subject="会议", body="三点开会", confirmed=False
    )

    # Only after explicit confirmation pop + send_email does it send.
    popped = guarded_server.confirmation_store.confirm_and_pop()
    assert popped is not None and popped.confirmed is True
    sent_payload = json.loads(asyncio.run(fake_outlook.send_email(popped)))
    assert sent_payload["status"] == "sent"
    assert len(fake_outlook.sent_drafts) == 1
    assert guarded_server.confirmation_store.peek() is None


def test_mcp_server_xiaohongshu_disabled_and_enabled() -> None:
    disabled_server = McpServer(AppConfig())
    for tool_name, args in (
        ("xiaohongshu_search", {"keyword": "咖啡"}),
        ("xiaohongshu_detail", {"feed_id": "feed_1", "xsec_token": "tok_1"}),
    ):
        resp = asyncio.run(
            disabled_server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 20,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": args},
                }
            )
        )
        assert resp is not None
        assert resp["result"]["isError"] is True
        err = json.loads(resp["result"]["content"][0]["text"])
        assert err["error"] == "xiaohongshu_mcp_not_enabled"
        assert "18060" in err["mcp_url"]

    enabled_cfg = AppConfig(
        tools=ToolsConfig(
            xiaohongshu=McpEndpointConfig(enabled=True, mcp_url="http://127.0.0.1:18060/mcp")
        )
    )
    fake_xhs = FakeHttpMcpClient(
        {
            "search_feeds": {"notes": [{"id": "n1", "title": "手冲咖啡"}]},
            "get_feed_detail": {"id": "n1", "content": "研磨度建议中细"},
        }
    )
    enabled_server = McpServer(enabled_cfg, xiaohongshu_client=fake_xhs)  # type: ignore[arg-type]

    search_resp = asyncio.run(
        enabled_server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "tools/call",
                "params": {"name": "xiaohongshu_search", "arguments": {"keyword": "咖啡"}},
            }
        )
    )
    assert search_resp is not None
    assert search_resp["result"]["isError"] is False
    assert json.loads(search_resp["result"]["content"][0]["text"])["notes"][0]["id"] == "n1"

    detail_resp = asyncio.run(
        enabled_server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 22,
                "method": "tools/call",
                "params": {
                    "name": "xiaohongshu_detail",
                    "arguments": {"feed_id": "n1", "xsec_token": "abc"},
                },
            }
        )
    )
    assert detail_resp is not None
    assert json.loads(detail_resp["result"]["content"][0]["text"])["content"] == "研磨度建议中细"
    assert fake_xhs.calls == [
        ("search_feeds", {"keyword": "咖啡"}),
        ("get_feed_detail", {"feed_id": "n1", "xsec_token": "abc"}),
    ]

    login_resp = asyncio.run(
        enabled_server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 23,
                "method": "tools/call",
                "params": {"name": "xiaohongshu_check_login", "arguments": {}},
            }
        )
    )
    assert login_resp is not None and login_resp["result"]["isError"] is False
    assert fake_xhs.calls[-1] == ("check_login_status", {})


def test_browser_fetch_summary_html_stripping_and_mcp_delegation() -> None:
    sample_html = """
    <html>
      <head>
        <title>测试页面 &amp; 指南</title>
        <style>body { color: red; }</style>
        <script>console.log("hidden script");</script>
      </head>
      <body>
        <!-- secret comment -->
        <h1>欢迎来到 Nomi</h1>
        <p>第一段内容 &lt;重点&gt;。</p>
        <noscript>no js</noscript>
        <div>第二段   内容。</div>
      </body>
    </html>
    """
    title, excerpt = extract_html_summary(sample_html, max_chars=200)
    assert title == "测试页面 & 指南"
    assert "console.log" not in excerpt
    assert "color: red" not in excerpt
    assert "secret comment" not in excerpt
    assert "no js" not in excerpt
    assert "欢迎来到 Nomi 第一段内容 <重点>。 第二段 内容。" == excerpt

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text=sample_html,
            request=request,
        )

    async def run_http_fallback() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            server = McpServer(AppConfig(), http_client=client)
            resp = await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 30,
                    "method": "tools/call",
                    "params": {
                        "name": "browser_fetch_summary",
                        "arguments": {"url": "https://example.com/guide", "max_chars": 120},
                    },
                }
            )
            assert resp is not None
            return resp

    resp = asyncio.run(run_http_fallback())
    assert resp["result"]["isError"] is False
    data = json.loads(resp["result"]["content"][0]["text"])
    assert data["source"] == "http_fetch"
    assert data["title"] == "测试页面 & 指南"
    assert "欢迎来到 Nomi" in data["excerpt"]

    # Reject non-HTTP schemes
    bad_url_resp = asyncio.run(
        McpServer(AppConfig()).handle_message(
            {
                "jsonrpc": "2.0",
                "id": 31,
                "method": "tools/call",
                "params": {
                    "name": "browser_fetch_summary",
                    "arguments": {"url": "file:///etc/passwd"},
                },
            }
        )
    )
    assert bad_url_resp is not None
    assert bad_url_resp["result"]["isError"] is True

    # Delegate when tools.browser.enabled is true
    browser_cfg = AppConfig(
        tools=ToolsConfig(
            browser=McpEndpointConfig(enabled=True, mcp_url="http://127.0.0.1:8931/mcp")
        )
    )
    fake_browser = FakeHttpMcpClient({"browser_navigate": {"snapshot": "Page Title"}})
    browser_server = McpServer(browser_cfg, browser_client=fake_browser)  # type: ignore[arg-type]
    mcp_resp = asyncio.run(
        browser_server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 32,
                "method": "tools/call",
                "params": {
                    "name": "browser_fetch_summary",
                    "arguments": {"url": "https://example.com"},
                },
            }
        )
    )
    assert mcp_resp is not None
    mcp_data = json.loads(mcp_resp["result"]["content"][0]["text"])
    assert mcp_data["source"] == "browser_mcp"
    assert mcp_data["result"]["snapshot"] == "Page Title"
    assert fake_browser.calls == [("browser_navigate", {"url": "https://example.com"})]

    snap_resp = asyncio.run(
        browser_server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 33,
                "method": "tools/call",
                "params": {"name": "browser_snapshot", "arguments": {}},
            }
        )
    )
    assert snap_resp is not None and snap_resp["result"]["isError"] is False
    assert fake_browser.calls[-1] == ("browser_snapshot", {})


def test_stdio_loop_and_cli_mcp_server_subcommand(monkeypatch) -> None:
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                "{bad json",
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()
    server = McpServer(AppConfig())
    asyncio.run(server.serve_stdio(stdin=stdin, stdout=stdout))

    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert [line["id"] for line in lines] == [1, 2, None]
    assert lines[0]["result"]["serverInfo"]["name"] == "xiao-speaker-assistant"
    assert len(lines[1]["result"]["tools"]) == 5
    assert lines[2]["error"]["code"] == -32700

    called = False

    async def fake_serve_stdio(self, stdin=None, stdout=None) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(McpServer, "serve_stdio", fake_serve_stdio)
    assert cli.main(["mcp-server"]) == 0
    assert called is True
    assert cli.main(["doctor"]) == 0


def test_parse_mcp_http_response_json_and_sse() -> None:
    json_obj = parse_mcp_http_response(
        "application/json",
        '{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}',
    )
    assert json_obj["id"] == 1

    sse_body = (
        "event: message\n"
        'data: {"result":{"protocolVersion":"2024-11-05","serverInfo":{"name":"Playwright"}},"jsonrpc":"2.0","id":1}\n\n'
    )
    sse_obj = parse_mcp_http_response("text/event-stream; charset=utf-8", sse_body)
    assert sse_obj["result"]["serverInfo"]["name"] == "Playwright"

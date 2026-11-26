import asyncio
import json
import os
import subprocess
from pathlib import Path

import httpx
import pytest

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.tools import (
    ALL_TOOL_DEFINITIONS,
    MacLocalAccessError,
    McpServer,
    build_edge_open_command,
    fetch_webpage_text,
    list_recent_files,
    open_in_edge,
    read_local_text_file,
    validate_http_url,
)


def test_validate_http_url_accepts_safe_and_rejects_unsafe() -> None:
    assert validate_http_url("https://example.com") == "https://example.com"
    assert validate_http_url("  http://localhost:8080/health  ") == "http://localhost:8080/health"
    assert (
        validate_http_url("https://example.com/search?q=coffee&lang=zh")
        == "https://example.com/search?q=coffee&lang=zh"
    )

    bad_urls = [
        "",
        "   ",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "ftp://example.com/file.txt",
        "https://example.com;rm -rf /",
        "https://example.com;id",
        "https://example.com|cat",
        "https://example.com&&whoami",
        "https://example.com||whoami",
        "https://example.com/$(id)",
        "https://example.com/`whoami`",
        "https://example.com\nopen",
        "https://example.com/path with spaces",
        "https://",
    ]
    for bad in bad_urls:
        with pytest.raises(ValueError):
            validate_http_url(bad)

    # SSRF protection when block_private_network=True (including CGNAT and alternate IPv4 notations)
    for ssrf_url in (
        "http://127.0.0.1:4397/api/overview",
        "http://localhost:4396/api/info",
        "http://[::1]:4397/",
        "http://[::ffff:127.0.0.1]:4397/",
        "http://[::ffff:100.64.0.1]/",
        "http://100.64.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://0.0.0.0:4397/",
        "http://router.local/",
        "http://2130706433:4397/",
        "http://0x7f000001:4397/",
        "http://127.1:4397/",
        "http://0177.0.0.1:4397/",
    ):
        with pytest.raises(ValueError):
            validate_http_url(ssrf_url, block_private_network=True)
    assert validate_http_url("https://example.com/ok", block_private_network=True) == "https://example.com/ok"


def test_ssrf_dns_fail_closed_and_redirect_validation(monkeypatch) -> None:
    import socket
    from xiao_speaker_assistant.tools.browser import async_fetch_webpage_summary

    def fake_getaddrinfo_fail(*_args, **_kwargs):
        raise socket.gaierror("DNS failed")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo_fail)
    with pytest.raises(ValueError, match="Failed to resolve"):
        validate_http_url("https://unresolvable.example/", block_private_network=True, resolve_dns=True)

    # Redirect to loopback must be blocked even if injected httpx.AsyncClient has follow_redirects=True
    def redirect_to_loopback(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example":
            return httpx.Response(
                302,
                headers={"Location": "http://127.0.0.1:4397/api/overview"},
                request=request,
            )
        return httpx.Response(200, text="LEAKED", request=request)

    async def run_redirect_ssrf() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(redirect_to_loopback),
            follow_redirects=True,
        ) as client:
            with pytest.raises(ValueError, match="blocked"):
                await async_fetch_webpage_summary(
                    "https://public.example/start",
                    client=client,
                    block_private_network=True,
                )

    asyncio.run(run_redirect_ssrf())

    # Redirect loop (>5 hops) raises explicit ValueError
    def endless_redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "https://public.example/next"},
            request=request,
        )

    async def run_too_many_redirects() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(endless_redirect)) as client:
            with pytest.raises(ValueError, match="Too many redirects"):
                await async_fetch_webpage_summary(
                    "https://public.example/loop",
                    client=client,
                    block_private_network=True,
                )

    asyncio.run(run_too_many_redirects())


def test_open_in_edge_command_building_dry_run_and_mocked_subprocess(monkeypatch) -> None:
    cmd = build_edge_open_command("https://example.com/docs?p=1")
    assert cmd == ["open", "-a", "Microsoft Edge", "https://example.com/docs?p=1"]

    called_cmds: list[tuple[list[str], bool]] = []

    def fake_run(argv, **kwargs):
        called_cmds.append((list(argv), bool(kwargs.get("shell", False))))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    dry_json = json.loads(open_in_edge("https://example.com/preview", dry_run=True))
    assert dry_json["status"] == "dry_run"
    assert dry_json["browser"] == "Microsoft Edge"
    assert dry_json["url"] == "https://example.com/preview"
    assert dry_json["command"] == ["open", "-a", "Microsoft Edge", "https://example.com/preview"]
    assert called_cmds == []

    opened_json = json.loads(open_in_edge("https://example.com/live"))
    assert opened_json["status"] == "opened"
    assert opened_json["browser"] == "Microsoft Edge"
    assert opened_json["command"] == ["open", "-a", "Microsoft Edge", "https://example.com/live"]
    assert called_cmds == [(["open", "-a", "Microsoft Edge", "https://example.com/live"], False)]

    with pytest.raises(ValueError):
        open_in_edge("javascript:alert(1)")
    assert len(called_cmds) == 1


def test_fetch_webpage_text_validates_and_strips_html() -> None:
    sample_html = """
    <html>
      <head>
        <title>Edge Guide</title>
        <style>.hidden { display: none; }</style>
        <script>alert("x")</script>
      </head>
      <body>
        <!-- internal comment -->
        <h1>Hello Nomi</h1>
        <p>First paragraph &amp; details.</p>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text=sample_html,
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        text = fetch_webpage_text("https://example.com/guide", max_chars=200, client=client)
        assert "Edge Guide" in text
        assert "Hello Nomi" in text
        assert "First paragraph & details." in text
        assert "alert" not in text
        assert "display: none" not in text

        short = fetch_webpage_text("https://example.com/guide", max_chars=20, client=client)
        assert len(short) <= 20

    with pytest.raises(ValueError):
        fetch_webpage_text("file:///etc/passwd")
    with pytest.raises(ValueError):
        fetch_webpage_text("https://example.com;id")


def test_mac_local_blocks_traversal_ssh_keychain_and_sensitive_files(tmp_path: Path) -> None:
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_secret = outside / "secret.txt"
    outside_secret.write_text("top secret", encoding="utf-8")

    # Outside default and custom roots
    for blocked_path in (
        "~/.ssh/id_rsa",
        "~/Library/Keychains/login.keychain-db",
        str(outside_secret),
        str(desktop / ".." / "outside" / "secret.txt"),
    ):
        with pytest.raises(MacLocalAccessError):
            read_local_text_file(blocked_path, allowed_roots=[desktop])

    with pytest.raises(MacLocalAccessError):
        read_local_text_file("~/.ssh/id_rsa")
    with pytest.raises(MacLocalAccessError):
        read_local_text_file("~/Library/Keychains/login.keychain-db")

    # Symlink escape from Desktop to outside file
    escape_link = desktop / "escape.txt"
    escape_link.symlink_to(outside_secret)
    with pytest.raises(MacLocalAccessError):
        read_local_text_file(str(escape_link), allowed_roots=[desktop])

    # Sensitive files even when placed inside an allowed root
    sensitive_names = [
        ".env",
        ".hidden_note.txt",
        "server.key",
        "cert.pem",
        "outlook_token_cache.json",
        "custom.keychain",
        "login.keychain-db",
        "Cookies.binarycookies",
        "Login Data",
    ]
    for name in sensitive_names:
        target = desktop / name
        target.write_text("sensitive", encoding="utf-8")
        with pytest.raises(MacLocalAccessError):
            read_local_text_file(str(target), allowed_roots=[desktop])

    # Binary file rejection
    binary_file = desktop / "image.bin"
    binary_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    with pytest.raises(ValueError):
        read_local_text_file(str(binary_file), allowed_roots=[desktop])

    # Traversal and ~/.ssh blocked in list_recent_files
    for bad_dir in ("~/.ssh", str(outside), str(desktop / ".." / "outside")):
        with pytest.raises(MacLocalAccessError):
            list_recent_files(bad_dir, allowed_roots=[desktop])


def test_mac_local_list_recent_files_and_read_text_file(tmp_path: Path) -> None:
    desktop = tmp_path / "Desktop"
    desktop.mkdir()

    older = desktop / "older.txt"
    newer = desktop / "newer.md"
    newest = desktop / "newest.txt"
    older.write_text("old", encoding="utf-8")
    newer.write_text("mid", encoding="utf-8")
    newest.write_text("第一行内容\n第二行内容", encoding="utf-8")

    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    os.utime(newest, (3000, 3000))

    # Excluded entries inside Desktop
    (desktop / ".DS_Store").write_text("hidden", encoding="utf-8")
    (desktop / "api.key").write_text("key", encoding="utf-8")
    (desktop / "tls.pem").write_text("pem", encoding="utf-8")
    (desktop / "msal_token_cache.json").write_text("tok", encoding="utf-8")
    (desktop / "user.keychain-db").write_text("kc", encoding="utf-8")
    (desktop / "subdir").mkdir()

    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    (desktop / "symlink_out.txt").symlink_to(outside)

    recent = list_recent_files(str(desktop), limit=2, allowed_roots=[desktop])
    assert [item["name"] for item in recent] == ["newest.txt", "newer.md"]
    assert all(set(item.keys()) == {"name", "path", "size", "mtime"} for item in recent)

    all_safe = list_recent_files(str(desktop), limit=10, allowed_roots=[desktop])
    assert [item["name"] for item in all_safe] == ["newest.txt", "newer.md", "older.txt"]

    content = read_local_text_file(str(newest), max_chars=7, allowed_roots=[desktop])
    assert content == "第一行内..."
    assert len(content) == 7
    full = read_local_text_file(str(newest), max_chars=100, allowed_roots=[desktop])
    assert full == "第一行内容\n第二行内容"


def test_mcp_server_registers_and_dispatches_edge_and_mac_tools(
    tmp_path: Path, monkeypatch
) -> None:
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    note = desktop / "notes.txt"
    note.write_text("会议纪要：周三下午三点。", encoding="utf-8")
    (desktop / "secret.key").write_text("PRIVATE", encoding="utf-8")

    # Default McpServer preserves the original 5-tool tools/list contract.
    default_list = asyncio.run(
        McpServer(AppConfig()).handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    )
    assert default_list is not None
    assert len(default_list["result"]["tools"]) == 5

    # Scoped/extended McpServer advertises all 8 tools including the 3 new ones.
    server = McpServer(
        AppConfig(),
        include_local_tools=True,
        mac_allowed_roots=[desktop],
    )
    ext_list = asyncio.run(
        server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    )
    assert ext_list is not None
    ext_names = {tool["name"] for tool in ext_list["result"]["tools"]}
    assert ext_names == {tool["name"] for tool in ALL_TOOL_DEFINITIONS}
    assert {"browser_open_edge", "mac_list_recent_files", "mac_read_text_file"} <= ext_names

    # browser_open_edge dry_run and mocked execution
    dry_resp = asyncio.run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "browser_open_edge",
                    "arguments": {"url": "https://example.com", "dry_run": True},
                },
            }
        )
    )
    assert dry_resp is not None
    assert dry_resp["result"]["isError"] is False
    dry_data = json.loads(dry_resp["result"]["content"][0]["text"])
    assert dry_data["status"] == "dry_run"
    assert dry_data["command"] == ["open", "-a", "Microsoft Edge", "https://example.com"]

    launched: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: launched.append(list(argv))
        or subprocess.CompletedProcess(argv, 0, "", ""),
    )
    open_resp = asyncio.run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "browser_open_edge",
                    "arguments": {"url": "https://example.com/app"},
                },
            }
        )
    )
    assert open_resp is not None
    assert open_resp["result"]["isError"] is False
    assert launched == [["open", "-a", "Microsoft Edge", "https://example.com/app"]]

    bad_edge = asyncio.run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "browser_open_edge",
                    "arguments": {"url": "file:///etc/passwd"},
                },
            }
        )
    )
    assert bad_edge is not None
    assert bad_edge["result"]["isError"] is True

    # mac_list_recent_files allowed & blocked
    list_resp = asyncio.run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "mac_list_recent_files",
                    "arguments": {"directory": str(desktop), "limit": 5},
                },
            }
        )
    )
    assert list_resp is not None
    assert list_resp["result"]["isError"] is False
    listed = json.loads(list_resp["result"]["content"][0]["text"])
    assert listed["count"] == 1
    assert listed["files"][0]["name"] == "notes.txt"

    blocked_list = asyncio.run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "mac_list_recent_files",
                    "arguments": {"directory": "~/.ssh"},
                },
            }
        )
    )
    assert blocked_list is not None
    assert blocked_list["result"]["isError"] is True

    # mac_read_text_file allowed & blocked
    read_resp = asyncio.run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "mac_read_text_file",
                    "arguments": {"path": str(note), "max_chars": 100},
                },
            }
        )
    )
    assert read_resp is not None
    assert read_resp["result"]["isError"] is False
    read_data = json.loads(read_resp["result"]["content"][0]["text"])
    assert read_data["content"] == "会议纪要：周三下午三点。"

    for bad_path in (
        "~/Library/Keychains/login.keychain-db",
        str(desktop / ".." / "etc" / "passwd"),
        str(desktop / "secret.key"),
    ):
        blocked_read = asyncio.run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 9,
                    "method": "tools/call",
                    "params": {
                        "name": "mac_read_text_file",
                        "arguments": {"path": bad_path},
                    },
                }
            )
        )
        assert blocked_read is not None
        assert blocked_read["result"]["isError"] is True

    # browser_fetch_summary & browser_open_edge reject loopback/private SSRF URLs
    for tool_name in ("browser_fetch_summary", "browser_open_edge"):
        ssrf_resp = asyncio.run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": {"url": "http://127.0.0.1:4397/api/overview"},
                    },
                }
            )
        )
        assert ssrf_resp is not None
        assert ssrf_resp["result"]["isError"] is True

    # codex_analyze rejects sensitive/outside workdirs, NUL bytes, hidden symlinks, and broad $HOME cwd
    hidden_link = desktop / ".hidden_link"
    hidden_link.symlink_to(desktop)
    for bad_workdir in ("~/.ssh", "~/.config", str(tmp_path / "outside"), str(hidden_link), f"{desktop}\x00x"):
        codex_blocked = asyncio.run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 11,
                    "method": "tools/call",
                    "params": {
                        "name": "codex_analyze",
                        "arguments": {"prompt": "inspect", "workdir": bad_workdir},
                    },
                }
            )
        )
        assert codex_blocked is not None
        assert codex_blocked["result"]["isError"] is True
        err_payload = json.loads(codex_blocked["result"]["content"][0]["text"])
        assert err_payload["error"] == "access_denied"

    # When codex.workdir='.' resolves to $HOME, it must NOT silently allow $HOME or ~/Library
    home_cfg = AppConfig()
    home_cfg.llm.providers.codex.workdir = str(Path.home())
    home_server = McpServer(home_cfg, include_local_tools=True, mac_allowed_roots=[desktop])
    home_blocked = asyncio.run(
        home_server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {
                    "name": "codex_analyze",
                    "arguments": {"prompt": "inspect", "workdir": "   "},
                },
            }
        )
    )
    assert home_blocked is not None
    assert home_blocked["result"]["isError"] is True
    assert json.loads(home_blocked["result"]["content"][0]["text"])["error"] == "access_denied"


def test_host_voiceprint_guard_blocks_non_host_and_allows_host(tmp_path: Path) -> None:
    from xiao_speaker_assistant.voice.speaker_id import SpeakerMatch

    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    note = desktop / "plan.txt"
    note.write_text("秘密计划", encoding="utf-8")

    cfg = AppConfig()
    cfg.tools.require_host_voiceprint_for_sensitive_tools = True
    server = McpServer(cfg, include_local_tools=True, mac_allowed_roots=[desktop])

    def call_read() -> dict:
        resp = asyncio.run(
            server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "method": "tools/call",
                    "params": {
                        "name": "mac_read_text_file",
                        "arguments": {"path": str(note)},
                    },
                }
            )
        )
        assert resp is not None
        return resp["result"]

    # 1. Unknown speaker -> blocked
    r_unknown = call_read()
    assert r_unknown["isError"] is True
    assert json.loads(r_unknown["content"][0]["text"])["error"] == "host_voiceprint_required"

    # 2. Child/guest speaker -> blocked
    server.set_active_voice_speaker(
        SpeakerMatch(id="c1", name="拳拳", role="child", style_hint="", score=0.88)
    )
    r_child = call_read()
    assert r_child["isError"] is True
    assert json.loads(r_child["content"][0]["text"])["error"] == "host_voiceprint_required"

    # 3. Host speaker -> allowed
    server.set_active_voice_speaker(
        SpeakerMatch(id="h1", name="大雄", role="host", style_hint="", score=0.91)
    )
    r_host = call_read()
    assert r_host["isError"] is False
    assert "秘密计划" in r_host["content"][0]["text"]

    # 4. Non-sensitive tool (e.g. speaker_list_voices) remains allowed for child/unknown
    server.set_active_voice_speaker(None)
    voices_resp = asyncio.run(
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 100,
                "method": "tools/call",
                "params": {"name": "speaker_list_voices", "arguments": {}},
            }
        )
    )
    assert voices_resp is not None
    assert voices_resp["result"]["isError"] is False

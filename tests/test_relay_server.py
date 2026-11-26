from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.tools.history import ConversationHistoryStore
from xiao_speaker_assistant.tools.mcp_server import McpServer
from xiao_speaker_assistant.voice.relay_client import (
    RelayConsoleClient,
    RelayHostClient,
    normalize_console_ws_url,
    normalize_host_ws_url,
)
from xiao_speaker_assistant.voice.ws_server import SpeakerGateway


def _load_relay_module() -> ModuleType:
    relay_path = Path(__file__).resolve().parent.parent / "relay" / "server.py"
    spec = importlib.util.spec_from_file_location("xsa_relay_server", relay_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


relay_mod = _load_relay_module()


async def _expect_close_code(uri: str, expected_code: int, headers: dict[str, str] | None = None) -> None:
    expected_http_map = {4401: 401, 4404: 404, 4429: 429}
    if expected_code in expected_http_map:
        with pytest.raises(InvalidStatus) as exc_info:
            async with websockets.connect(uri, additional_headers=headers) as ws:
                await ws.recv()
        assert exc_info.value.response.status_code == expected_http_map[expected_code]
        return
    with pytest.raises(ConnectionClosed) as exc_info:
        async with websockets.connect(uri, additional_headers=headers) as ws:
            await ws.recv()
    rcvd = exc_info.value.rcvd
    assert rcvd is not None
    assert rcvd.code == expected_code


def test_unauthorized_connection_rejected_with_4401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XSA_RELAY_TOKEN", "top-secret-token")

    async def scenario() -> None:
        async with relay_mod.RelayServer(host="127.0.0.1", port=0) as relay:
            base = f"ws://127.0.0.1:{relay.bound_port}"
            await _expect_close_code(f"{base}/ws/speaker", 4401)
            await _expect_close_code(
                f"{base}/ws/host",
                4401,
                headers={"Authorization": "Bearer wrong-token"},
            )
            await _expect_close_code(
                f"{base}/ws/speaker",
                4401,
                headers={"Authorization": "Basic dXNlcjpwYXNz"},
            )
            # Query-param token is accepted ONLY for /ws/speaker, never /ws/host
            await _expect_close_code(f"{base}/ws/speaker?token=wrong-token", 4401)
            await _expect_close_code(f"{base}/ws/host?token=top-secret-token", 4401)

    asyncio.run(scenario())


def test_speaker_query_token_auth_allows_forwarding() -> None:
    token = "speaker-query-secret"

    async def scenario() -> None:
        async with relay_mod.RelayServer(host="127.0.0.1", port=0, token=token) as relay:
            base = f"ws://127.0.0.1:{relay.bound_port}"
            async with (
                websockets.connect(f"{base}/ws/speaker?token={token}") as speaker,
                websockets.connect(
                    f"{base}/ws/host",
                    additional_headers={"Authorization": f"Bearer {token}"},
                ) as host,
            ):
                await speaker.send('{"Event":{"id":"q1"}}')
                assert await asyncio.wait_for(host.recv(), timeout=2.0) == '{"Event":{"id":"q1"}}'

    asyncio.run(scenario())


def test_unknown_endpoint_rejected_with_4404() -> None:
    async def scenario() -> None:
        async with relay_mod.RelayServer(host="127.0.0.1", port=0, token="tok") as relay:
            await _expect_close_code(
                f"ws://127.0.0.1:{relay.bound_port}/ws/unknown",
                4404,
                headers={"Authorization": "Bearer tok"},
            )

    asyncio.run(scenario())


def test_bidirectional_text_and_binary_forwarding() -> None:
    token = "relay-test-token"
    headers = {"Authorization": f"Bearer {token}"}

    async def scenario() -> None:
        async with relay_mod.RelayServer(host="127.0.0.1", port=0, token=token) as relay:
            base = f"ws://127.0.0.1:{relay.bound_port}"
            async with (
                websockets.connect(f"{base}/ws/speaker", additional_headers=headers) as speaker,
                websockets.connect(f"{base}/ws/host", additional_headers=headers) as host,
            ):
                speaker_event = json.dumps({"Event": {"id": "1", "text": "你好"}}, ensure_ascii=False)
                await speaker.send(speaker_event)
                assert await asyncio.wait_for(host.recv(), timeout=2.0) == speaker_event

                host_cmd = json.dumps(
                    {"Request": {"id": "2", "command": "run_shell", "payload": "echo hi"}},
                    ensure_ascii=False,
                )
                await host.send(host_cmd)
                assert await asyncio.wait_for(speaker.recv(), timeout=2.0) == host_cmd

                pcm_chunk = b"\x00\x01\x02\x03\xff\xfe\xfd\xfc" * 64
                await speaker.send(pcm_chunk)
                assert await asyncio.wait_for(host.recv(), timeout=2.0) == pcm_chunk

                binary_reply = b"\xaa\xbb\xcc\xdd" * 16
                await host.send(binary_reply)
                assert await asyncio.wait_for(speaker.recv(), timeout=2.0) == binary_reply

    asyncio.run(scenario())


def test_reconnecting_same_role_replaces_old_connection_cleanly() -> None:
    token = "replace-token"
    headers = {"Authorization": f"Bearer {token}"}

    async def scenario() -> None:
        async with relay_mod.RelayServer(host="127.0.0.1", port=0, token=token) as relay:
            base = f"ws://127.0.0.1:{relay.bound_port}"
            speaker1 = await websockets.connect(f"{base}/ws/speaker", additional_headers=headers)
            host1 = await websockets.connect(f"{base}/ws/host", additional_headers=headers)
            try:
                await speaker1.send('{"msg":"from-s1"}')
                assert await asyncio.wait_for(host1.recv(), timeout=2.0) == '{"msg":"from-s1"}'

                speaker2 = await websockets.connect(f"{base}/ws/speaker", additional_headers=headers)
                try:
                    with pytest.raises(ConnectionClosed) as exc_info:
                        await asyncio.wait_for(speaker1.recv(), timeout=2.0)
                    assert exc_info.value.rcvd is not None
                    assert exc_info.value.rcvd.code == 4000

                    await speaker2.send('{"msg":"from-s2"}')
                    assert await asyncio.wait_for(host1.recv(), timeout=2.0) == '{"msg":"from-s2"}'

                    await host1.send('{"msg":"to-s2"}')
                    assert await asyncio.wait_for(speaker2.recv(), timeout=2.0) == '{"msg":"to-s2"}'

                    host2 = await websockets.connect(f"{base}/ws/host", additional_headers=headers)
                    try:
                        with pytest.raises(ConnectionClosed) as host_exc:
                            await asyncio.wait_for(host1.recv(), timeout=2.0)
                        assert host_exc.value.rcvd is not None
                        assert host_exc.value.rcvd.code == 4000

                        await speaker2.send(b"\x10\x20\x30")
                        assert await asyncio.wait_for(host2.recv(), timeout=2.0) == b"\x10\x20\x30"
                    finally:
                        await host2.close()
                finally:
                    await speaker2.close()
            finally:
                await speaker1.close()
                await host1.close()

    asyncio.run(scenario())


def test_normalize_host_ws_url() -> None:
    assert normalize_host_ws_url("wss://relay.example.com") == "wss://relay.example.com/ws/host"
    assert normalize_host_ws_url("wss://relay.example.com/") == "wss://relay.example.com/ws/host"
    assert normalize_host_ws_url("ws://127.0.0.1:8765/ws/host") == "ws://127.0.0.1:8765/ws/host"
    assert normalize_host_ws_url("wss://relay.example.com/prefix") == "wss://relay.example.com/prefix/ws/host"
    with pytest.raises(ValueError):
        normalize_host_ws_url("https://relay.example.com")


def test_relay_host_client_forwards_frames_to_speaker_gateway() -> None:
    token = "gateway-token"
    received_queries: list[str] = []
    received_binary: list[bytes] = []

    async def on_query(query: str) -> str:
        received_queries.append(query)
        return f"回答：{query}"

    async def scenario() -> None:
        config = AppConfig()
        config.speaker.tts_provider = "xiaoai"
        config.speech.asr_provider = "xiaoai_asr"
        gateway = SpeakerGateway(config, on_query=on_query)

        orig_binary_handler = gateway._handle_binary_message

        async def spy_binary(conn: object, raw: bytes) -> None:
            received_binary.append(raw)
            await orig_binary_handler(conn, raw)  # type: ignore[arg-type]

        gateway._handle_binary_message = spy_binary  # type: ignore[method-assign]

        async with relay_mod.RelayServer(host="127.0.0.1", port=0, token=token) as relay:
            base = f"ws://127.0.0.1:{relay.bound_port}"
            client = RelayHostClient(
                gateway,
                base,
                auth_token=token,
                initial_backoff=0.05,
                max_backoff=0.2,
            )
            task = asyncio.create_task(client.serve_forever())
            try:
                await client.wait_connected(timeout=3.0)
                async with websockets.connect(
                    f"{base}/ws/speaker",
                    additional_headers={"Authorization": f"Bearer {token}"},
                ) as speaker:
                    line = json.dumps(
                        {
                            "header": {
                                "namespace": "SpeechRecognizer",
                                "name": "RecognizeResult",
                                "dialog_id": "dlg-relay-1",
                            },
                            "payload": {
                                "is_final": True,
                                "results": [{"text": "帮我查一下日程"}],
                            },
                        },
                        ensure_ascii=False,
                    )
                    asr_event = json.dumps(
                        {
                            "Event": {
                                "id": "evt-relay-1",
                                "event": "instruction",
                                "data": {"NewLine": line},
                            }
                        },
                        ensure_ascii=False,
                    )
                    await speaker.send(asr_event)

                    commands: list[dict[str, object]] = []
                    for _ in range(2):
                        raw_cmd = await asyncio.wait_for(speaker.recv(), timeout=3.0)
                        assert isinstance(raw_cmd, str)
                        commands.append(json.loads(raw_cmd))

                    assert received_queries == ["查一下日程"]
                    tts_payload = commands[1]["Request"]["payload"]  # type: ignore[index]
                    assert "回答：查一下日程" in tts_payload

                    await speaker.send(b"\x01\x02\x03\x04")
                    for _ in range(40):
                        if received_binary:
                            break
                        await asyncio.sleep(0.02)
                    assert received_binary == [b"\x01\x02\x03\x04"]
            finally:
                await client.stop()
                await asyncio.wait_for(task, timeout=3.0)

    asyncio.run(scenario())


def test_relay_refuses_to_start_without_token_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XSA_RELAY_TOKEN", raising=False)
    monkeypatch.delenv("XSA_RELAY_HOST_TOKEN", raising=False)
    monkeypatch.delenv("XSA_RELAY_SPEAKER_TOKEN", raising=False)
    with pytest.raises(ValueError, match="token"):
        relay_mod.RelayServer(host="127.0.0.1", port=0)


def test_relay_refuses_half_configured_split_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XSA_RELAY_TOKEN", "legacy-should-not-mix")
    monkeypatch.setenv("XSA_RELAY_HOST_TOKEN", "only-host-set")
    monkeypatch.delenv("XSA_RELAY_SPEAKER_TOKEN", raising=False)
    with pytest.raises(ValueError, match="Both XSA_RELAY_HOST_TOKEN and XSA_RELAY_SPEAKER_TOKEN"):
        relay_mod.RelayServer(host="127.0.0.1", port=0)


def test_split_host_and_speaker_tokens_enforce_role_separation() -> None:
    host_tok = "host-secret-only"
    spk_tok = "speaker-secret-only"

    async def scenario() -> None:
        async with relay_mod.RelayServer(
            host="127.0.0.1",
            port=0,
            host_token=host_tok,
            speaker_token=spk_tok,
        ) as relay:
            assert relay.separate_role_tokens is True
            base = f"ws://127.0.0.1:{relay.bound_port}"
            # Speaker token MUST NOT authenticate /ws/host (prevents speaker compromise -> RCE/spoof host)
            await _expect_close_code(
                f"{base}/ws/host",
                4401,
                headers={"Authorization": f"Bearer {spk_tok}"},
            )
            await _expect_close_code(
                f"{base}/xsa/ws/host",
                4401,
                headers={"Authorization": f"Bearer {spk_tok}"},
            )
            # Host token MUST NOT authenticate /ws/speaker
            await _expect_close_code(
                f"{base}/ws/speaker?token={host_tok}",
                4401,
            )
            # Matching role tokens succeed on both /ws/* and /xsa/ws/* prefix
            async with (
                websockets.connect(f"{base}/xsa/ws/speaker?token={spk_tok}") as speaker,
                websockets.connect(
                    f"{base}/xsa/ws/host",
                    additional_headers={"Authorization": f"Bearer {host_tok}"},
                ) as host,
            ):
                await host.send('{"Request":{"id":"r1","command":"ping"}}')
                assert await asyncio.wait_for(speaker.recv(), timeout=2.0) == '{"Request":{"id":"r1","command":"ping"}}'
                await speaker.send('{"Response":{"id":"r1","code":0}}')
                assert await asyncio.wait_for(host.recv(), timeout=2.0) == '{"Response":{"id":"r1","code":0}}'

    asyncio.run(scenario())


def test_relay_rate_limits_repeated_auth_failures_and_ignores_spoofed_xff() -> None:
    async def scenario() -> None:
        async with relay_mod.RelayServer(
            host="127.0.0.1",
            port=0,
            host_token="good-host",
            speaker_token="good-spk",
            auth_fail_limit=3,
            auth_fail_window_sec=60.0,
            auth_ban_sec=30.0,
        ) as relay:
            base = f"ws://127.0.0.1:{relay.bound_port}"
            # Spoofing different X-Forwarded-For values MUST NOT bypass rate limiting when untrusted.
            await _expect_close_code(
                f"{base}/ws/host",
                4401,
                headers={"Authorization": "Bearer bad1", "X-Forwarded-For": "1.1.1.1"},
            )
            await _expect_close_code(
                f"{base}/ws/unknown",
                4401,
                headers={"Authorization": "Bearer bad2", "X-Forwarded-For": "2.2.2.2"},
            )
            # 3rd failure hits ban threshold -> 4429 despite another spoofed XFF
            await _expect_close_code(
                f"{base}/ws/host",
                4429,
                headers={"Authorization": "Bearer bad3", "X-Forwarded-For": "3.3.3.3"},
            )
            # Subsequent attempt (even with valid token) from banned IP is still rejected with 4429
            await _expect_close_code(f"{base}/ws/host", 4429, headers={"Authorization": "Bearer good-host"})

    asyncio.run(scenario())


def test_relay_trusted_proxy_uses_rightmost_untrusted_xff_hop() -> None:
    async def scenario() -> None:
        async with relay_mod.RelayServer(
            host="127.0.0.1",
            port=0,
            host_token="good-host",
            speaker_token="good-spk",
            auth_fail_limit=2,
            auth_fail_window_sec=60.0,
            auth_ban_sec=30.0,
            trusted_proxies={"127.0.0.1"},
        ) as relay:
            base = f"ws://127.0.0.1:{relay.bound_port}"
            # Attacker sends spoofed leftmost XFF, trusted proxy appends real IP (203.0.113.9) on the right.
            await _expect_close_code(
                f"{base}/ws/host",
                4401,
                headers={"Authorization": "Bearer bad1", "X-Forwarded-For": "10.0.0.1, 203.0.113.9"},
            )
            await _expect_close_code(
                f"{base}/ws/host",
                4429,
                headers={"Authorization": "Bearer bad2", "X-Forwarded-For": "10.0.0.2, 203.0.113.9"},
            )
            # A different real client IP (203.0.113.55) behind the same proxy is NOT banned.
            async with websockets.connect(
                f"{base}/ws/host",
                additional_headers={
                    "Authorization": "Bearer good-host",
                    "X-Forwarded-For": "203.0.113.55",
                },
            ):
                pass

    asyncio.run(scenario())


def test_relay_http_health_and_non_ws_paths() -> None:
    import httpx

    async def scenario() -> None:
        async with relay_mod.RelayServer(
            host="127.0.0.1",
            port=0,
            host_token="good-host",
            speaker_token="good-spk",
        ) as relay:
            base = f"http://127.0.0.1:{relay.bound_port}"
            async with httpx.AsyncClient() as client:
                for ok_path in ("/", "/health", "/healthz"):
                    r = await client.get(f"{base}{ok_path}")
                    assert r.status_code == 200
                for bad_path in ("/ws/host", "/ws/speaker", "/xsa/ws/host", "/unknown"):
                    r = await client.get(f"{base}{bad_path}")
                    assert r.status_code == 404

    asyncio.run(scenario())


def test_relay_host_client_prefers_host_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XSA_RELAY_TOKEN", "legacy-shared")
    monkeypatch.setenv("XSA_RELAY_HOST_TOKEN", "dedicated-host-tok")
    cfg = AppConfig()
    cfg.speech.asr_provider = "xiaoai_asr"

    async def dummy(q: str) -> str:
        return q

    gw = SpeakerGateway(cfg, on_query=dummy)
    client = RelayHostClient(gw, "wss://relay.example.com/xsa/ws/host")
    assert client.relay_url == "wss://relay.example.com/xsa/ws/host"
    assert client._build_headers() == {"Authorization": "Bearer dedicated-host-tok"}


def test_relay_host_client_reconnects_with_backoff() -> None:
    token = "reconnect-token"

    async def on_query(query: str) -> str:
        return query

    async def scenario() -> None:
        # Reserve an unused ephemeral port, then close it so initial dials fail.
        probe = relay_mod.RelayServer(host="127.0.0.1", port=0, token=token)
        await probe.start()
        port = probe.bound_port
        await probe.close()

        cfg = AppConfig()
        cfg.speech.asr_provider = "xiaoai_asr"
        gateway = SpeakerGateway(cfg, on_query=on_query)
        client = RelayHostClient(
            gateway,
            f"ws://127.0.0.1:{port}/ws/host",
            auth_token=token,
            initial_backoff=0.05,
            max_backoff=0.2,
            backoff_factor=2.0,
        )
        task = asyncio.create_task(client.serve_forever())
        relay: relay_mod.RelayServer | None = None
        try:
            await asyncio.sleep(0.2)
            assert not client.is_connected

            relay = relay_mod.RelayServer(host="127.0.0.1", port=port, token=token)
            await relay.start()
            await client.wait_connected(timeout=3.0)
            assert client.is_connected
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3.0)
            if relay is not None:
                await relay.close()

    asyncio.run(scenario())


def test_normalize_console_ws_url() -> None:
    assert normalize_console_ws_url("wss://relay.example.com") == "wss://relay.example.com/ws/console"
    assert (
        normalize_console_ws_url("wss://relay.example.com/ws/host")
        == "wss://relay.example.com/ws/console"
    )
    assert (
        normalize_console_ws_url("wss://relay.example.com/xsa/ws/host")
        == "wss://relay.example.com/xsa/ws/console"
    )
    assert (
        normalize_console_ws_url("wss://relay.example.com/xsa")
        == "wss://relay.example.com/xsa/ws/console"
    )
    assert (
        normalize_console_ws_url("wss://relay.example.com/xsa/ws/console")
        == "wss://relay.example.com/xsa/ws/console"
    )
    with pytest.raises(ValueError):
        normalize_console_ws_url("https://relay.example.com")


def test_console_ws_auth_requires_host_token_only() -> None:
    host_tok = "host-console-secret"
    spk_tok = "speaker-only-secret"

    async def scenario() -> None:
        async with relay_mod.RelayServer(
            host="127.0.0.1",
            port=0,
            console_port=0,
            host_token=host_tok,
            speaker_token=spk_tok,
        ) as relay:
            base = f"ws://127.0.0.1:{relay.bound_port}"
            # Speaker token and query-param token MUST NOT authenticate /ws/console
            await _expect_close_code(
                f"{base}/xsa/ws/console",
                4401,
                headers={"Authorization": f"Bearer {spk_tok}"},
            )
            await _expect_close_code(
                f"{base}/xsa/ws/console?token={host_tok}",
                4401,
            )
            async with websockets.connect(
                f"{base}/xsa/ws/console",
                additional_headers={"Authorization": f"Bearer {host_tok}"},
            ):
                pass

    asyncio.run(scenario())


def test_console_relay_e2e_auth_cookie_mcp_block_and_xff_rate_limit(tmp_path: Path) -> None:
    import httpx

    host_tok = "host-secret-console"
    spk_tok = "speaker-secret-console"
    console_pwd = "super-secret-console-password"

    async def scenario() -> None:
        cfg = AppConfig()
        cfg.memory.file_path = str(tmp_path / "memories.json")
        cfg.reminders.file_path = str(tmp_path / "reminders.json")
        cfg.speech.speaker_profiles_file = str(tmp_path / "speakers.json")
        mcp = McpServer(
            cfg,
            history_store=ConversationHistoryStore(tmp_path / "history.jsonl"),
            reminders_path=tmp_path / "reminders.json",
            console_password=console_pwd,
        )
        runner, mcp_port = await mcp.start_http_server(
            host="127.0.0.1",
            port=0,
            auth_token="internal-mcp-token",
            console_password=console_pwd,
        )
        try:
            async with relay_mod.RelayServer(
                host="127.0.0.1",
                port=0,
                console_port=0,
                host_token=host_tok,
                speaker_token=spk_tok,
                trusted_proxies={"127.0.0.1"},
            ) as relay:
                assert relay.bound_console_port is not None
                console_base = f"http://127.0.0.1:{relay.bound_console_port}"

                async with httpx.AsyncClient() as client:
                    # Before Mac connects: /healthz reports offline, console returns 503
                    hz = await client.get(f"{console_base}/healthz")
                    assert hz.status_code == 200
                    assert hz.json()["console_connected"] is False
                    offline = await client.get(f"{console_base}/")
                    assert offline.status_code == 503

                console_client = RelayConsoleClient(
                    relay_url=f"ws://127.0.0.1:{relay.bound_port}/xsa/ws/host",
                    local_console_url=f"http://127.0.0.1:{mcp_port}",
                    auth_token=host_tok,
                    initial_backoff=0.05,
                    max_backoff=0.2,
                )
                task = asyncio.create_task(console_client.serve_forever())
                try:
                    await console_client.wait_connected(timeout=3.0)
                    async with httpx.AsyncClient() as client:
                        # 1. Healthz reports connected
                        hz2 = await client.get(f"{console_base}/healthz")
                        assert hz2.status_code == 200
                        assert hz2.json()["console_connected"] is True

                        # 2. Console HTML loads (200)
                        idx = await client.get(
                            f"{console_base}/",
                            headers={"Host": "nomi.mxhan.com", "X-Forwarded-Proto": "https"},
                        )
                        assert idx.status_code == 200
                        assert "text/html" in idx.headers.get("content-type", "")

                        # 3. /api/auth/status MUST treat tunneled traffic as remote (is_local=False)
                        st = await client.get(
                            f"{console_base}/api/auth/status",
                            headers={"Host": "nomi.mxhan.com", "X-Forwarded-Proto": "https"},
                        )
                        assert st.status_code == 200
                        assert st.json() == {
                            "authenticated": False,
                            "auth_required": True,
                            "is_local": False,
                        }

                        # 4. Unauthenticated /api/overview -> 401
                        ov_unauth = await client.get(
                            f"{console_base}/api/overview",
                            headers={"Host": "nomi.mxhan.com", "X-Forwarded-Proto": "https"},
                        )
                        assert ov_unauth.status_code == 401

                        # 5. /mcp and traversal variants -> 403
                        for blocked in ("/mcp", "/api/../mcp", "/api/%2e%2e/mcp", "/admin"):
                            r_blk = await client.post(
                                f"{console_base}{blocked}",
                                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                headers={"Host": "nomi.mxhan.com"},
                            )
                            assert r_blk.status_code == 403, f"expected 403 for {blocked}, got {r_blk.status_code}"

                        # 6. Spoofed leftmost X-Forwarded-For MUST NOT bypass login rate limit (5/min)
                        for idx_fail in range(5):
                            r_fail = await client.post(
                                f"{console_base}/api/login",
                                json={"password": f"wrong-{idx_fail}"},
                                headers={
                                    "Host": "nomi.mxhan.com",
                                    "Origin": "https://nomi.mxhan.com",
                                    "X-Forwarded-Proto": "https",
                                    "X-Forwarded-For": f"10.0.0.{idx_fail}, 198.51.100.77",
                                },
                            )
                            assert r_fail.status_code == 401
                        r_limited = await client.post(
                            f"{console_base}/api/login",
                            json={"password": console_pwd},
                            headers={
                                "Host": "nomi.mxhan.com",
                                "Origin": "https://nomi.mxhan.com",
                                "X-Forwarded-Proto": "https",
                                "X-Forwarded-For": "10.0.0.99, 198.51.100.77",
                            },
                        )
                        assert r_limited.status_code == 429

                        # 7. Real client IP logs in with console_password, receives HttpOnly+Secure cookie,
                        #    and can fetch /api/overview (200)
                        r_login = await client.post(
                            f"{console_base}/api/login",
                            json={"password": console_pwd},
                            headers={
                                "Host": "nomi.mxhan.com",
                                "Origin": "https://nomi.mxhan.com",
                                "X-Forwarded-Proto": "https",
                                "X-Forwarded-For": "203.0.113.42",
                            },
                        )
                        assert r_login.status_code == 200
                        set_cookie = r_login.headers.get("set-cookie", "")
                        assert "xsa_console_session=" in set_cookie
                        assert "HttpOnly" in set_cookie
                        assert "Secure" in set_cookie
                        cookie_pair = set_cookie.split(";", 1)[0]

                        r_ov = await client.get(
                            f"{console_base}/api/overview",
                            headers={
                                "Host": "nomi.mxhan.com",
                                "X-Forwarded-Proto": "https",
                                "Cookie": cookie_pair,
                            },
                        )
                        assert r_ov.status_code == 200
                        assert "capabilities" in r_ov.json()
                finally:
                    await console_client.stop()
                    await asyncio.wait_for(task, timeout=3.0)
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_console_client_defense_in_depth_path_filter() -> None:
    assert RelayConsoleClient._is_path_allowed("/") is True
    assert RelayConsoleClient._is_path_allowed("/health") is True
    assert RelayConsoleClient._is_path_allowed("/api/overview?foo=1") is True
    assert RelayConsoleClient._is_path_allowed("/mcp") is False
    assert RelayConsoleClient._is_path_allowed("/api/../mcp") is False
    assert RelayConsoleClient._is_path_allowed("/api/%2e%2e/mcp") is False
    assert RelayConsoleClient._is_path_allowed("http://127.0.0.1:4397/api/overview") is False
    assert RelayConsoleClient._is_path_allowed("//evil.example/api/overview") is False

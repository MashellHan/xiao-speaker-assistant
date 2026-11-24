from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import websockets
from websockets.exceptions import ConnectionClosed

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.voice.relay_client import RelayHostClient, normalize_host_ws_url
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

from __future__ import annotations

import asyncio
import importlib.util
import json
import plistlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import websockets

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.llm.opencode_server import load_or_create_mcp_token
from xiao_speaker_assistant.voice.relay_client import RelayHostClient, run_with_relay
from xiao_speaker_assistant.voice.ws_server import SpeakerGateway

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tcp_bridge_mod = _load_script("xsa_tcp_bridge", "scripts/tcp_bridge.py")
launchagent_mod = _load_script("xsa_launchagent_p3", "scripts/install_launchagent.py")
relay_path = REPO_ROOT / "relay" / "server.py"
relay_spec = importlib.util.spec_from_file_location("xsa_relay_server_p3", relay_path)
assert relay_spec is not None and relay_spec.loader is not None
relay_mod = importlib.util.module_from_spec(relay_spec)
sys.modules[relay_spec.name] = relay_mod
relay_spec.loader.exec_module(relay_mod)


def test_relay_host_client_sends_configured_start_recording_and_cleans_up() -> None:
    token = "p3-relay-tok"
    cfg = AppConfig()
    cfg.speech.asr_provider = "xiaoai_asr"
    cfg.speech.capture_pcm = "xsa_aec"
    cfg.speech.capture_channels = 2
    cfg.speech.capture_bits_per_sample = 32
    cfg.speaker.tts_provider = "edge"

    async def fake_stream(q: str):
        yield f"流式回答：{q}"

    gw = SpeakerGateway(
        cfg,
        on_stream_query=fake_stream,
        audio_watchdog_interval=0.05,
        audio_stall_timeout=0.1,
    )
    # Inject a fake pipeline so start_recording + watchdog are exercised without loading 470 MB models.
    gw._pipeline = MagicMock()

    async def scenario() -> None:
        async with relay_mod.RelayServer(host="127.0.0.1", port=0, token=token) as relay:
            base = f"ws://127.0.0.1:{relay.bound_port}"
            client = RelayHostClient(gateway=gw, relay_url=base, auth_token=token)
            task = asyncio.create_task(client.serve_forever())
            try:
                await client.wait_connected(timeout=3.0)
                # Connect speaker AFTER host is connected to verify watchdog recovers start_recording!
                async with websockets.connect(
                    f"{base}/ws/speaker",
                    additional_headers={"Authorization": f"Bearer {token}"},
                ) as speaker:
                    raw_cmd = await asyncio.wait_for(speaker.recv(), timeout=3.0)
                    cmd = json.loads(raw_cmd)
                    req = cmd["Request"]
                    assert req["command"] == "start_recording"
                    assert req["payload"]["pcm"] == "xsa_aec"
                    assert req["payload"]["channels"] == 2
                    assert req["payload"]["bits_per_sample"] == 32
                    assert req["payload"]["period_size"] == 360
                    assert len(gw._sessions) == 1
                    assert len(gw._playbacks) == 1
            finally:
                await client.stop()
                await asyncio.wait_for(task, timeout=3.0)
        assert gw._sessions == {}
        assert gw._playbacks == {}
        assert gw._pending_rpc == {}

    asyncio.run(scenario())


def test_enqueue_audio_frame_bounds_queue_by_dropping_oldest() -> None:
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=2)
    SpeakerGateway.enqueue_audio_frame(q, b"frame-1")
    SpeakerGateway.enqueue_audio_frame(q, b"frame-2")
    SpeakerGateway.enqueue_audio_frame(q, b"frame-3")
    assert q.qsize() == 2
    assert q.get_nowait() == b"frame-2"
    assert q.get_nowait() == b"frame-3"


def test_per_connection_session_isolation() -> None:
    async def dummy(q: str) -> str:
        return q

    cfg = AppConfig()
    cfg.speech.asr_provider = "xiaoai_asr"
    gw = SpeakerGateway(cfg, on_query=dummy)

    class FakeConn:
        def __init__(self, name: str) -> None:
            self.name = name

        async def send(self, _raw) -> None:
            pass

    c1, c2 = FakeConn("c1"), FakeConn("c2")
    s1 = gw._get_session(c1)
    s2 = gw._get_session(c2)
    s1.continuous_mode = True
    s1.seen_dialogs.add("dlg-1:hi")
    gw._get_playback(c1)
    gw._get_playback(c2)

    assert s2.continuous_mode is False
    assert "dlg-1:hi" not in s2.seen_dialogs

    gw._cleanup_connection(c1)
    assert c1 not in gw._sessions
    assert c1 not in gw._playbacks
    assert c2 in gw._sessions
    assert c2 in gw._playbacks


def test_tcp_bridge_closes_client_when_upstream_closes() -> None:
    async def scenario() -> None:
        async def upstream_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            data = await reader.read(100)
            writer.write(data.upper())
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
        up_port = upstream.sockets[0].getsockname()[1]
        bridge = await asyncio.start_server(
            lambda r, w: tcp_bridge_mod.handle_client(r, w, "127.0.0.1", up_port),
            "127.0.0.1",
            0,
        )
        br_port = bridge.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", br_port)
            writer.write(b"hello")
            await writer.drain()
            reply = await asyncio.wait_for(reader.read(), timeout=2.0)
            assert reply == b"HELLO"
            writer.close()
            await writer.wait_closed()
        finally:
            bridge.close()
            await bridge.wait_closed()
            upstream.close()
            await upstream.wait_closed()

    asyncio.run(scenario())


def test_launchagent_cloud_asr_model_and_0600_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "env"
    env_file.write_text("OUTLOOK_CLIENT_ID=abc\n", encoding="utf-8")
    env_file.chmod(0o644)

    # Rejects group/world-readable env files
    rc_bad = launchagent_mod.main(
        ["--repo-root", str(tmp_path), "--env-file", str(env_file)]
    )
    assert rc_bad == 2

    env_file.chmod(0o600)
    plist_bytes = launchagent_mod.render_launchagent_plist(
        repo_root=tmp_path,
        asr="sense_voice",
        asr_model="doubao_flash",
        env_file=env_file,
    )
    data = plistlib.loads(plist_bytes)
    assert "--asr-model" in data["ProgramArguments"]
    assert "doubao_flash" in data["ProgramArguments"]
    assert data["EnvironmentVariables"]["XSA_ENV_FILE"] == str(env_file.resolve())

    tok1 = load_or_create_mcp_token(tmp_path / "agent-home")
    tok2 = load_or_create_mcp_token(tmp_path / "agent-home")
    assert tok1 == tok2 and len(tok1) >= 16
    assert ((tmp_path / "agent-home" / "mcp.token").stat().st_mode & 0o777) == 0o600


def test_run_with_relay_passes_streaming_and_reset_callbacks(monkeypatch) -> None:
    created = {}

    class FakeGateway:
        def __init__(self, config, *, on_query=None, on_stream_query=None, on_session_reset=None):
            created["on_query"] = on_query
            created["on_stream_query"] = on_stream_query
            created["on_session_reset"] = on_session_reset

        def ensure_pipeline(self):
            created["ensured"] = True

        async def serve_forever(self):
            pass

    class FakeRelayClient:
        def __init__(self, gateway, **kwargs):
            created["client_gateway"] = gateway

        async def serve_forever(self):
            pass

    import xiao_speaker_assistant.voice.relay_client as rc_mod

    monkeypatch.setattr(rc_mod, "SpeakerGateway", FakeGateway)
    monkeypatch.setattr(rc_mod, "RelayHostClient", FakeRelayClient)

    async def q(x: str) -> str:
        return x

    async def sq(x: str):
        yield x

    def rst() -> None:
        pass

    asyncio.run(
        run_with_relay(
            AppConfig(),
            on_query=q,
            relay_url="wss://relay.example.com",
            on_stream_query=sq,
            on_session_reset=rst,
        )
    )
    assert created["on_query"] is q
    assert created["on_stream_query"] is sq
    assert created["on_session_reset"] is rst
    assert created["ensured"] is True

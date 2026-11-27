"""Tests for async cloud ASR non-blocking pipeline, pluggable web search, and cross-platform helpers."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from typing import Any

import httpx
import numpy as np
import pytest

from xiao_speaker_assistant.config import AppConfig, SearchToolConfig, SpeechConfig
from xiao_speaker_assistant.tools.browser import build_edge_open_command
from xiao_speaker_assistant.tools.mcp_server import McpServer
from xiao_speaker_assistant.tools.web_search import resolve_search_provider, run_web_search
from xiao_speaker_assistant.voice.audio_pipeline import LocalSpeechPipeline
from xiao_speaker_assistant.voice.tts_stream import synthesize_local_fallback_pcm
from xiao_speaker_assistant.voice.ws_server import SpeakerGateway


class _FakeKWSStream:
    def __init__(self, kws: _FakeKWS) -> None:
        self._kws = kws

    def accept_waveform(self, _sr: int, _samples: Any) -> None:
        if self._kws._hits and not self._kws._pending:
            self._kws._pending = self._kws._hits.pop(0)


class _FakeKWS:
    def __init__(self, hits: list[str] | None = None) -> None:
        self._hits = list(hits or [])
        self._pending: str = ""

    def create_stream(self) -> _FakeKWSStream:
        return _FakeKWSStream(self)

    def reset_stream(self, _stream: object) -> None:
        self._pending = ""

    def is_ready(self, _stream: object) -> bool:
        return bool(self._pending)

    def decode_stream(self, _stream: object) -> None:
        return None

    def get_result(self, _stream: object) -> str:
        hit = self._pending
        self._pending = ""
        return hit


class _FakeVadSegment:
    def __init__(self, samples: np.ndarray) -> None:
        self.samples = samples


class _FakeVAD:
    def __init__(self, segments: list[np.ndarray]) -> None:
        self._segments = list(segments)

    def reset(self) -> None:
        self._segments.clear()

    def accept_waveform(self, _chunk: Any) -> None:
        return None

    def is_speech_detected(self) -> bool:
        return False

    def empty(self) -> bool:
        return len(self._segments) == 0

    @property
    def front(self) -> _FakeVadSegment:
        return _FakeVadSegment(self._segments[0])

    def pop(self) -> None:
        self._segments.pop(0)


class _SlowCloudASR:
    def __init__(self, delay_sec: float = 0.25, text: str = "今天北京天气怎么样") -> None:
        self.delay_sec = delay_sec
        self.text = text
        self.sync_called = False
        self.async_called = 0

    def transcribe(self, _samples: Any, *, sample_rate: int | None = None) -> str:
        self.sync_called = True
        time.sleep(self.delay_sec)
        return self.text

    async def atranscribe(self, _samples: Any, *, sample_rate: int | None = None) -> str:
        self.async_called += 1
        await asyncio.sleep(self.delay_sec)
        return self.text


class _DummyConn:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send(self, data: Any) -> None:
        self.sent.append(data)


def _make_stub_pipeline(cloud_asr: _SlowCloudASR, segments: list[np.ndarray], kws_hits: list[str] | None = None) -> LocalSpeechPipeline:
    p = LocalSpeechPipeline.__new__(LocalSpeechPipeline)
    p._config = SpeechConfig(asr_model="openai_whisper", aec_enabled=False, agc_enabled=False, input_gain=1.0)
    p._lock = threading.RLock()
    p._wake_set = {"糯米糯米"}
    p._interrupt_set = {"退出", "停止"}
    p._kws = _FakeKWS(kws_hits)
    p._kws_stream = p._kws.create_stream()
    p._en_kws = None
    p._en_kws_stream = None
    p._vad = _FakeVAD(segments)
    p._cloud_asr = cloud_asr
    p._asr = None
    p.awake = True
    p._last_active_at = time.monotonic()
    p._muted_until = 0.0
    p._explicit_muted = False
    p._vad_buffer = np.zeros(0, dtype=np.float32)
    p._logged_first_audio = True
    p._hp_prev_x = 0.0
    p._hp_prev_y = 0.0
    p._hp_ref_prev_x = 0.0
    p._hp_ref_prev_y = 0.0
    p._last_asr_metrics = {}
    p._aec = None
    p._agc = None
    p._speaker_id = None
    return p


def test_async_cloud_asr_does_not_block_feed_pcm_or_interrupts() -> None:
    """Slow cloud ASR (250ms) runs off the audio thread so subsequent frames & interrupts stay instant."""
    cloud = _SlowCloudASR(delay_sec=0.25, text="帮我查一下上海天气")
    seg = np.ones(1600, dtype=np.float32) * 0.2
    pipeline = _make_stub_pipeline(cloud, segments=[seg], kws_hits=["", "退出"])

    received_queries: list[str] = []
    interrupt_called = asyncio.Event()

    async def on_query(q: str) -> str:
        received_queries.append(q)
        return "上海今天晴。"

    async def scenario() -> None:
        gw = SpeakerGateway(AppConfig(), on_query=on_query)
        gw._pipeline = pipeline
        conn = _DummyConn()
        gw._connections.add(conn)

        orig_interrupt = gw._handle_interrupt

        async def wrapped_interrupt(c: Any) -> None:
            interrupt_called.set()
            await orig_interrupt(c)

        gw._handle_interrupt = wrapped_interrupt  # type: ignore[method-assign]

        # Build a valid open-xiaoai binary record frame (512 samples of int16).
        pcm_bytes = (np.zeros(512, dtype="<i2")).tobytes()
        frame = json.dumps({"tag": "record", "bytes": list(pcm_bytes)}).encode("utf-8")

        t0 = time.monotonic()
        await gw._handle_binary_message(conn, frame)
        first_frame_ms = (time.monotonic() - t0) * 1000.0
        # Must return well before the 250ms cloud ASR finishes and never call sync transcribe()
        assert first_frame_ms < 120.0
        assert cloud.sync_called is False

        # Second frame triggers KWS interrupt immediately while cloud ASR is still in flight!
        t1 = time.monotonic()
        await gw._handle_binary_message(conn, frame)
        second_frame_ms = (time.monotonic() - t1) * 1000.0
        assert second_frame_ms < 120.0
        await asyncio.wait_for(interrupt_called.wait(), timeout=0.1)

        # Wait for in-flight cloud ASR task to complete cleanly.
        sess = gw._get_session(conn)
        if sess.cloud_asr_tasks:
            await asyncio.gather(*list(sess.cloud_asr_tasks), return_exceptions=True)
        assert cloud.async_called == 1

    asyncio.run(scenario())


def test_pluggable_web_search_tavily_searxng_and_bing_captcha_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    assert resolve_search_provider(SearchToolConfig()) == "bing"

    # 1. Tavily JSON provider selected automatically when TAVILY_API_KEY is present
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    assert resolve_search_provider(SearchToolConfig()) == "tavily"

    def tavily_handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.tavily.com/search"
        body = json.loads(request.content.decode("utf-8"))
        assert body["api_key"] == "tvly-test-key"
        assert body["query"] == "Python 3.13"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Python 3.13 Release",
                        "content": "Free-threaded mode and JIT improvements.",
                        "url": "https://python.org/3.13",
                    }
                ]
            },
            request=request,
        )

    async def check_tavily() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(tavily_handler)) as client:
            res = await run_web_search("Python 3.13", limit=3, client=client)
        assert res["provider"] == "tavily"
        assert res["count"] == 1
        assert res["results"][0]["title"] == "Python 3.13 Release"

    asyncio.run(check_tavily())

    # 2. SearXNG JSON provider via config or env
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    searx_cfg = SearchToolConfig(provider="auto", searxng_url="http://127.0.0.1:8080")
    assert resolve_search_provider(searx_cfg) == "searxng"

    def searx_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params.get("format") == "json"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "SearXNG Result",
                        "content": "Privacy respecting search.",
                        "url": "https://example.org/searx",
                    }
                ]
            },
            request=request,
        )

    async def check_searx() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(searx_handler)) as client:
            cfg = AppConfig()
            cfg.tools.search = searx_cfg
            server = McpServer(cfg, include_local_tools=True, http_client=client)
            resp = await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "web_search", "arguments": {"query": "test", "limit": 2}},
                }
            )
            assert resp is not None
            assert resp["result"]["isError"] is False
            data = json.loads(resp["result"]["content"][0]["text"])
            assert data["provider"] == "searxng"
            assert data["count"] == 1

    asyncio.run(check_searx())

    # 3. Bing HTML fallback surfaces explicit warning on anti-bot captcha page
    def bing_captcha_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html><body>Please complete CAPTCHA to continue (unusual traffic).</body></html>",
            request=request,
        )

    async def check_bing_captcha() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(bing_captcha_handler)) as client:
            res = await run_web_search("weather", config=SearchToolConfig(provider="bing"), client=client)
        assert res["provider"] == "bing"
        assert res["count"] == 0
        assert "bing_captcha_or_blocked" in res["warning"]

    asyncio.run(check_bing_captcha())


def test_cross_platform_browser_command_and_tts_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert build_edge_open_command("https://example.com", platform="darwin") == [
        "open",
        "-a",
        "Microsoft Edge",
        "https://example.com",
    ]
    assert build_edge_open_command("https://example.com", platform="win32") == [
        "cmd",
        "/c",
        "start",
        "",
        "https://example.com",
    ]

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None)
    assert build_edge_open_command("https://example.com", platform="linux") == [
        "xdg-open",
        "https://example.com",
    ]

    # When neither `say` nor `espeak-ng`/`espeak` exists, raise clear cross-platform message
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="espeak-ng"):
        asyncio.run(synthesize_local_fallback_pcm("你好"))

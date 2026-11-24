"""Serialized playback controller managing open-xiaoai-client `start_play`/`stop_play` PCM streams."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from xiao_speaker_assistant.config import SpeakerConfig
from xiao_speaker_assistant.voice.tts_stream import generate_chime_pcm, synthesize_tts_pcm_stream

logger = logging.getLogger(__name__)

SendTextFn = Callable[[str], Awaitable[None]]
SendBinaryFn = Callable[[bytes], Awaitable[None]]
RunShellFn = Callable[[str, bool, float], Awaitable[dict[str, Any] | None]]


class PlaybackController:
    """Owns serialized speaker playback, eliminating boolean races and per-sentence file delays."""

    def __init__(
        self,
        config: SpeakerConfig,
        *,
        send_text: SendTextFn,
        send_binary: SendBinaryFn,
        run_shell: RunShellFn,
        on_playback_state: Callable[[bool], None] | None = None,
    ) -> None:
        self._config = config
        self._send_text = send_text
        self._send_binary = send_binary
        self._run_shell = run_shell
        self._on_playback_state = on_playback_state
        self._lock = asyncio.Lock()
        self._stream_open = False
        self._cancel_event = asyncio.Event()
        self._chime_pcm = generate_chime_pcm(sample_rate=config.tts_sample_rate)

    @property
    def is_playing(self) -> bool:
        return self._lock.locked()

    async def _send_rpc(self, command: str, payload: Any = None) -> None:
        req = {
            "Request": {
                "id": str(uuid.uuid4()),
                "command": command,
                "payload": payload,
            }
        }
        await self._send_text(json.dumps(req, ensure_ascii=False))

    async def ensure_stream_started(self) -> None:
        if self._stream_open:
            return
        await self._send_rpc(
            "start_play",
            {
                "pcm": "default",
                "channels": 1,
                "bits_per_sample": 16,
                "sample_rate": self._config.tts_sample_rate,
                "period_size": 960,
                "buffer_size": 3840,
            },
        )
        self._stream_open = True

    async def stop_playback(self, *, stop_native: bool = True) -> None:
        """Immediately halt active PCM stream and optionally stop native players."""
        self._cancel_event.set()
        if self._stream_open:
            self._stream_open = False
            try:
                await self._send_rpc("stop_play", None)
            except Exception:
                pass
        if stop_native:
            try:
                await self._run_shell(
                    "killall miplayer tts_play.sh >/dev/null 2>&1 || true; mphelper pause >/dev/null 2>&1 || true",
                    False,
                    5.0,
                )
            except Exception:
                pass

    async def _send_pcm_chunks(self, pcm: bytes, *, chunk_bytes: int = 4800) -> None:
        bytes_per_sec = self._config.tts_sample_rate * 2
        for i in range(0, len(pcm), chunk_bytes):
            if self._cancel_event.is_set():
                return
            piece = pcm[i : i + chunk_bytes]
            frame = json.dumps(
                {"id": str(uuid.uuid4()), "tag": "play", "bytes": list(piece)}
            ).encode("utf-8")
            await self._send_binary(frame)
            # Pace slightly faster than real-time so the 50-slot client queue stays smoothly fed.
            await asyncio.sleep(max(0.01, (len(piece) / bytes_per_sec) * 0.85))

    async def play_chime(self) -> None:
        async with self._lock:
            self._cancel_event.clear()
            if self._on_playback_state:
                self._on_playback_state(True)
            try:
                await self.ensure_stream_started()
                await self._send_pcm_chunks(self._chime_pcm)
            finally:
                if self._on_playback_state:
                    self._on_playback_state(False)

    async def speak(self, text: str, *, interrupt_native: bool = True) -> None:
        if not text.strip():
            return
        async with self._lock:
            self._cancel_event.clear()
            if self._on_playback_state:
                self._on_playback_state(True)
            try:
                if self._config.tts_provider == "xiaoai":
                    quoted = shlex.quote(text)
                    prefix = (
                        "killall miplayer tts_play.sh >/dev/null 2>&1 || true; "
                        "mphelper pause >/dev/null 2>&1 || true; "
                        if interrupt_native
                        else ""
                    )
                    await self._run_shell(f"{prefix}/usr/sbin/tts_play.sh -n {quoted}", True, 25.0)
                    return

                if interrupt_native:
                    await self._run_shell(
                        "killall miplayer tts_play.sh >/dev/null 2>&1 || true; mphelper pause >/dev/null 2>&1 || true",
                        False,
                        5.0,
                    )
                await self.ensure_stream_started()
                bytes_per_sec = self._config.tts_sample_rate * 2
                async for chunk in synthesize_tts_pcm_stream(text, self._config):
                    if self._cancel_event.is_set():
                        return
                    frame = json.dumps(
                        {"id": str(uuid.uuid4()), "tag": "play", "bytes": list(chunk)}
                    ).encode("utf-8")
                    await self._send_binary(frame)
                    await asyncio.sleep(max(0.01, (len(chunk) / bytes_per_sec) * 0.85))
            finally:
                if self._on_playback_state:
                    self._on_playback_state(False)

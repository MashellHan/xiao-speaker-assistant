"""Serialized playback controller managing open-xiaoai-client `start_play`/`stop_play` PCM streams."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from xiao_speaker_assistant.config import SpeakerConfig
from xiao_speaker_assistant.voice.tts_stream import generate_chime_pcm, synthesize_tts_pcm_stream

logger = logging.getLogger(__name__)

SendTextFn = Callable[[str], Awaitable[None]]
SendBinaryFn = Callable[[bytes], Awaitable[None]]
RunShellFn = Callable[[str, bool, float], Awaitable[dict[str, Any] | None]]
SendRpcFn = Callable[[str, Any, bool, float], Awaitable[dict[str, Any] | None]]

STOP_NATIVE_SCRIPT = (
    "ubus -t 2 call mediaplayer player_wakeup '{\"action\":\"stop\"}' >/dev/null 2>&1 || true; "
    "killall miplayer tts_play.sh >/dev/null 2>&1 || true; "
    "mphelper pause >/dev/null 2>&1 || true"
)


class PlaybackController:
    """Owns serialized speaker playback, eliminating races, premature unmute, and queue overflows."""

    def __init__(
        self,
        config: SpeakerConfig,
        *,
        send_text: SendTextFn,
        send_binary: SendBinaryFn,
        run_shell: RunShellFn,
        send_rpc: SendRpcFn | None = None,
        on_playback_state: Callable[[bool], None] | None = None,
    ) -> None:
        self._config = config
        self._send_text = send_text
        self._send_binary = send_binary
        self._run_shell = run_shell
        self._send_rpc_fn = send_rpc
        self._on_playback_state = on_playback_state
        self._lock = asyncio.Lock()
        self._stream_open = False
        self._cancel_event = asyncio.Event()
        self._cancel_epoch = 0
        self._play_drain_until = 0.0
        self._chime_pcm = generate_chime_pcm(sample_rate=config.tts_sample_rate)

    @property
    def is_playing(self) -> bool:
        return self._lock.locked()

    async def _send_rpc(
        self,
        command: str,
        payload: Any = None,
        *,
        wait_response: bool = False,
        timeout: float = 5.0,
    ) -> dict[str, Any] | None:
        if self._send_rpc_fn is not None:
            return await self._send_rpc_fn(command, payload, wait_response, timeout)
        req = {
            "Request": {
                "id": str(uuid.uuid4()),
                "command": command,
                "payload": payload,
            }
        }
        await self._send_text(json.dumps(req, ensure_ascii=False))
        return None

    async def ensure_stream_started(self, *, epoch: int | None = None) -> bool:
        if self._stream_open:
            return epoch is None or epoch == self._cancel_epoch
        # Mark stream open before awaiting the start_play Response so a mid-flight stop_playback()
        # still sends `stop_play` to tear down `aplay` if it finished spawning on the speaker.
        self._stream_open = True
        try:
            # Wait for open-xiaoai-client to spawn `aplay` and initialize its `AudioPlayer.sender`
            # channel before sending binary `tag="play"` frames (prevents silent initial drops).
            resp = await self._send_rpc(
                "start_play",
                {
                    "pcm": "default",
                    "channels": 1,
                    "bits_per_sample": 16,
                    "sample_rate": self._config.tts_sample_rate,
                    "period_size": 1200,
                    "buffer_size": 4800,  # 200ms ring buffer at 24kHz matching open-xiaoai-bridge
                },
                wait_response=True,
                timeout=5.0,
            )
            if self._send_rpc_fn is not None and resp is None:
                self._stream_open = False
                return False
        except asyncio.CancelledError:
            self._stream_open = False
            if self._cancel_event.is_set():
                return False
            raise
        if epoch is not None and (epoch != self._cancel_epoch or self._cancel_event.is_set()):
            return False
        return True

    async def _close_stream(self) -> None:
        self._play_drain_until = 0.0
        if self._stream_open:
            self._stream_open = False
            try:
                await self._send_rpc("stop_play", None, wait_response=True, timeout=1.5)
            except Exception:
                pass

    async def stop_playback(self, *, stop_native: bool = True) -> None:
        """Immediately halt active and queued playback and optionally stop native players."""
        self._cancel_epoch += 1
        self._cancel_event.set()
        await self._close_stream()
        if stop_native:
            try:
                await self._run_shell(STOP_NATIVE_SCRIPT, False, 5.0)
            except Exception:
                pass

    async def _wait_cancellable(self, seconds: float) -> bool:
        """Sleep up to `seconds`, returning False immediately if cancelled."""
        if seconds <= 0:
            return not self._cancel_event.is_set()
        try:
            await asyncio.wait_for(self._cancel_event.wait(), timeout=seconds)
            return False
        except asyncio.TimeoutError:
            return True

    async def _send_play_chunk(self, chunk: bytes, epoch: int) -> bool:
        if epoch != self._cancel_epoch or self._cancel_event.is_set():
            return False
        if not chunk:
            return True
        bytes_per_sec = self._config.tts_sample_rate * 2
        lead_buffer_sec = 0.30
        now = time.monotonic()
        # Keep at most ~300ms prebuffered ahead of real-time so the 50-slot queue never overflows.
        if self._play_drain_until > now + lead_buffer_sec:
            if not await self._wait_cancellable(self._play_drain_until - now - lead_buffer_sec):
                return False
            if epoch != self._cancel_epoch:
                return False
            now = time.monotonic()

        frame = json.dumps(
            {"id": str(uuid.uuid4()), "tag": "play", "bytes": list(chunk)}
        ).encode("utf-8")
        await self._send_binary(frame)
        chunk_dur = len(chunk) / bytes_per_sec
        self._play_drain_until = max(now, self._play_drain_until) + chunk_dur
        return True

    async def _stream_chunks(
        self,
        chunks: AsyncIterator[bytes],
        epoch: int,
        *,
        pad_tail_sec: float = 0.22,
    ) -> None:
        sent_any = False
        async for chunk in chunks:
            if chunk:
                sent_any = True
            if not await self._send_play_chunk(chunk, epoch):
                return

        # Pad trailing silence so ALSA `aplay` flushes the final audio past its start_threshold/buffer.
        if sent_any and pad_tail_sec > 0 and epoch == self._cancel_epoch and not self._cancel_event.is_set():
            pad_bytes = int(self._config.tts_sample_rate * pad_tail_sec) * 2
            chunk_bytes = 4800
            for i in range(0, pad_bytes, chunk_bytes):
                silence = b"\x00" * min(chunk_bytes, pad_bytes - i)
                if not await self._send_play_chunk(silence, epoch):
                    return

        # Wait until the speaker actually finishes draining the prebuffered tail before unmuting.
        remaining = self._play_drain_until - time.monotonic()
        if remaining > 0 and epoch == self._cancel_epoch:
            await self._wait_cancellable(remaining)

    async def play_chime(self) -> None:
        """Play crisp wakeup tone without resetting VAD or erasing one-shot follow-up speech."""
        epoch = self._cancel_epoch
        async with self._lock:
            if epoch != self._cancel_epoch:
                return
            self._cancel_event.clear()
            # Ensure native XiaoAI audio (if active) stops cleanly before Nomi chimes in.
            await self._run_shell(STOP_NATIVE_SCRIPT, False, 3.0)
            if epoch != self._cancel_epoch or self._cancel_event.is_set():
                return
            if not await self.ensure_stream_started(epoch=epoch):
                return

            async def chime_iter() -> AsyncIterator[bytes]:
                chunk_bytes = 4800
                for i in range(0, len(self._chime_pcm), chunk_bytes):
                    yield self._chime_pcm[i : i + chunk_bytes]

            # 150ms chime + 250ms trailing silence = 400ms (> 200ms ALSA buffer_size) so it always plays immediately.
            try:
                await self._stream_chunks(chime_iter(), epoch, pad_tail_sec=0.25)
            finally:
                await self._close_stream()

    async def speak(self, text: str, *, interrupt_native: bool = True) -> None:
        if not text.strip():
            return
        epoch = self._cancel_epoch
        async with self._lock:
            if epoch != self._cancel_epoch:
                return
            self._cancel_event.clear()
            if self._on_playback_state:
                self._on_playback_state(True)
            try:
                if self._config.tts_provider == "xiaoai":
                    quoted = shlex.quote(text)
                    prefix = f"{STOP_NATIVE_SCRIPT}; " if interrupt_native else ""
                    await self._run_shell(f"{prefix}/usr/sbin/tts_play.sh -n {quoted}", True, 25.0)
                    return

                if interrupt_native:
                    await self._run_shell(STOP_NATIVE_SCRIPT, False, 5.0)
                    if epoch != self._cancel_epoch or self._cancel_event.is_set():
                        return

                if not await self.ensure_stream_started(epoch=epoch):
                    return

                await self._stream_chunks(
                    synthesize_tts_pcm_stream(text, self._config),
                    epoch,
                    pad_tail_sec=0.22,
                )
            finally:
                if self._config.tts_provider != "xiaoai":
                    await self._close_stream()
                aborted = epoch != self._cancel_epoch or self._cancel_event.is_set()
                if self._on_playback_state and not aborted:
                    self._on_playback_state(False)

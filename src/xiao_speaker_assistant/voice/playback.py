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
        wan_relay: bool = False,
    ) -> None:
        self._config = config
        self._send_text = send_text
        self._send_binary = send_binary
        self._run_shell = run_shell
        self._send_rpc_fn = send_rpc
        self._on_playback_state = on_playback_state
        self._wan_relay = wan_relay
        self._lock = asyncio.Lock()
        self._stream_open = False
        self._playback_notified = False
        self._stream_sent_sec = 0.0
        self._cancel_event = asyncio.Event()
        self._cancel_epoch = 0
        self._play_drain_until = 0.0
        self._chime_pcm = generate_chime_pcm(sample_rate=config.tts_sample_rate)

    @property
    def is_playing(self) -> bool:
        return self._lock.locked() or self._stream_open

    @property
    def stream_open(self) -> bool:
        return self._stream_open

    def _alsa_buffer_frames(self) -> int:
        return 19200 if self._wan_relay else 4800

    def _notify_playback_started(self) -> None:
        if self._on_playback_state and not self._playback_notified:
            self._playback_notified = True
            self._on_playback_state(True)

    def _notify_playback_stopped(self) -> None:
        if self._on_playback_state and self._playback_notified:
            self._playback_notified = False
            self._on_playback_state(False)

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
        self._stream_sent_sec = 0.0
        period_size = 2400 if self._wan_relay else 1200
        buffer_size = self._alsa_buffer_frames()
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
                    "period_size": period_size,
                    "buffer_size": buffer_size,  # 800ms WAN jitter buffer / 200ms LAN buffer at 24kHz
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
        self._stream_sent_sec = 0.0
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
        self._playback_notified = False
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
        # Over WAN/SSH relay, prebuffer ~1.2s of PCM ahead of real-time (well below the 50-slot
        # open-xiaoai-client mpsc queue when using 200ms chunks) to absorb internet jitter.
        lead_buffer_sec = 1.20 if self._wan_relay else 0.30
        now = time.monotonic()
        if self._play_drain_until > now + lead_buffer_sec:
            if not await self._wait_cancellable(self._play_drain_until - now - lead_buffer_sec):
                return False
            if epoch != self._cancel_epoch:
                return False
            now = time.monotonic()

        frame = json.dumps(
            {"id": str(uuid.uuid4()), "tag": "play", "bytes": list(chunk)},
            separators=(",", ":"),
        ).encode("utf-8")
        await self._send_binary(frame)
        chunk_dur = len(chunk) / bytes_per_sec
        self._stream_sent_sec += chunk_dur
        self._play_drain_until = max(now, self._play_drain_until) + chunk_dur
        return True

    def _effective_tail_pad_sec(self, base_pad_sec: float) -> float:
        """Ensure total written PCM exceeds ALSA buffer_size (start_threshold) so short audio never drops."""
        min_total_sec = (self._alsa_buffer_frames() / self._config.tts_sample_rate) + 0.05
        min_pad_for_start = max(0.0, min_total_sec - self._stream_sent_sec)
        return max(base_pad_sec, min_pad_for_start)

    async def _flush_tail_and_wait(self, epoch: int, *, pad_tail_sec: float) -> None:
        effective_pad = self._effective_tail_pad_sec(pad_tail_sec)
        if effective_pad > 0 and epoch == self._cancel_epoch and not self._cancel_event.is_set():
            pad_bytes = int(self._config.tts_sample_rate * effective_pad) * 2
            chunk_bytes = 9600 if self._wan_relay else 4800
            for i in range(0, pad_bytes, chunk_bytes):
                silence = b"\x00" * min(chunk_bytes, pad_bytes - i)
                if not await self._send_play_chunk(silence, epoch):
                    return
        remaining = self._play_drain_until - time.monotonic()
        if remaining > 0 and epoch == self._cancel_epoch:
            await self._wait_cancellable(remaining)

    async def _stream_chunks(
        self,
        chunks: AsyncIterator[bytes],
        epoch: int,
        *,
        pad_tail_sec: float = 0.22,
        keep_open: bool = False,
    ) -> None:
        sent_any = False
        async for chunk in chunks:
            if chunk:
                sent_any = True
            if not await self._send_play_chunk(chunk, epoch):
                return

        if keep_open:
            # Multi-sentence streaming: keep `aplay` open and skip tail silence/drain sleep
            # so back-to-back sentences play seamlessly with zero underrun gap.
            return

        if sent_any:
            await self._flush_tail_and_wait(epoch, pad_tail_sec=pad_tail_sec)

    async def finish_stream(self, *, pad_tail_sec: float | None = None) -> None:
        """Flush trailing silence, wait for ALSA drain, and close a kept-open multi-sentence stream."""
        epoch = self._cancel_epoch
        async with self._lock:
            if epoch != self._cancel_epoch or self._cancel_event.is_set():
                return
            if not self._stream_open:
                self._notify_playback_stopped()
                return
            base_pad = (
                pad_tail_sec
                if pad_tail_sec is not None
                else (0.45 if self._wan_relay else 0.22)
            )
            try:
                if self._stream_sent_sec > 0:
                    await self._flush_tail_and_wait(epoch, pad_tail_sec=base_pad)
            finally:
                aborted = epoch != self._cancel_epoch or self._cancel_event.is_set()
                if not aborted:
                    await self._close_stream()
                    self._notify_playback_stopped()

    async def abort_stream(self, *, unmute: bool = True) -> None:
        """Close a kept-open stream immediately without padding/draining (e.g. after mid-stream error)."""
        epoch = self._cancel_epoch
        async with self._lock:
            if epoch != self._cancel_epoch or self._cancel_event.is_set():
                return
            if self._stream_open:
                await self._close_stream()
            if unmute:
                self._notify_playback_stopped()

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

            chunk_bytes = 9600 if self._wan_relay else 4800

            async def chime_iter() -> AsyncIterator[bytes]:
                for i in range(0, len(self._chime_pcm), chunk_bytes):
                    yield self._chime_pcm[i : i + chunk_bytes]

            # Pad trailing silence beyond ALSA buffer_size so short chime always flushes immediately.
            pad_sec = 0.70 if self._wan_relay else 0.25
            try:
                await self._stream_chunks(chime_iter(), epoch, pad_tail_sec=pad_sec)
            finally:
                await self._close_stream()

    async def speak(
        self,
        text: str,
        *,
        interrupt_native: bool = True,
        voice_override: str | None = None,
        keep_stream_open: bool = False,
    ) -> None:
        if not text.strip():
            return
        epoch = self._cancel_epoch
        async with self._lock:
            if epoch != self._cancel_epoch:
                return
            self._cancel_event.clear()
            self._notify_playback_started()
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

                chunk_bytes = 9600 if self._wan_relay else 4800
                tts_stream = (
                    synthesize_tts_pcm_stream(
                        text,
                        self._config,
                        voice_override=voice_override,
                        chunk_bytes=chunk_bytes,
                    )
                    if (voice_override or self._wan_relay)
                    else synthesize_tts_pcm_stream(text, self._config)
                )
                await self._stream_chunks(
                    tts_stream,
                    epoch,
                    pad_tail_sec=0.45 if self._wan_relay else 0.22,
                    keep_open=keep_stream_open,
                )
            finally:
                aborted = epoch != self._cancel_epoch or self._cancel_event.is_set()
                if self._config.tts_provider != "xiaoai" and (not keep_stream_open or aborted):
                    await self._close_stream()
                if not aborted and (not keep_stream_open or not self._stream_open):
                    self._notify_playback_stopped()

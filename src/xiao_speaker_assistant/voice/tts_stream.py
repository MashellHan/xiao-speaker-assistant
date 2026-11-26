"""Streaming PCM TTS synthesizers and non-XiaoAI chime generator."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from xiao_speaker_assistant.config import SpeakerConfig

logger = logging.getLogger(__name__)


def generate_chime_pcm(sample_rate: int = 24000) -> bytes:
    """Synthesize a crisp 150ms two-tone marimba chime (distinct from XiaoAI's voice)."""
    dur1 = int(sample_rate * 0.065)
    dur2 = int(sample_rate * 0.085)
    t1 = np.arange(dur1, dtype=np.float32) / sample_rate
    t2 = np.arange(dur2, dtype=np.float32) / sample_rate

    env1 = np.exp(-t1 * 38.0) * np.minimum(1.0, t1 / 0.004)
    env2 = np.exp(-t2 * 30.0) * np.minimum(1.0, t2 / 0.004)

    # C6 (1046.5 Hz) -> G6 (1567.98 Hz) gentle bell tone
    note1 = 0.28 * np.sin(2.0 * np.pi * 1046.5 * t1) * env1
    note2 = 0.30 * np.sin(2.0 * np.pi * 1567.98 * t2) * env2
    wave = np.concatenate([note1, note2])
    pcm16 = np.clip(wave * 32767.0, -32768, 32767).astype("<i2")
    return pcm16.tobytes()


async def decode_audio_bytes_to_pcm(
    audio_bytes: bytes,
    *,
    input_format: str = "mp3",
    sample_rate: int = 24000,
) -> bytes:
    """Decode encoded audio (e.g. MP3) to mono S16_LE PCM using ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for streaming TTS decoding")
    proc = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        input_format,
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(audio_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {stderr.decode(errors='replace').strip()}")
    return stdout


async def synthesize_say_pcm(
    text: str,
    *,
    voice: str = "Tingting",
    sample_rate: int = 24000,
) -> bytes:
    """Offline fallback synthesizer using macOS `say`."""
    say_bin = shutil.which("say")
    if not say_bin:
        raise RuntimeError("macOS `say` binary not found")
    fd, wav_path = tempfile.mkstemp(prefix="xsa-say-", suffix=".wav")
    os.close(fd)
    try:
        proc = await asyncio.create_subprocess_exec(
            say_bin,
            "-v",
            voice,
            "-o",
            wav_path,
            f"--data-format=LEI16@{sample_rate}",
            text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"`say` failed: {stderr.decode(errors='replace').strip()}")
        raw = Path(wav_path).read_bytes()
        # Strip standard 44-byte RIFF WAV header or decode via ffmpeg if non-standard.
        if raw[:4] == b"RIFF" and b"data" in raw[:128]:
            idx = raw.index(b"data") + 8
            return raw[idx:]
        return await decode_audio_bytes_to_pcm(raw, input_format="wav", sample_rate=sample_rate)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


async def stream_edge_tts_pcm(
    text: str,
    *,
    voice: str = "zh-CN-XiaoxiaoNeural",
    rate: str = "+0%",
    sample_rate: int = 24000,
    chunk_bytes: int = 4800,
) -> AsyncIterator[bytes]:
    """Stream Edge TTS MP3 chunks through an incremental ffmpeg pipe to S16_LE PCM."""
    import edge_tts

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for edge-tts streaming")

    proc = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "mp3",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    async def feed_mp3() -> None:
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate)
            async for item in comm.stream():
                if item.get("type") == "audio":
                    data = item.get("data")
                    if data:
                        proc.stdin.write(data)  # type: ignore[union-attr]
                        await proc.stdin.drain()  # type: ignore[union-attr]
        finally:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass

    feeder = asyncio.create_task(feed_mp3())
    buf = bytearray()
    try:
        while True:
            out = await proc.stdout.read(chunk_bytes)
            if not out:
                break
            buf.extend(out)
            while len(buf) >= chunk_bytes:
                piece = bytes(buf[:chunk_bytes])
                del buf[:chunk_bytes]
                yield piece
        if buf:
            if len(buf) % 2 == 1:
                buf.append(0)
            yield bytes(buf)
        await feeder
        await proc.wait()
        if proc.returncode != 0:
            err = (await proc.stderr.read()).decode(errors="replace").strip() if proc.stderr else ""
            raise RuntimeError(f"edge-tts ffmpeg failed: {err}")
    finally:
        if not feeder.done():
            feeder.cancel()
            try:
                await feeder
            except BaseException:
                pass
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()


async def synthesize_tts_pcm_stream(
    text: str,
    config: SpeakerConfig,
    *,
    voice_override: str | None = None,
    chunk_bytes: int = 4800,
    sample_rate_override: int | None = None,
) -> AsyncIterator[bytes]:
    """Stream PCM chunks using configured TTS provider with offline macOS `say` fallback."""
    effective_sr = sample_rate_override or config.tts_sample_rate
    if config.tts_provider == "edge":
        try:
            got_any = False
            async for chunk in stream_edge_tts_pcm(
                text,
                voice=voice_override or config.edge_voice,
                rate=config.edge_rate,
                sample_rate=effective_sr,
                chunk_bytes=chunk_bytes,
            ):
                got_any = True
                yield chunk
            if got_any:
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("edge-tts failed (%s); falling back to macOS `say`", exc)

    pcm = await synthesize_say_pcm(
        text,
        voice=config.say_voice,
        sample_rate=effective_sr,
    )
    for i in range(0, len(pcm), chunk_bytes):
        yield pcm[i : i + chunk_bytes]

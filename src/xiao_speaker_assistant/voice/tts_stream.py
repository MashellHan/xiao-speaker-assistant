"""Streaming PCM TTS synthesizers and non-XiaoAI chime generator."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
import time
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from xiao_speaker_assistant.config import SpeakerConfig

logger = logging.getLogger(__name__)

# Low-latency ffmpeg input flags cut MP3 probe/buffer delay from ~150-300ms down to ~5ms.
_FFMPEG_MP3_LOW_LATENCY_ARGS: tuple[str, ...] = (
    "-probesize",
    "32",
    "-analyzeduration",
    "0",
    "-fflags",
    "nobuffer",
    "-flags",
    "low_delay",
)


class PrewarmedEdgeTTS:
    """Pre-opened Edge-TTS WebSocket + speech.config ready to synthesize immediately on first LLM sentence.

    Opening the TLS + WSS connection to Microsoft Speech takes ~550-750ms from China.
    By starting `PrewarmedEdgeTTS.open()` in parallel with LLM generation, sending SSML
    produces the first PCM chunk in ~150-350ms instead of ~1200-1800ms.
    """

    def __init__(self, session: object, ws: object, *, created_at: float) -> None:
        self._session = session
        self._ws = ws
        self.created_at = created_at
        self._claimed = False
        self._closed = False

    @classmethod
    async def open(cls, *, connect_timeout: float = 8.0) -> PrewarmedEdgeTTS:
        import aiohttp
        from edge_tts.communicate import connect_id, date_to_string
        from edge_tts.constants import SEC_MS_GEC_VERSION, WSS_HEADERS, WSS_URL
        from edge_tts.drm import DRM
        import edge_tts.communicate as edge_comm

        ssl_ctx = getattr(edge_comm, "_SSL_CTX", None)
        timeout = aiohttp.ClientTimeout(total=60.0, sock_connect=connect_timeout, sock_read=30.0)
        session = aiohttp.ClientSession(trust_env=True, timeout=timeout)
        ws = None
        try:
            url = (
                f"{WSS_URL}&ConnectionId={connect_id()}"
                f"&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
                f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}"
            )
            headers = {**WSS_HEADERS, "Cookie": f"muid={DRM.generate_muid()};"}
            ws = await session.ws_connect(url, headers=headers, compress=15, ssl=ssl_ctx)
            await ws.send_str(
                f"X-Timestamp:{date_to_string()}\r\n"
                "Content-Type:application/json; charset=utf-8\r\n"
                "Path:speech.config\r\n\r\n"
                '{"context":{"synthesis":{"audio":{"metadataoptions":'
                '{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"false"},'
                '"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}\r\n'
            )
            return cls(session, ws, created_at=time.monotonic())
        except BaseException:
            try:
                if ws is not None:
                    await asyncio.wait_for(ws.close(), timeout=1.0)
            except BaseException:
                pass
            finally:
                try:
                    await asyncio.wait_for(session.close(), timeout=1.0)
                except BaseException:
                    pass
            raise

    @property
    def usable(self) -> bool:
        if self._claimed or self._closed:
            return False
        if (time.monotonic() - self.created_at) > 25.0:
            return False
        return not bool(getattr(self._ws, "closed", True))

    def claim(self) -> bool:
        if not self.usable:
            return False
        self._claimed = True
        return True

    async def stream_mp3(
        self,
        text: str,
        *,
        voice: str,
        rate: str,
    ) -> AsyncIterator[bytes]:
        import aiohttp
        from edge_tts.communicate import (
            connect_id,
            date_to_string,
            mkssml,
            remove_incompatible_characters,
            ssml_headers_plus_data,
        )
        from edge_tts.data_classes import TTSConfig

        try:
            ws = self._ws
            tts_cfg = TTSConfig(
                voice=voice,
                rate=rate,
                volume="+0%",
                pitch="+0Hz",
                boundary="SentenceBoundary",
            )
            cleaned = xml_escape(remove_incompatible_characters(text))
            ssml = mkssml(tts_cfg, cleaned)
            await ws.send_str(ssml_headers_plus_data(connect_id(), date_to_string(), ssml))  # type: ignore[attr-defined]
            async for msg in ws:  # type: ignore[attr-defined]
                if msg.type == aiohttp.WSMsgType.BINARY and len(msg.data) >= 2:
                    hdr_len = int.from_bytes(msg.data[:2], "big")
                    if 2 + hdr_len > len(msg.data):
                        continue
                    header_bytes = msg.data[2 : 2 + hdr_len]
                    if not any(line.strip() == b"Path:audio" for line in header_bytes.splitlines()):
                        continue
                    audio = msg.data[2 + hdr_len :]
                    if audio:
                        yield audio
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    if "Path:turn.end" in msg.data:
                        break
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
        finally:
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.wait_for(self._ws.close(), timeout=1.0)  # type: ignore[attr-defined]
        except BaseException:
            pass
        finally:
            try:
                await asyncio.wait_for(self._session.close(), timeout=1.0)  # type: ignore[attr-defined]
            except BaseException:
                pass


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


async def synthesize_edge_mp3_bytes(
    text: str,
    *,
    voice: str = "zh-CN-XiaoxiaoNeural",
    rate: str = "+0%",
) -> bytes:
    """Synthesize text directly to MP3 bytes via Edge-TTS for browser audio preview."""
    import edge_tts

    comm = edge_tts.Communicate(text, voice, rate=rate)
    buf = bytearray()
    async for item in comm.stream():
        if item.get("type") == "audio":
            data = item.get("data")
            if data:
                buf.extend(data)
    if not buf:
        raise RuntimeError("Edge-TTS produced empty audio stream")
    return bytes(buf)


async def stream_edge_tts_pcm(
    text: str,
    *,
    voice: str = "zh-CN-XiaoxiaoNeural",
    rate: str = "+0%",
    sample_rate: int = 24000,
    chunk_bytes: int = 4800,
    prewarmed: PrewarmedEdgeTTS | None = None,
) -> AsyncIterator[bytes]:
    """Stream Edge TTS MP3 chunks through a low-latency incremental ffmpeg pipe to S16_LE PCM."""
    import edge_tts

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if prewarmed is not None:
            await prewarmed.close()
        raise RuntimeError("ffmpeg is required for edge-tts streaming")

    use_prewarmed = prewarmed is not None and prewarmed.claim()
    try:
        proc = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            *_FFMPEG_MP3_LOW_LATENCY_ARGS,
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
    except BaseException:
        if prewarmed is not None:
            await prewarmed.close()
        raise
    assert proc.stdin is not None
    assert proc.stdout is not None

    async def feed_mp3() -> None:
        try:
            got_audio = False
            if use_prewarmed and prewarmed is not None:
                try:
                    async for data in prewarmed.stream_mp3(text, voice=voice, rate=rate):
                        if data:
                            got_audio = True
                            proc.stdin.write(data)  # type: ignore[union-attr]
                            await proc.stdin.drain()  # type: ignore[union-attr]
                except Exception as exc:
                    if got_audio:
                        raise
                    logger.debug("Prewarmed Edge-TTS WS failed before audio (%s); falling back", exc)
            if not got_audio:
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
        if use_prewarmed and prewarmed is not None:
            await prewarmed.close()
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()


async def synthesize_espeak_pcm(
    text: str,
    *,
    sample_rate: int = 24000,
    voice: str = "cmn",
) -> bytes:
    """Offline Linux/cross-platform fallback synthesizer using `espeak-ng` or `espeak`."""
    espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak_bin:
        raise RuntimeError("Neither macOS `say` nor Linux `espeak-ng`/`espeak` binary was found")
    # Pass text via `--stdin` so replies starting with `-` (e.g. `-5度`) are never parsed as flags.
    candidates = [voice]
    for fallback_voice in ("cmn", "zh", ""):
        if fallback_voice not in candidates:
            candidates.append(fallback_voice)
    last_err = ""
    for cand in candidates:
        argv = [espeak_bin]
        if cand:
            argv.extend(["-v", cand])
        argv.extend(["--stdout", "--stdin"])
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(text.encode("utf-8"))
        if proc.returncode == 0 and stdout:
            return await decode_audio_bytes_to_pcm(stdout, input_format="wav", sample_rate=sample_rate)
        last_err = stderr.decode(errors="replace").strip()
    raise RuntimeError(f"`{Path(espeak_bin).name}` failed: {last_err}")


async def synthesize_local_fallback_pcm(
    text: str,
    *,
    voice: str = "Tingting",
    sample_rate: int = 24000,
) -> bytes:
    """Cross-platform offline TTS fallback: macOS `say` -> Linux `espeak-ng`/`espeak`."""
    if shutil.which("say"):
        return await synthesize_say_pcm(text, voice=voice, sample_rate=sample_rate)
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return await synthesize_espeak_pcm(text, sample_rate=sample_rate)
    raise RuntimeError(
        "Offline TTS fallback unavailable: install `espeak-ng` on Linux or check Edge-TTS connectivity."
    )


async def synthesize_tts_pcm_stream(
    text: str,
    config: SpeakerConfig,
    *,
    voice_override: str | None = None,
    chunk_bytes: int = 4800,
    sample_rate_override: int | None = None,
    prewarmed: PrewarmedEdgeTTS | None = None,
) -> AsyncIterator[bytes]:
    """Stream PCM chunks using configured TTS provider with cross-platform offline fallback."""
    effective_sr = sample_rate_override or config.tts_sample_rate
    if config.tts_provider == "edge":
        try:
            got_any = False
            stream_kwargs: dict[str, object] = {
                "voice": voice_override or config.edge_voice,
                "rate": config.edge_rate,
                "sample_rate": effective_sr,
                "chunk_bytes": chunk_bytes,
            }
            if prewarmed is not None:
                stream_kwargs["prewarmed"] = prewarmed
            async for chunk in stream_edge_tts_pcm(text, **stream_kwargs):  # type: ignore[arg-type]
                got_any = True
                yield chunk
            if got_any:
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("edge-tts failed (%s); falling back to local offline TTS", exc)

    pcm = await synthesize_local_fallback_pcm(
        text,
        voice=config.say_voice,
        sample_rate=effective_sr,
    )
    for i in range(0, len(pcm), chunk_bytes):
        yield pcm[i : i + chunk_bytes]

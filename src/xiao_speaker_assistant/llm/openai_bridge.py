"""OpenAI-compatible HTTP bridge (`/v1/chat/completions`, `/v1/audio/speech`) for open-xiaoai-bridge."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

from aiohttp import web
import edge_tts

from xiao_speaker_assistant.config import SpeakerConfig
from xiao_speaker_assistant.llm.router import resolve_binary

logger = logging.getLogger(__name__)

AskFn = Callable[[str], Awaitable[str]]
StreamAskFn = Callable[[str], AsyncIterator[str]]
ResetFn = Callable[[], None]

ALLOWED_AUDIO_FORMATS = frozenset({"mp3", "wav", "pcm", "opus", "aac", "flac"})


def extract_last_user_prompt(messages: list[dict[str, Any]]) -> str:
    """Extract the latest user message text from an OpenAI chat `messages` list."""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = [
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            text = "".join(parts).strip()
        else:
            continue
        for marker in ("\n注意：将结果处理成纯文字版", "\n注意：这条消息是主人通过小爱音箱发送的"):
            if marker in text:
                text = text.split(marker, 1)[0].strip()
        if text:
            return text
    return ""


async def synthesize_edge_audio_bytes(
    text: str,
    *,
    voice: str = "zh-CN-XiaoxiaoNeural",
    response_format: str = "mp3",
) -> tuple[bytes, str]:
    """Synthesize speech using `edge-tts` and return `(audio_bytes, content_type)`."""
    fmt = response_format.strip().lower() or "mp3"
    if fmt not in ALLOWED_AUDIO_FORMATS:
        raise ValueError(f"Unsupported response_format: {response_format!r}")

    comm = edge_tts.Communicate(text=text, voice=voice)
    mp3_chunks: list[bytes] = []
    async for msg in comm.stream():
        if msg.get("type") == "audio":
            data = msg.get("data")
            if isinstance(data, (bytes, bytearray)):
                mp3_chunks.append(bytes(data))
    mp3_bytes = b"".join(mp3_chunks)
    if fmt == "mp3":
        return mp3_bytes, "audio/mpeg"

    ffmpeg_bin = resolve_binary("ffmpeg")
    out_fmt = "s16le" if fmt == "pcm" else fmt
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "mp3",
        "-i",
        "pipe:0",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-f",
        out_fmt,
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(input=mp3_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {err.decode(errors='replace')}")
    ctype = "audio/wav" if fmt == "wav" else "application/octet-stream"
    return out, ctype


class OpenAIBridgeServer:
    """Local OpenAI-compatible Chat + Speech server backed by `LLMRouter` and `edge-tts`."""

    def __init__(
        self,
        ask_fn: AskFn,
        speaker_config: SpeakerConfig,
        *,
        stream_fn: StreamAskFn | None = None,
        on_reset: ResetFn | None = None,
        auth_token: str | None = None,
    ) -> None:
        self._ask_fn = ask_fn
        self._stream_fn = stream_fn
        self._speaker_config = speaker_config
        self._on_reset = on_reset
        self._auth_token = auth_token.strip() if auth_token else None
        self._last_session_key: str | None = None

    def _check_security(self, request: web.Request) -> web.Response | None:
        if request.headers.get("Origin"):
            return web.json_response(
                {"error": {"message": "Cross-origin requests are not allowed"}},
                status=403,
            )
        if self._auth_token:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or not hmac.compare_digest(
                auth[7:].strip(), self._auth_token
            ):
                return web.json_response(
                    {"error": {"message": "Unauthorized"}},
                    status=401,
                )
        return None

    async def start(self, host: str = "127.0.0.1", port: int = 4395) -> tuple[web.AppRunner, int]:
        async def handle_health(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True})

        async def handle_chat(request: web.Request) -> web.StreamResponse:
            denied = self._check_security(request)
            if denied is not None:
                return denied
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

            session_key = request.headers.get("X-Hermes-Session-Key", "").strip() or None
            if (
                session_key is not None
                and self._last_session_key is not None
                and session_key != self._last_session_key
                and self._on_reset is not None
            ):
                self._on_reset()
            if session_key is not None:
                self._last_session_key = session_key

            messages = body.get("messages") or []
            prompt = extract_last_user_prompt(messages)
            if not prompt:
                return web.json_response({"error": {"message": "Missing user message"}}, status=400)

            model = str(body.get("model") or "nomi")
            cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            if bool(body.get("stream")):
                sse = web.StreamResponse(
                    status=200,
                    headers={
                        "Content-Type": "text/event-stream; charset=utf-8",
                        "Cache-Control": "no-cache",
                    },
                )
                await sse.prepare(request)

                async def write_chunk(delta: dict[str, Any], finish_reason: str | None = None) -> None:
                    chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                    }
                    await sse.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))

                await write_chunk({"role": "assistant"})
                if self._stream_fn is not None:
                    async for piece in self._stream_fn(prompt):
                        if piece:
                            await write_chunk({"content": piece})
                else:
                    reply = await self._ask_fn(prompt)
                    if reply:
                        await write_chunk({"content": reply})
                await write_chunk({}, finish_reason="stop")
                await sse.write(b"data: [DONE]\n\n")
                await sse.write_eof()
                return sse

            reply = await self._ask_fn(prompt)
            payload = {
                "id": cmpl_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
            }
            return web.json_response(payload)

        async def handle_speech(request: web.Request) -> web.Response:
            denied = self._check_security(request)
            if denied is not None:
                return denied
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

            text = str(body.get("input") or "").strip()
            if not text:
                return web.json_response({"error": {"message": "Missing input text"}}, status=400)
            voice = str(body.get("voice") or "").strip() or self._speaker_config.edge_voice
            if voice in {"alloy", "nova", "shimmer", "echo", "fable", "onyx", "default"}:
                voice = self._speaker_config.edge_voice
            fmt = str(body.get("response_format") or "mp3").strip().lower()
            if fmt not in ALLOWED_AUDIO_FORMATS:
                return web.json_response(
                    {"error": {"message": f"Unsupported response_format: {fmt!r}"}},
                    status=400,
                )

            audio_bytes, ctype = await synthesize_edge_audio_bytes(
                text, voice=voice, response_format=fmt
            )
            return web.Response(body=audio_bytes, content_type=ctype)

        app = web.Application()
        app.router.add_get("/health", handle_health)
        app.router.add_post("/v1/chat/completions", handle_chat)
        app.router.add_post("/v1/audio/speech", handle_speech)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        actual_port = port
        if site._server and site._server.sockets:  # type: ignore[attr-defined]
            actual_port = int(site._server.sockets[0].getsockname()[1])  # type: ignore[attr-defined]
        logger.info("OpenAI-compatible Nomi bridge listening on http://%s:%d/v1", host, actual_port)
        return runner, actual_port

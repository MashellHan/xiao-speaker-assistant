"""WebSocket server (:4399) speaking the exact open-xiaoai Rust client protocol."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
import uuid
from typing import Any, Awaitable, Callable

import websockets
from websockets.asyncio.server import ServerConnection

from xiao_speaker_assistant.config import AppConfig, read_env_secret

logger = logging.getLogger(__name__)
QueryHandler = Callable[[str], Awaitable[str]]

IGNORE_BARE_WORDS = {"小爱同学", "小爱"}


class SpeakerGateway:
    def __init__(self, config: AppConfig, on_query: QueryHandler) -> None:
        self._config = config
        self._on_query = on_query
        self._connections: set[ServerConnection] = set()
        self._seen_dialogs: set[str] = set()
        self._continuous_mode: bool = False
        self._query_lock = asyncio.Lock()
        self._mute_native_until: float = 0.0

    async def run_shell(self, conn: ServerConnection, script: str) -> None:
        cmd = {
            "Request": {
                "id": str(uuid.uuid4()),
                "command": "run_shell",
                "payload": script,
            }
        }
        await conn.send(json.dumps(cmd, ensure_ascii=False))

    async def stop_native_audio(self, conn: ServerConnection) -> None:
        await self.run_shell(
            conn,
            "killall miplayer tts_play.sh >/dev/null 2>&1 || true; mphelper pause >/dev/null 2>&1 || true",
        )

    async def abort_and_speak(self, conn: ServerConnection, text: str) -> None:
        quoted = shlex.quote(text)
        script = (
            "killall miplayer tts_play.sh >/dev/null 2>&1 || true; "
            "mphelper pause >/dev/null 2>&1 || true; "
            f"/usr/sbin/tts_play.sh {quoted}"
        )
        await self.run_shell(conn, script)

    async def send_tts(self, conn: ServerConnection, text: str) -> None:
        await self.abort_and_speak(conn, text)

    async def _handle_connection(self, conn: ServerConnection) -> None:
        expected_token = read_env_secret(self._config.speaker.auth_token_env)
        if expected_token:
            token = conn.request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if token != expected_token:
                await conn.close(code=4401, reason="Unauthorized")
                return

        self._connections.add(conn)
        logger.info("Speaker connected from %s", conn.remote_address)
        try:
            async for raw in conn:
                if isinstance(raw, bytes):
                    # Binary Stream frames from open-xiaoai-client (for local VAD/KWS/ASR).
                    continue
                await self._handle_text_message(conn, raw)
        finally:
            self._connections.discard(conn)
            logger.info("Speaker disconnected")

    async def _handle_text_message(self, conn: ServerConnection, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON message")
            return

        if "Response" in msg:
            logger.debug("Speaker command response: %s", msg["Response"])
            return

        if time.monotonic() < self._mute_native_until and self._is_native_speak_event(msg):
            logger.debug("Suppressing native XiaoAI Speak event")
            asyncio.create_task(self.stop_native_audio(conn))
            return

        extracted = self._extract_final_asr(msg)
        if not extracted:
            return
        dialog_id, text = extracted
        dedup_key = f"{dialog_id}:{text}"
        if dedup_key in self._seen_dialogs:
            return
        self._seen_dialogs.add(dedup_key)
        if len(self._seen_dialogs) > 200:
            self._seen_dialogs.clear()

        if self.should_ignore_utterance(text):
            logger.debug("Ignoring bare wake word: %s", text)
            return

        logger.info("ASR recognized: %s", text)
        asyncio.create_task(self._process_utterance(conn, text))

    @staticmethod
    def should_ignore_utterance(text: str) -> bool:
        return text.strip(" ，。！？,.!?") in IGNORE_BARE_WORDS

    @staticmethod
    def _is_native_speak_event(msg: dict[str, Any]) -> bool:
        evt = msg.get("Event")
        if not isinstance(evt, dict) or evt.get("event") != "instruction":
            return False
        data = evt.get("data")
        if not isinstance(data, dict):
            return False
        line = data.get("NewLine")
        if not isinstance(line, str):
            return False
        try:
            log_msg = json.loads(line)
        except json.JSONDecodeError:
            return False
        header = log_msg.get("header") or {}
        return header.get("namespace") == "SpeechSynthesizer" and header.get("name") == "Speak"

    async def _process_utterance(self, conn: ServerConnection, text: str) -> None:
        if any(word in text for word in ("退出对话", "结束对话", "退下吧")):
            self._continuous_mode = False
            self._mute_native_until = time.monotonic() + 6.0
            await self.abort_and_speak(conn, "好的，已退出对话。")
            return

        matched_kw = next((kw for kw in self._config.speech.wakeup_keywords if text.startswith(kw)), None)
        if not matched_kw and not self._continuous_mode:
            return

        self._mute_native_until = time.monotonic() + 10.0
        await self.stop_native_audio(conn)

        query = text
        if matched_kw:
            query = text[len(matched_kw) :].lstrip(" ，,：:")
            if not query:
                self._continuous_mode = True
                await self.abort_and_speak(conn, "我在，请讲。")
                return

        async with self._query_lock:
            wait_notice_task: asyncio.Task[None] | None = None
            try:
                async def delayed_notice() -> None:
                    await asyncio.sleep(3.5)
                    await self.abort_and_speak(conn, "正在思考，请稍等。")

                wait_notice_task = asyncio.create_task(delayed_notice())
                reply = await self._on_query(query)
                wait_notice_task.cancel()
                logger.info("LLM reply: %s", reply)
                self._mute_native_until = time.monotonic() + 6.0
                await self.abort_and_speak(conn, reply)
            except Exception as exc:
                if wait_notice_task:
                    wait_notice_task.cancel()
                logger.exception("Error handling query %r", query)
                await self.abort_and_speak(conn, f"处理出错了：{exc}")

    @staticmethod
    def _extract_final_asr(msg: dict[str, Any]) -> tuple[str, str] | None:
        evt = msg.get("Event")
        if not isinstance(evt, dict) or evt.get("event") != "instruction":
            return None
        data = evt.get("data")
        if not isinstance(data, dict):
            return None

        line = data.get("NewLine")
        if isinstance(line, str):
            try:
                log_msg = json.loads(line)
            except json.JSONDecodeError:
                return None
            header = log_msg.get("header") or {}
            payload = log_msg.get("payload") or {}
            if (
                header.get("namespace") == "SpeechRecognizer"
                and header.get("name") == "RecognizeResult"
                and payload.get("is_final") is True
            ):
                results = payload.get("results") or []
                if results and isinstance(results[0], dict):
                    text = (results[0].get("text") or "").strip()
                    dialog_id = str(header.get("dialog_id") or evt.get("id") or "")
                    if text:
                        return dialog_id, text
            return None

        text = (data.get("text") or data.get("asr_text") or "").strip()
        if text:
            return str(evt.get("id") or ""), text
        return None

    async def serve_forever(self) -> None:
        host = self._config.speaker.ws_host
        port = self._config.speaker.ws_port
        logger.info("Listening for speaker WebSocket on ws://%s:%s", host, port)
        async with websockets.serve(self._handle_connection, host, port):
            await asyncio.Future()

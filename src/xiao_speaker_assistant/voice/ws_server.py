"""WebSocket server (:4399) speaking open-xiaoai protocol with streaming sentence TTS."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

import websockets
from websockets.asyncio.server import ServerConnection

from xiao_speaker_assistant.config import AppConfig, read_env_secret
from xiao_speaker_assistant.voice.audio_pipeline import (
    LocalSpeechPipeline,
    extract_record_pcm_from_binary,
    match_wakeup_phrase,
)

logger = logging.getLogger(__name__)
QueryHandler = Callable[[str], Awaitable[str]]
StreamQueryHandler = Callable[[str], AsyncIterator[str]]

IGNORE_BARE_WORDS = {"小爱同学", "小爱"}
EXIT_EXACT_WORDS = {
    "退出",
    "退下",
    "再见",
    "拜拜",
    "停止",
    "关闭",
    "结束",
    "不用了",
    "闭嘴",
    "退出吧",
    "结束吧",
}
EXIT_PHRASES = ("退出对话", "结束对话", "停止对话", "退出模式", "结束模式", "退下吧")


class SpeakerGateway:
    def __init__(
        self,
        config: AppConfig,
        on_query: QueryHandler | None = None,
        on_stream_query: StreamQueryHandler | None = None,
    ) -> None:
        if on_query is None and on_stream_query is None:
            raise ValueError("Either on_query or on_stream_query must be provided")
        self._config = config
        self._on_query = on_query
        self._on_stream_query = on_stream_query
        self._connections: set[ServerConnection] = set()
        self._seen_dialogs: set[str] = set()
        self._continuous_mode: bool = False
        self._query_lock = asyncio.Lock()
        self._mute_native_until: float = 0.0
        self._pipeline: LocalSpeechPipeline | None = None
        self._pending_rpc: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def send_command(
        self,
        conn: ServerConnection,
        command: str,
        payload: Any = None,
        *,
        wait_response: bool = False,
        timeout: float = 20.0,
    ) -> dict[str, Any] | None:
        req_id = str(uuid.uuid4())
        fut: asyncio.Future[dict[str, Any]] | None = None
        if wait_response:
            fut = asyncio.get_running_loop().create_future()
            self._pending_rpc[req_id] = fut

        cmd = {
            "Request": {
                "id": req_id,
                "command": command,
                "payload": payload,
            }
        }
        await conn.send(json.dumps(cmd, ensure_ascii=False))
        if not wait_response or fut is None:
            return None
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_rpc.pop(req_id, None)

    async def run_shell(
        self, conn: ServerConnection, script: str, *, wait_response: bool = False, timeout: float = 25.0
    ) -> dict[str, Any] | None:
        return await self.send_command(
            conn, "run_shell", script, wait_response=wait_response, timeout=timeout
        )

    async def stop_native_audio(self, conn: ServerConnection) -> None:
        await self.run_shell(
            conn,
            "killall miplayer tts_play.sh >/dev/null 2>&1 || true; mphelper pause >/dev/null 2>&1 || true",
        )

    async def speak_sentence(
        self,
        conn: ServerConnection,
        text: str,
        *,
        interrupt: bool = True,
        wait_response: bool = True,
    ) -> None:
        if self._pipeline is not None:
            self._pipeline.set_muted(True)
        quoted = shlex.quote(text)
        prefix = (
            "killall miplayer tts_play.sh >/dev/null 2>&1 || true; "
            "mphelper pause >/dev/null 2>&1 || true; "
            if interrupt
            else ""
        )
        script = f"{prefix}/usr/sbin/tts_play.sh {quoted}"
        try:
            await self.run_shell(conn, script, wait_response=wait_response, timeout=25.0)
        finally:
            if self._pipeline is not None:
                self._pipeline.mute_for(0.4)
                self._pipeline.set_muted(False)

    async def abort_and_speak(self, conn: ServerConnection, text: str) -> None:
        await self.speak_sentence(conn, text, interrupt=True, wait_response=False)

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
        if self._pipeline is not None:
            bits = self._config.speech.capture_bits_per_sample
            logger.info(
                "Starting speaker audio stream (%d-bit) for local KWS (`%s`) + %s",
                bits,
                ", ".join(self._config.speech.local_keywords),
                self._config.speech.asr_model,
            )
            await self.send_command(
                conn,
                "start_recording",
                {
                    "pcm": "noop",
                    "channels": 1,
                    "bits_per_sample": bits,
                    "sample_rate": 16000,
                    "period_size": 160,
                    "buffer_size": 480,
                },
            )

        try:
            async for raw in conn:
                if isinstance(raw, bytes):
                    await self._handle_binary_message(conn, raw)
                    continue
                await self._handle_text_message(conn, raw)
        finally:
            self._connections.discard(conn)
            for fut in self._pending_rpc.values():
                if not fut.done():
                    fut.cancel()
            self._pending_rpc.clear()
            logger.info("Speaker disconnected")

    async def _handle_binary_message(self, conn: ServerConnection, raw: bytes) -> None:
        if self._pipeline is None:
            return
        pcm = extract_record_pcm_from_binary(raw)
        if not pcm:
            return
        loop = asyncio.get_running_loop()

        def on_wakeup(keyword: str) -> None:
            logger.info("Local wake word detected: %s", keyword)
            loop.create_task(self.speak_sentence(conn, "我在，请讲。", interrupt=True, wait_response=True))

        def on_utterance(text: str) -> None:
            logger.info("Local ASR recognized: %s", text)
            loop.create_task(self._process_local_utterance(conn, text))

        def on_sleep() -> None:
            logger.info("Returned silently to wake-word idle")

        self._pipeline.feed_pcm(
            pcm,
            on_wakeup=on_wakeup,
            on_utterance=on_utterance,
            on_sleep=on_sleep,
        )

    async def _process_local_utterance(self, conn: ServerConnection, text: str) -> None:
        if self.is_exit_command(text):
            if self._pipeline is not None:
                self._pipeline.set_awake(False)
            await self.speak_sentence(conn, "好的，已退出对话。", interrupt=True, wait_response=True)
            return
        matched = match_wakeup_phrase(text, self._config.speech.local_keywords)
        if matched:
            _, rest = matched
            if not rest:
                await self.speak_sentence(conn, "我在，请讲。", interrupt=True, wait_response=True)
                if self._pipeline is not None:
                    self._pipeline.refresh_activity()
                return
            text = rest
        await self._run_query_and_speak(conn, text)

    async def _handle_text_message(self, conn: ServerConnection, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON message")
            return

        if "Response" in msg:
            resp = msg["Response"]
            if isinstance(resp, dict):
                req_id = resp.get("id")
                fut = self._pending_rpc.get(req_id) if isinstance(req_id, str) else None
                if fut is not None and not fut.done():
                    fut.set_result(resp)
            logger.debug("Speaker command response: %s", resp)
            return

        if self._pipeline is not None:
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
    def is_exit_command(text: str) -> bool:
        cleaned = text.strip(" ，。！？,.!?")
        if cleaned in EXIT_EXACT_WORDS:
            return True
        return any(phrase in cleaned for phrase in EXIT_PHRASES)

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
        if self.is_exit_command(text):
            self._continuous_mode = False
            self._mute_native_until = time.monotonic() + 6.0
            await self.speak_sentence(conn, "好的，已退出对话。", interrupt=True, wait_response=True)
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
                await self.speak_sentence(conn, "我在，请讲。", interrupt=True, wait_response=True)
                return

        await self._run_query_and_speak(conn, query)

    async def _run_query_and_speak(self, conn: ServerConnection, query: str) -> None:
        async with self._query_lock:
            if self._on_stream_query is not None:
                await self._run_streaming_reply(conn, query)
                return
            assert self._on_query is not None
            wait_notice_task: asyncio.Task[None] | None = None
            try:
                async def delayed_notice() -> None:
                    await asyncio.sleep(3.5)
                    await self.speak_sentence(conn, "正在思考，请稍等。", interrupt=True, wait_response=True)

                wait_notice_task = asyncio.create_task(delayed_notice())
                reply = await self._on_query(query)
                wait_notice_task.cancel()
                logger.info("LLM reply: %s", reply)
                self._mute_native_until = time.monotonic() + 6.0
                await self.speak_sentence(conn, reply, interrupt=True, wait_response=True)
            except Exception as exc:
                if wait_notice_task:
                    wait_notice_task.cancel()
                logger.exception("Error handling query %r", query)
                await self.speak_sentence(conn, f"处理出错了：{exc}", interrupt=True, wait_response=True)
            finally:
                if self._pipeline is not None:
                    self._pipeline.refresh_activity()

    async def _run_streaming_reply(self, conn: ServerConnection, query: str) -> None:
        assert self._on_stream_query is not None
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        first_chunk_event = asyncio.Event()

        async def producer() -> None:
            try:
                async for sentence in self._on_stream_query(query):
                    if sentence:
                        first_chunk_event.set()
                        await queue.put(sentence)
            except Exception as exc:
                logger.exception("Streaming error for %r", query)
                first_chunk_event.set()
                await queue.put(f"处理出错了：{exc}")
            finally:
                await queue.put(None)

        async def delayed_notice() -> None:
            try:
                await asyncio.wait_for(first_chunk_event.wait(), timeout=3.5)
            except asyncio.TimeoutError:
                await self.speak_sentence(conn, "正在思考，请稍等。", interrupt=True, wait_response=True)

        prod_task = asyncio.create_task(producer())
        notice_task = asyncio.create_task(delayed_notice())
        first_spoken = True
        try:
            while True:
                sentence = await queue.get()
                if sentence is None:
                    break
                logger.info("Streaming TTS sentence: %s", sentence)
                self._mute_native_until = time.monotonic() + 6.0
                await self.speak_sentence(conn, sentence, interrupt=first_spoken, wait_response=True)
                first_spoken = False
            await prod_task
        finally:
            notice_task.cancel()
            if not prod_task.done():
                prod_task.cancel()
            if self._pipeline is not None:
                self._pipeline.refresh_activity()

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
        if self._config.speech.asr_provider == "sense_voice":
            logger.info("Loading local KWS + VAD + %s models...", self._config.speech.asr_model)
            self._pipeline = LocalSpeechPipeline(self._config.speech)
            logger.info("Local speech pipeline ready!")

        host = self._config.speaker.ws_host
        port = self._config.speaker.ws_port
        logger.info("Listening for speaker WebSocket on ws://%s:%s", host, port)
        async with websockets.serve(self._handle_connection, host, port):
            await asyncio.Future()

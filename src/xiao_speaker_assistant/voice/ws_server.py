"""WebSocket server (:4399) speaking open-xiaoai protocol with PlaybackController & per-connection sessions."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

import websockets
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from xiao_speaker_assistant.config import AppConfig, read_env_secret
from xiao_speaker_assistant.voice.audio_pipeline import (
    LocalSpeechPipeline,
    extract_record_pcm_from_binary,
    match_wakeup_phrase,
)
from xiao_speaker_assistant.voice.playback import PlaybackController

logger = logging.getLogger(__name__)
QueryHandler = Callable[[str], Awaitable[str]]
StreamQueryHandler = Callable[[str], AsyncIterator[str]]

IGNORE_BARE_WORDS = {"小爱同学", "小爱"}
FILLER_NOISE_WORDS = {"嗯", "啊", "哦", "呃", "额", "唉", "唔", "哈", "嘿", "喂"}
EXIT_EXACT_WORDS = {
    "退出",
    "退下",
    "退一下",
    "再见",
    "拜拜",
    "停止",
    "停下",
    "关闭",
    "结束",
    "不用了",
    "闭嘴",
    "算了",
    "退出吧",
    "结束吧",
}
EXIT_PHRASES = ("退出对话", "结束对话", "停止对话", "退出模式", "结束模式", "退下吧", "别说了")
SHORT_EXIT_SUBSTRINGS = ("退出", "退下", "退一下", "闭嘴", "停止", "停下", "别说了", "再见", "拜拜")


@dataclass
class SpeakerSession:
    """Per-connection mutable state isolated per speaker socket."""

    seen_dialogs: set[str] = field(default_factory=set)
    continuous_mode: bool = False
    query_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    mute_native_until: float = 0.0
    last_audio_at: float = 0.0
    active_reply_task: asyncio.Task[None] | None = None
    interrupted: bool = False


class SpeakerGateway:
    def __init__(
        self,
        config: AppConfig,
        on_query: QueryHandler | None = None,
        on_stream_query: StreamQueryHandler | None = None,
        on_session_reset: Callable[[], None] | None = None,
        *,
        audio_watchdog_interval: float = 3.0,
        audio_stall_timeout: float = 4.0,
    ) -> None:
        if on_query is None and on_stream_query is None:
            raise ValueError("Either on_query or on_stream_query must be provided")
        self._config = config
        self._on_query = on_query
        self._on_stream_query = on_stream_query
        self._on_session_reset = on_session_reset
        self._audio_watchdog_interval = audio_watchdog_interval
        self._audio_stall_timeout = audio_stall_timeout
        self._connections: set[ServerConnection] = set()
        self._sessions: dict[Any, SpeakerSession] = {}
        self._mute_native_until: float = 0.0
        self._pipeline: LocalSpeechPipeline | None = None
        self._pending_rpc: dict[Any, dict[str, asyncio.Future[dict[str, Any]]]] = {}
        self._playbacks: dict[Any, PlaybackController] = {}
        self._asr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xsa-asr")

    def ensure_pipeline(self) -> LocalSpeechPipeline | None:
        if self._config.speech.asr_provider == "sense_voice" and self._pipeline is None:
            logger.info("Loading local KWS + VAD + %s models...", self._config.speech.asr_model)
            self._pipeline = LocalSpeechPipeline(self._config.speech)
            logger.info("Local speech pipeline ready!")
        return self._pipeline

    def _get_session(self, conn: Any) -> SpeakerSession:
        sess = self._sessions.get(conn)
        if sess is None:
            sess = SpeakerSession()
            self._sessions[conn] = sess
        return sess

    def _get_playback(self, conn: Any) -> PlaybackController:
        pb = self._playbacks.get(conn)
        if pb is not None:
            return pb

        def on_state(playing: bool) -> None:
            if self._pipeline is None:
                return
            # Always keep VAD muted during playback to prevent nonlinear residual echo self-talk;
            # unified KWS still runs unmuted on the AEC-cleaned signal for low-volume barge-in.
            if playing:
                self._pipeline.set_muted(True)
            else:
                tail = (
                    0.15
                    if (self._config.speech.aec_enabled and self._config.speech.capture_channels >= 2)
                    else 0.25
                )
                self._pipeline.mute_for(tail)
                self._pipeline.set_muted(False)

        pb = PlaybackController(
            self._config.speaker,
            send_text=lambda text: conn.send(text),
            send_binary=lambda data: conn.send(data),
            run_shell=lambda script, wait, timeout: self.run_shell(
                conn, script, wait_response=wait, timeout=timeout
            ),
            send_rpc=lambda cmd, payload, wait, timeout: self.send_command(
                conn, cmd, payload, wait_response=wait, timeout=timeout
            ),
            on_playback_state=on_state,
        )
        self._playbacks[conn] = pb
        return pb

    def _cleanup_connection(self, conn: Any) -> None:
        self._connections.discard(conn)
        sess = self._sessions.pop(conn, None)
        if sess is not None and sess.active_reply_task is not None and not sess.active_reply_task.done():
            sess.interrupted = True
            sess.active_reply_task.cancel()
        self._playbacks.pop(conn, None)
        conn_rpcs = self._pending_rpc.pop(conn, {})
        for fut in conn_rpcs.values():
            if not fut.done():
                fut.cancel()

    async def send_command(
        self,
        conn: Any,
        command: str,
        payload: Any = None,
        *,
        wait_response: bool = False,
        timeout: float = 20.0,
    ) -> dict[str, Any] | None:
        req_id = str(uuid.uuid4())
        conn_rpcs = self._pending_rpc.setdefault(conn, {})
        fut: asyncio.Future[dict[str, Any]] | None = None
        if wait_response:
            fut = asyncio.get_running_loop().create_future()
            conn_rpcs[req_id] = fut

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
            conn_rpcs.pop(req_id, None)

    async def run_shell(
        self, conn: Any, script: str, *, wait_response: bool = False, timeout: float = 25.0
    ) -> dict[str, Any] | None:
        return await self.send_command(
            conn, "run_shell", script, wait_response=wait_response, timeout=timeout
        )

    async def stop_native_audio(self, conn: Any) -> None:
        await self.run_shell(
            conn,
            "ubus -t 2 call mediaplayer player_wakeup '{\"action\":\"stop\"}' >/dev/null 2>&1 || true; "
            "killall miplayer tts_play.sh >/dev/null 2>&1 || true; "
            "mphelper pause >/dev/null 2>&1 || true",
        )

    async def play_wakeup_chime(self, conn: Any) -> None:
        pb = self._get_playback(conn)
        await pb.play_chime()

    async def speak_sentence(
        self,
        conn: Any,
        text: str,
        *,
        interrupt: bool = True,
        wait_response: bool = True,
    ) -> None:
        pb = self._get_playback(conn)
        await pb.speak(text, interrupt_native=interrupt)

    async def abort_and_speak(self, conn: Any, text: str) -> None:
        await self.speak_sentence(conn, text, interrupt=True, wait_response=False)

    async def send_tts(self, conn: Any, text: str) -> None:
        await self.abort_and_speak(conn, text)

    async def start_recording(self, conn: Any) -> None:
        sess = self._get_session(conn)
        bits = self._config.speech.capture_bits_per_sample
        ch = self._config.speech.capture_channels
        pcm_dev = self._config.speech.capture_pcm
        logger.info(
            "Starting speaker audio stream (%s, %dch, %d-bit, aec=%s) for local KWS (`%s`) + %s",
            pcm_dev,
            ch,
            bits,
            self._config.speech.aec_enabled,
            ", ".join(self._config.speech.local_keywords),
            self._config.speech.asr_model,
        )
        await self.send_command(
            conn,
            "start_recording",
            {
                "pcm": pcm_dev,
                "channels": ch,
                "bits_per_sample": bits,
                "sample_rate": 16000,
                "period_size": 360,
                "buffer_size": 1440,
            },
        )
        sess.last_audio_at = time.monotonic()

    async def _audio_watchdog(self, conn: Any) -> None:
        sess = self._get_session(conn)
        while True:
            await asyncio.sleep(self._audio_watchdog_interval)
            if self._pipeline is None:
                return
            if (time.monotonic() - sess.last_audio_at) > self._audio_stall_timeout:
                logger.warning("Audio stream stalled >4s; restarting speaker recording")
                bits = self._config.speech.capture_bits_per_sample
                ch = self._config.speech.capture_channels
                pcm_dev = self._config.speech.capture_pcm
                await self.send_command(
                    conn,
                    "start_recording",
                    {
                        "pcm": pcm_dev,
                        "channels": ch,
                        "bits_per_sample": bits,
                        "sample_rate": 16000,
                        "period_size": 360,
                        "buffer_size": 1440,
                    },
                )
                sess.last_audio_at = time.monotonic()

    async def _audio_worker(self, conn: Any, queue: asyncio.Queue[bytes]) -> None:
        while True:
            raw = await queue.get()
            try:
                await self._handle_binary_message(conn, raw)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error processing speaker audio frame")

    @staticmethod
    def enqueue_audio_frame(queue: asyncio.Queue[bytes], raw: bytes) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(raw)
        except asyncio.QueueFull:
            pass

    async def _handle_connection(self, conn: ServerConnection) -> None:
        expected_token = read_env_secret(self._config.speaker.auth_token_env)
        if expected_token:
            token = conn.request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if token != expected_token:
                await conn.close(code=4401, reason="Unauthorized")
                return

        self._connections.add(conn)
        self._pending_rpc.setdefault(conn, {})
        self._get_session(conn)
        self._get_playback(conn)
        logger.info("Speaker connected from %s", conn.remote_address)

        audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        audio_worker_task = asyncio.create_task(self._audio_worker(conn, audio_queue))

        if self._pipeline is not None:
            await self.start_recording(conn)
            watchdog_task: asyncio.Task[None] | None = asyncio.create_task(self._audio_watchdog(conn))
        else:
            watchdog_task = None

        try:
            async for raw in conn:
                if isinstance(raw, bytes):
                    self.enqueue_audio_frame(audio_queue, raw)
                    continue
                await self._handle_text_message(conn, raw)
        except ConnectionClosed:
            logger.info("Speaker connection closed")
        finally:
            if watchdog_task is not None:
                watchdog_task.cancel()
            audio_worker_task.cancel()
            self._cleanup_connection(conn)
            logger.info("Speaker disconnected")

    async def _handle_binary_message(self, conn: Any, raw: bytes) -> None:
        if self._pipeline is None:
            return
        pcm = extract_record_pcm_from_binary(raw)
        if not pcm:
            return
        sess = self._get_session(conn)
        sess.last_audio_at = time.monotonic()
        if sess.active_reply_task is not None and not sess.active_reply_task.done():
            self._pipeline.refresh_activity()
        loop = asyncio.get_running_loop()

        def on_wakeup(keyword: str) -> None:
            logger.info("Local wake word detected: %s", keyword)
            sess.mute_native_until = time.monotonic() + 8.0
            loop.call_soon_threadsafe(lambda: loop.create_task(self.play_wakeup_chime(conn)))

        def on_utterance(text: str) -> None:
            logger.info("Local ASR recognized: %s", text)
            loop.call_soon_threadsafe(
                lambda t=text: loop.create_task(self._process_local_utterance(conn, t))
            )

        def on_sleep() -> None:
            logger.info("Returned silently to wake-word idle")
            if self._on_session_reset is not None:
                loop.call_soon_threadsafe(self._on_session_reset)

        def on_interrupt(keyword: str) -> None:
            logger.info("Barge-in interrupt triggered (%s)", keyword)
            loop.call_soon_threadsafe(lambda: loop.create_task(self._handle_interrupt(conn)))

        def on_barge_in_wakeup(keyword: str) -> None:
            logger.info("Barge-in wake word triggered (%s)", keyword)
            loop.call_soon_threadsafe(lambda: loop.create_task(self._handle_barge_in_wakeup(conn)))

        await loop.run_in_executor(
            self._asr_executor,
            lambda: self._pipeline.feed_pcm(  # type: ignore[union-attr]
                pcm,
                on_wakeup=on_wakeup,
                on_utterance=on_utterance,
                on_sleep=on_sleep,
                on_interrupt=on_interrupt,
                on_barge_in_wakeup=on_barge_in_wakeup,
            ),
        )

    async def _handle_barge_in_wakeup(self, conn: Any) -> None:
        sess = self._get_session(conn)
        sess.interrupted = True
        sess.mute_native_until = time.monotonic() + 8.0
        if sess.active_reply_task is not None and not sess.active_reply_task.done():
            sess.active_reply_task.cancel()
        for fut in self._pending_rpc.get(conn, {}).values():
            if not fut.done():
                fut.cancel()
        pb = self._playbacks.get(conn)
        if pb is not None:
            await pb.stop_playback(stop_native=True)
        else:
            await self.stop_native_audio(conn)
        await self.play_wakeup_chime(conn)

    async def _handle_interrupt(self, conn: Any) -> None:
        sess = self._get_session(conn)
        sess.interrupted = True
        if sess.active_reply_task is not None and not sess.active_reply_task.done():
            sess.active_reply_task.cancel()
        for fut in self._pending_rpc.get(conn, {}).values():
            if not fut.done():
                fut.cancel()
        pb = self._playbacks.get(conn)
        if pb is not None:
            await pb.stop_playback(stop_native=True)
        else:
            await self.stop_native_audio(conn)
        if self._pipeline is not None:
            self._pipeline.set_awake(False)
            self._pipeline.set_muted(False)
            self._pipeline.mute_for(0.5)
        if self._on_session_reset is not None:
            self._on_session_reset()

    async def _yield_to_native_xiaoai(self, conn: Any) -> None:
        logger.info("Native XiaoAI activated; yielding control cleanly to avoid crosstalk")
        sess = self._get_session(conn)
        sess.interrupted = True
        sess.mute_native_until = 0.0
        self._mute_native_until = 0.0
        if sess.active_reply_task is not None and not sess.active_reply_task.done():
            sess.active_reply_task.cancel()
        for fut in self._pending_rpc.get(conn, {}).values():
            if not fut.done():
                fut.cancel()
        pb = self._playbacks.get(conn)
        if pb is not None:
            # Stop only our Stream("play") audio, never kill native XiaoAI's playback.
            await pb.stop_playback(stop_native=False)
        if self._pipeline is not None:
            self._pipeline.set_awake(False)
            self._pipeline.set_muted(False)
            self._pipeline.mute_for(6.0)
        if self._on_session_reset is not None:
            self._on_session_reset()

    async def _process_local_utterance(self, conn: Any, text: str) -> None:
        sess = self._get_session(conn)
        if self.is_exit_command(text):
            await self._handle_interrupt(conn)
            await self.speak_sentence(conn, "好的，已退出。", interrupt=True, wait_response=True)
            return
        cleaned = text.strip(" ，。！？,.!?")
        if cleaned in FILLER_NOISE_WORDS or (len(cleaned) <= 2 and cleaned not in {"确认", "取消", "是的", "好"}):
            logger.info("Ignoring short filler/noise utterance: %s", text)
            if self._pipeline is not None:
                self._pipeline.refresh_activity()
            return
        matched = match_wakeup_phrase(text, self._config.speech.local_keywords)
        if matched:
            _, rest = matched
            if not rest:
                await self.play_wakeup_chime(conn)
                if self._pipeline is not None:
                    self._pipeline.refresh_activity()
                return
            text = rest
        sess.mute_native_until = time.monotonic() + 10.0
        await self._run_query_and_speak(conn, text)

    async def _handle_text_message(self, conn: Any, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON message")
            return

        if "Response" in msg:
            resp = msg["Response"]
            if isinstance(resp, dict):
                req_id = resp.get("id")
                fut = self._pending_rpc.get(conn, {}).get(req_id) if isinstance(req_id, str) else None
                if fut is not None and not fut.done():
                    fut.set_result(resp)
            logger.debug("Speaker command response: %s", resp)
            return

        sess = self._get_session(conn)
        mute_until = max(sess.mute_native_until, self._mute_native_until)

        if self._pipeline is not None:
            pb = self._playbacks.get(conn)
            is_nomi_active = (
                self._pipeline.awake
                or (sess.active_reply_task is not None and not sess.active_reply_task.done())
                or (pb is not None and pb.is_playing)
            )

            if self._is_explicit_native_xiaoai_wakeup(msg) and is_nomi_active:
                await self._yield_to_native_xiaoai(conn)
                return

            is_playing_own_xiaoai = (
                self._config.speaker.tts_provider == "xiaoai" and pb is not None and pb.is_playing
            )
            if (
                not is_playing_own_xiaoai
                and (is_nomi_active or time.monotonic() < mute_until)
                and self._is_native_speak_event(msg)
            ):
                logger.info("Suppressing stray native XiaoAI Speak event while Nomi is active")
                asyncio.create_task(self.stop_native_audio(conn))
                return
            return

        pb = self._playbacks.get(conn)
        if (
            (pb is None or not pb.is_playing)
            and time.monotonic() < mute_until
            and self._is_native_speak_event(msg)
        ):
            logger.debug("Suppressing native XiaoAI Speak event")
            asyncio.create_task(self.stop_native_audio(conn))
            return

        extracted = self._extract_final_asr(msg)
        if not extracted:
            return
        dialog_id, text = extracted
        dedup_key = f"{dialog_id}:{text}"
        if dedup_key in sess.seen_dialogs:
            return
        sess.seen_dialogs.add(dedup_key)
        if len(sess.seen_dialogs) > 200:
            sess.seen_dialogs.clear()

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
        if any(phrase in cleaned for phrase in EXIT_PHRASES):
            return True
        if len(cleaned) <= 7 and "登录" not in cleaned and "账号" not in cleaned:
            return any(sub in cleaned for sub in SHORT_EXIT_SUBSTRINGS)
        return False

    @staticmethod
    def _parse_instruction_log(msg: dict[str, Any]) -> dict[str, Any] | None:
        evt = msg.get("Event")
        if not isinstance(evt, dict) or evt.get("event") != "instruction":
            return None
        data = evt.get("data")
        if not isinstance(data, dict):
            return None
        line = data.get("NewLine")
        if not isinstance(line, str):
            return None
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @classmethod
    def _is_explicit_native_xiaoai_wakeup(cls, msg: dict[str, Any]) -> bool:
        """True only on genuine XiaoAI wake events (never VoiceprintIdle/StopCapture/你好管家)."""
        log_msg = cls._parse_instruction_log(msg)
        if not log_msg:
            return False
        header = log_msg.get("header") or {}
        ns = header.get("namespace")
        name = header.get("name")
        if ns == "SpeechWakeup" and name == "Wakeup":
            return True
        if ns == "SpeechRecognizer" and name == "RecognizeResult":
            payload = log_msg.get("payload") or {}
            for item in payload.get("results") or []:
                if isinstance(item, dict):
                    txt = (item.get("text") or "").strip()
                    if txt.startswith("小爱"):
                        return True
        return False

    @classmethod
    def _is_native_speak_event(cls, msg: dict[str, Any]) -> bool:
        log_msg = cls._parse_instruction_log(msg)
        if not log_msg:
            return False
        header = log_msg.get("header") or {}
        return header.get("namespace") == "SpeechSynthesizer" and header.get("name") in {
            "Speak",
            "SpeakStream",
        }

    async def _process_utterance(self, conn: Any, text: str) -> None:
        sess = self._get_session(conn)
        if self.is_exit_command(text):
            sess.continuous_mode = False
            sess.mute_native_until = time.monotonic() + 6.0
            if self._on_session_reset is not None:
                self._on_session_reset()
            await self.speak_sentence(conn, "好的，已退出对话。", interrupt=True, wait_response=True)
            return

        matched_kw = next((kw for kw in self._config.speech.wakeup_keywords if kw in text), None)
        if not matched_kw and not sess.continuous_mode:
            return

        sess.mute_native_until = time.monotonic() + 10.0
        await self.stop_native_audio(conn)

        query = text
        if matched_kw and text.startswith(matched_kw):
            query = text[len(matched_kw) :].lstrip(" ，,：:")
            if not query:
                sess.continuous_mode = True
                await self.speak_sentence(conn, "我在，请讲。", interrupt=True, wait_response=True)
                return

        await self._run_query_and_speak(conn, query)

    async def _run_query_and_speak(self, conn: Any, query: str) -> None:
        sess = self._get_session(conn)
        async with sess.query_lock:
            sess.interrupted = False
            task = asyncio.current_task()
            sess.active_reply_task = task
            try:
                if self._on_stream_query is not None:
                    await self._run_streaming_reply(conn, query)
                    return
                assert self._on_query is not None
                reply = await self._on_query(query)
                logger.info("LLM reply: %s", reply)
                sess.mute_native_until = time.monotonic() + 8.0
                await self.speak_sentence(conn, reply, interrupt=True, wait_response=True)
            except asyncio.CancelledError:
                logger.info("Reply task cancelled due to interrupt")
                raise
            except Exception as exc:
                logger.exception("Error handling query %r", query)
                await self.speak_sentence(conn, f"处理出错了：{exc}", interrupt=True, wait_response=True)
            finally:
                if self._pipeline is not None and not sess.interrupted:
                    self._pipeline.refresh_activity()
                if sess.active_reply_task is task:
                    sess.active_reply_task = None

    async def _run_streaming_reply(self, conn: Any, query: str) -> None:
        assert self._on_stream_query is not None
        sess = self._get_session(conn)
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def producer() -> None:
            try:
                async for sentence in self._on_stream_query(query):
                    if sentence:
                        await queue.put(sentence)
            except Exception as exc:
                logger.exception("Streaming error for %r", query)
                await queue.put(f"处理出错了：{exc}")
            finally:
                await queue.put(None)

        prod_task = asyncio.create_task(producer())
        first_spoken = True
        try:
            while True:
                if sess.interrupted:
                    break
                sentence = await queue.get()
                if sentence is None or sess.interrupted:
                    break
                logger.info("Streaming TTS sentence: %s", sentence)
                sess.mute_native_until = time.monotonic() + 8.0
                await self.speak_sentence(conn, sentence, interrupt=first_spoken, wait_response=True)
                first_spoken = False
            await prod_task
        except asyncio.CancelledError:
            logger.info("Streaming reply cancelled due to barge-in")
        finally:
            if not prod_task.done():
                prod_task.cancel()
            if self._pipeline is not None and not sess.interrupted:
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
        self.ensure_pipeline()

        host = self._config.speaker.ws_host
        port = self._config.speaker.ws_port
        logger.info("Listening for speaker WebSocket on ws://%s:%s", host, port)
        async with websockets.serve(
            self._handle_connection,
            host,
            port,
            ping_interval=30,
            ping_timeout=60,
        ):
            await asyncio.Future()

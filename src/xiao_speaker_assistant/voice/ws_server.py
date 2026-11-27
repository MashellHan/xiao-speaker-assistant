"""WebSocket server (:4399) speaking open-xiaoai protocol with PlaybackController & per-connection sessions."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from typing import Any, Awaitable, Callable

import websockets
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from xiao_speaker_assistant.config import AppConfig, read_env_secret
from xiao_speaker_assistant.tools.confirmation import is_cancel_utterance, is_confirm_utterance
from xiao_speaker_assistant.tools.history import (
    CURRENT_TURN_TRACE,
    TurnTraceBuilder,
    TurnTraceStore,
)
from xiao_speaker_assistant.voice.audio_pipeline import (
    LocalSpeechPipeline,
    extract_record_pcm_from_binary,
    match_wakeup_phrase,
)
from xiao_speaker_assistant.voice.playback import PlaybackController
from xiao_speaker_assistant.voice.speaker_id import SpeakerMatch, SpeakerProfileStore
from xiao_speaker_assistant.voice.tts_stream import PrewarmedEdgeTTS, synthesize_tts_pcm_stream
from xiao_speaker_assistant.voice.voice_catalog import resolve_edge_voice

logger = logging.getLogger(__name__)
QueryHandler = Callable[[str], Awaitable[str]]
StreamQueryHandler = Callable[[str], AsyncIterator[str]]
CURRENT_SPEAKER: ContextVar[SpeakerMatch | None] = ContextVar("xsa_current_speaker", default=None)

IGNORE_BARE_WORDS = {"小爱同学", "小爱"}
WAKE_FRAGMENT_WORDS = {
    "管家",
    "关家",
    "诺米",
    "糯米",
}
FILLER_NOISE_WORDS = {"嗯", "啊", "哦", "呃", "额", "唉", "唔", "哈", "嘿", "喂", "嗯嗯", "啊啊", "哦哦", "哈哈"}
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
    cloud_asr_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cloud_asr_tasks: deque[asyncio.Task[None]] = field(default_factory=deque)
    asr_epoch: int = 0
    mute_native_until: float = 0.0
    last_audio_at: float = 0.0
    last_chime_at: float = 0.0
    last_playback_ended_at: float = 0.0
    active_reply_task: asyncio.Task[None] | None = None
    interrupted: bool = False
    active_voice: str | None = None
    recent_spoken_texts: deque[tuple[float, str]] = field(default_factory=deque)


_MAX_PENDING_CLOUD_ASR_TASKS = 2
_CLOUD_ASR_TIMEOUT_SEC = 25.0


_ECHO_STRIP_RE = re.compile(r"[\s，,。.!！？?、；;：:\"'`~～]+")

_KNOWN_WEATHER_CITIES: tuple[str, ...] = tuple(
    sorted(
        (
            "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆", "武汉", "西安",
            "天津", "苏州", "长沙", "郑州", "青岛", "大连", "宁波", "厦门", "福州", "济南",
            "合肥", "昆明", "沈阳", "长春", "哈尔滨", "石家庄", "南昌", "贵阳", "兰州", "海口",
            "三亚", "珠海", "佛山", "东莞", "无锡", "常州", "南通", "温州", "泉州", "烟台",
            "太原", "银川", "西宁", "拉萨", "乌鲁木齐", "呼和浩特", "南宁", "桂林", "扬州",
            "徐州", "绍兴", "嘉兴", "金华", "台州", "惠州", "中山", "江门", "湛江", "汕头",
            "威海", "潍坊", "临沂", "洛阳", "芜湖", "赣州", "遵义", "大理", "丽江", "西双版纳",
            "香港", "澳门", "台北", "高雄", "海淀", "朝阳", "浦东", "雄安",
            "东京", "大阪", "首尔", "新加坡", "曼谷", "纽约", "旧金山", "洛杉矶", "西雅图",
            "伦敦", "巴黎", "悉尼", "墨尔本", "多伦多", "温哥华",
        ),
        key=len,
        reverse=True,
    )
)
_SIMPLE_GREETINGS = frozenset({"你好", "您好", "在吗", "在不在", "哈喽", "hello", "hi", "嗨", "早安", "晚安", "谢谢", "感谢"})
_INDOOR_TEMP_HINTS = ("空调", "室内", "房间", "客厅", "卧室", "热水", "水温", "体温", "发烧", "冰箱", "烤箱")


def build_conversational_ack(query: str) -> tuple[str | None, float | None]:
    """Return `(ack_phrase, delay_sec)` for slow/tool-likely queries, or `(None, None)` to skip.

    Tool-likely intents (weather, Xiaohongshu, email, Mac files, Codex, web search, browser)
    speak a short context-aware acknowledgement if the LLM hasn't emitted its first sentence
    within ~1.5s; general slow turns fall back to a gentle filler after `tool_ack_delay_sec`
    (default 3.5s) so the user never waits 5+ seconds in silence.
    """
    text = query.strip()
    cleaned = _ECHO_STRIP_RE.sub("", text).lower()
    if len(cleaned) < 2:
        return None, None
    if cleaned in _SIMPLE_GREETINGS:
        return None, None
    # Only skip short direct confirmation/cancellation utterances, not longer requests like "帮我取消明天的闹钟".
    if len(cleaned) <= 6 and (is_confirm_utterance(text) or is_cancel_utterance(text)):
        return None, None

    lowered = text.lower()
    if "小红书" in text:
        return "好的，我这就去小红书帮你搜一下。", 1.2
    if any(k in text for k in ("邮件", "邮箱")) or "outlook" in lowered or "mail" in lowered:
        return "好的，我这就帮你查一下邮件。", 1.5
    has_weather_kw = any(k in text for k in ("天气", "气温", "下雨", "下雪", "空气质量", "穿什么")) or (
        "温度" in text and not any(h in text for h in _INDOOR_TEMP_HINTS)
    )
    if has_weather_kw:
        city = next((c for c in _KNOWN_WEATHER_CITIES if c in text), "")
        if city:
            return f"好的，我这就帮你查一下{city}天气。", 1.2
        return "好的，我这就帮你查一下天气。", 1.2
    if any(k in text for k in ("磁盘", "硬盘", "磁盘空间", "硬盘空间", "存储空间", "桌面文件", "下载目录", "读一下文件", "查看文件")):
        return "好的，我这就帮你看一下电脑。", 1.5
    if any(k in text for k in ("代码", "仓库", "报错")) or "bug" in lowered or "codex" in lowered:
        return "好的，我这就帮你分析一下。", 1.8
    if any(k in text for k in ("网页", "网站", "打开链接", "打开浏览器")):
        return "好的，我这就帮你看一下网页。", 1.5
    if any(k in text for k in ("搜一下", "搜索", "查一下", "查下", "新闻", "股价", "汇率", "多少钱", "最新")):
        return "好的，我这就帮你查一下。", 1.5
    if any(k in text for k in ("提醒我", "定个闹钟", "倒计时", "早报")):
        return "好的，我这就帮你安排一下。", 1.8
    # General non-tool query: only speak a gentle filler if LLM/tool takes longer than `tool_ack_delay_sec`.
    return "稍等一下，我这就帮你查一下。", None


def is_self_echo(asr_text: str, recent_spoken: Iterable[str]) -> bool:
    """Return True if `asr_text` looks like a residual TTS self-echo fragment."""
    if is_confirm_utterance(asr_text) or is_cancel_utterance(asr_text):
        return False
    norm_asr = _ECHO_STRIP_RE.sub("", asr_text).lower()
    if len(norm_asr) < 2:
        return False
    combined = ""
    for spoken in recent_spoken:
        norm_spoken = _ECHO_STRIP_RE.sub("", spoken).lower()
        if not norm_spoken:
            continue
        if norm_asr in norm_spoken:
            return True
        combined += norm_spoken
    if len(norm_asr) >= 3 and combined:
        bigrams = [norm_asr[i : i + 2] for i in range(len(norm_asr) - 1)]
        hits = sum(1 for bg in bigrams if bg in combined)
        if hits / len(bigrams) >= 0.65:
            return True
    return False


class SpeakerGateway:
    def __init__(
        self,
        config: AppConfig,
        on_query: QueryHandler | None = None,
        on_stream_query: StreamQueryHandler | None = None,
        on_session_reset: Callable[[], None] | None = None,
        *,
        speaker_store: SpeakerProfileStore | None = None,
        trace_store: TurnTraceStore | None = None,
        audio_watchdog_interval: float = 3.0,
        audio_stall_timeout: float = 4.0,
    ) -> None:
        if on_query is None and on_stream_query is None:
            raise ValueError("Either on_query or on_stream_query must be provided")
        self._config = config
        self._on_query = on_query
        self._on_stream_query = on_stream_query
        self._on_session_reset = on_session_reset
        self._speaker_store = speaker_store
        self._trace_store = trace_store
        self._audio_watchdog_interval = audio_watchdog_interval
        self._audio_stall_timeout = audio_stall_timeout
        self._connections: set[ServerConnection] = set()
        self._relay_connections: set[Any] = set()
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

    def reload_config(self, new_config: AppConfig) -> None:
        self._config = new_config
        for pb in self._playbacks.values():
            pb._config = new_config.speaker
        if self._pipeline is not None:
            self._pipeline.update_config(new_config.speech)

    def apply_voice_change(self, voice: str, rate: str) -> None:
        self._config.speaker.edge_voice = voice
        self._config.speaker.edge_rate = rate
        for pb in self._playbacks.values():
            pb._config = self._config.speaker
        for sess in self._sessions.values():
            sess.active_voice = None

    def _resolve_voice_for_keyword(self, keyword: str) -> str | None:
        routes = self._config.speaker.voice_routes
        if not routes:
            return None
        norm = keyword.strip().lower().replace(" ", "_")
        if not norm:
            return None
        for route_key, voice in routes.items():
            rk = route_key.strip().lower().replace(" ", "_")
            if not rk:
                continue
            if rk == norm or rk.replace("_", "") == norm.replace("_", "") or rk in norm:
                return voice.strip() or None
        return None

    def _get_playback(self, conn: Any) -> PlaybackController:
        pb = self._playbacks.get(conn)
        if pb is not None:
            return pb

        def on_state(playing: bool) -> None:
            if not playing:
                self._get_session(conn).last_playback_ended_at = time.monotonic()
            if self._pipeline is None:
                return
            # When 2ch hardware ch3 AEC is active, keep VAD unmuted so the user can interrupt
            # with direct follow-ups ("北京呢？") while Nomi is speaking; self-echoes are filtered
            # via LinearEchoCanceller + is_self_echo().
            if self._config.speech.aec_enabled and self._config.speech.capture_channels >= 2:
                return
            if playing:
                self._pipeline.set_muted(True)
            else:
                self._pipeline.mute_for(0.25)
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
            wan_relay=(conn in self._relay_connections),
        )
        self._playbacks[conn] = pb
        return pb

    @staticmethod
    def _cancel_cloud_asr_tasks(sess: SpeakerSession) -> None:
        """Invalidate and cancel any in-flight/queued cloud ASR tasks for `sess`."""
        sess.asr_epoch += 1
        while sess.cloud_asr_tasks:
            t = sess.cloud_asr_tasks.popleft()
            if not t.done():
                t.cancel()

    def _cleanup_connection(self, conn: Any) -> None:
        self._connections.discard(conn)
        self._relay_connections.discard(conn)
        sess = self._sessions.pop(conn, None)
        if sess is not None:
            self._cancel_cloud_asr_tasks(sess)
            if sess.active_reply_task is not None and not sess.active_reply_task.done():
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
        sess = self._get_session(conn)
        sess.last_chime_at = time.monotonic()
        pb = self._get_playback(conn)
        await pb.play_chime()

    def _resolve_active_voice_override(self, conn: Any) -> str | None:
        sess = self._get_session(conn)
        voice_override = sess.active_voice
        if voice_override is None:
            spk = CURRENT_SPEAKER.get()
            if spk is not None and spk.preferred_voice:
                try:
                    voice_override = resolve_edge_voice(spk.preferred_voice).id
                except ValueError:
                    pass
        return voice_override

    async def speak_sentence(
        self,
        conn: Any,
        text: str,
        *,
        interrupt: bool = True,
        wait_response: bool = True,
        keep_stream_open: bool = False,
        prefetched_pcm: bytes | None = None,
        prewarmed_tts: PrewarmedEdgeTTS | None = None,
    ) -> None:
        pb = self._get_playback(conn)
        voice_override = self._resolve_active_voice_override(conn)
        if keep_stream_open or prefetched_pcm is not None or prewarmed_tts is not None:
            kwargs: dict[str, Any] = {
                "interrupt_native": interrupt,
                "voice_override": voice_override,
                "keep_stream_open": keep_stream_open,
            }
            if prefetched_pcm is not None:
                kwargs["prefetched_pcm"] = prefetched_pcm
            if prewarmed_tts is not None:
                kwargs["prewarmed_tts"] = prewarmed_tts
            await pb.speak(text, **kwargs)
        else:
            await pb.speak(
                text,
                interrupt_native=interrupt,
                voice_override=voice_override,
            )

    async def abort_and_speak(self, conn: Any, text: str) -> None:
        await self.speak_sentence(conn, text, interrupt=True, wait_response=False)

    async def send_tts(self, conn: Any, text: str) -> None:
        await self.abort_and_speak(conn, text)

    def _is_conn_idle(self, conn: Any) -> bool:
        sess = self._get_session(conn)
        pb = self._playbacks.get(conn)
        busy = (sess.active_reply_task is not None and not sess.active_reply_task.done()) or (
            pb is not None and pb.is_playing
        )
        return not busy

    def has_idle_speaker(self) -> bool:
        return any(self._is_conn_idle(conn) for conn in list(self._connections))

    def get_runtime_status(self) -> dict[str, Any]:
        """Return live speaker connectivity and pipeline state for Web Console telemetry.

        Over WAN relay, the host socket connects to the relay server even when the speaker
        is powered off; therefore relay speaker presence requires recent audio frames (<15s)
        when the local speech pipeline is active.
        """
        conns = list(self._connections)
        now = time.monotonic()
        relay_count = sum(1 for c in conns if c in self._relay_connections)
        lan_count = len(conns) - relay_count
        relay_audio_active = False
        for c in conns:
            if c in self._relay_connections:
                sess = self._sessions.get(c)
                if sess is not None and sess.last_audio_at > 0 and (now - sess.last_audio_at) < 15.0:
                    relay_audio_active = True
                    break
        speaker_connected = lan_count > 0 or (
            relay_audio_active if self._pipeline is not None else relay_count > 0
        )
        return {
            "speaker_connected": speaker_connected,
            "relay_connected": relay_count > 0,
            "active_connections": len(conns),
            "relay_connections": relay_count,
            "lan_connections": lan_count,
            "pipeline_awake": bool(self._pipeline.awake) if self._pipeline is not None else False,
        }

    async def proactive_speak(self, text: str) -> bool:
        """Speak an unsolicited proactive reminder/briefing on an idle connected speaker."""
        if not text.strip():
            return False
        for conn in list(self._connections):
            if not self._is_conn_idle(conn):
                continue
            sess = self._get_session(conn)
            self._record_spoken(sess, text)
            sess.mute_native_until = time.monotonic() + 8.0
            await self.play_wakeup_chime(conn)
            await self.speak_sentence(conn, text, interrupt=True, wait_response=True)
            return True
        return False

    def _recording_params(self, conn: Any) -> dict[str, Any]:
        # Over WAN/SSH relay, use 60ms capture periods (960 frames vs 360) so uplink frame rate
        # drops from ~44 frames/s to ~16.6 frames/s, cutting Dropbear/WS overhead.
        is_wan = conn in self._relay_connections
        period_size = 960 if is_wan else 360
        buffer_size = 3840 if is_wan else 1440
        return {
            "pcm": self._config.speech.capture_pcm,
            "channels": self._config.speech.capture_channels,
            "bits_per_sample": self._config.speech.capture_bits_per_sample,
            "sample_rate": 16000,
            "period_size": period_size,
            "buffer_size": buffer_size,
        }

    async def start_recording(self, conn: Any) -> None:
        sess = self._get_session(conn)
        params = self._recording_params(conn)
        logger.info(
            "Starting speaker audio stream (%s, %dch, %d-bit, period=%d, aec=%s) for local KWS (`%s`) + %s",
            params["pcm"],
            params["channels"],
            params["bits_per_sample"],
            params["period_size"],
            self._config.speech.aec_enabled,
            ", ".join(self._config.speech.local_keywords),
            self._config.speech.asr_model,
        )
        await self.send_command(conn, "start_recording", params)
        sess.last_audio_at = time.monotonic()

    async def _audio_watchdog(self, conn: Any) -> None:
        sess = self._get_session(conn)
        consecutive_stalls = 0
        while True:
            await asyncio.sleep(self._audio_watchdog_interval)
            if self._pipeline is None:
                return
            if (time.monotonic() - sess.last_audio_at) > self._audio_stall_timeout:
                consecutive_stalls += 1
                # Throttle warning logs when the remote speaker is offline (still probe
                # start_recording periodically so a reconnecting speaker resumes promptly).
                if consecutive_stalls <= 3 or consecutive_stalls % 5 == 0:
                    logger.warning("Audio stream stalled >4s; restarting speaker recording")
                try:
                    await self.send_command(conn, "start_recording", self._recording_params(conn))
                except ConnectionClosed:
                    return
                sess.last_audio_at = time.monotonic()
            else:
                consecutive_stalls = 0

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
            sess.active_voice = self._resolve_voice_for_keyword(keyword)
            sess.mute_native_until = time.monotonic() + 8.0
            loop.call_soon_threadsafe(lambda: loop.create_task(self.play_wakeup_chime(conn)))

        def on_utterance(text: str, embedding: list[float] | None = None) -> None:
            logger.info("Local ASR recognized: %s", text)
            metrics = dict(getattr(self._pipeline, "_last_asr_metrics", None) or {})
            loop.call_soon_threadsafe(
                lambda t=text, emb=embedding, m=metrics: loop.create_task(
                    self._process_local_utterance(conn, t, embedding=emb, asr_metrics=m)
                )
            )

        def on_sleep() -> None:
            logger.info("Returned silently to wake-word idle")
            sess.active_voice = None
            loop.call_soon_threadsafe(lambda: self._cancel_cloud_asr_tasks(sess))
            if self._on_session_reset is not None:
                loop.call_soon_threadsafe(self._on_session_reset)

        def on_interrupt(keyword: str) -> None:
            logger.info("Barge-in interrupt triggered (%s)", keyword)
            loop.call_soon_threadsafe(lambda: self._cancel_cloud_asr_tasks(sess))
            loop.call_soon_threadsafe(lambda: loop.create_task(self._handle_interrupt(conn)))

        def on_barge_in_wakeup(keyword: str) -> None:
            logger.info("Barge-in wake word triggered (%s)", keyword)
            sess.active_voice = self._resolve_voice_for_keyword(keyword)
            loop.call_soon_threadsafe(lambda: self._cancel_cloud_asr_tasks(sess))
            loop.call_soon_threadsafe(lambda: loop.create_task(self._handle_barge_in_wakeup(conn)))

        def on_cloud_segment(
            segment: Any,
            embedding: list[float] | None,
            vad_end_mono: float,
            speaker_id_ms: float | None,
        ) -> None:
            def _schedule() -> None:
                if conn not in self._connections or self._sessions.get(conn) is not sess:
                    return
                # Bound queued cloud ASR tasks so a slow/broken network cannot accumulate stale segments.
                while len(sess.cloud_asr_tasks) >= _MAX_PENDING_CLOUD_ASR_TASKS:
                    stale = sess.cloud_asr_tasks.popleft()
                    if not stale.done():
                        stale.cancel()
                epoch = sess.asr_epoch
                task = loop.create_task(
                    self._process_cloud_segment(
                        conn,
                        sess,
                        epoch,
                        segment,
                        embedding=embedding,
                        vad_end_mono=vad_end_mono,
                        speaker_id_ms=speaker_id_ms,
                    )
                )
                sess.cloud_asr_tasks.append(task)

                def _discard_done(done_task: asyncio.Task[None]) -> None:
                    try:
                        sess.cloud_asr_tasks.remove(done_task)
                    except ValueError:
                        pass

                task.add_done_callback(_discard_done)

            loop.call_soon_threadsafe(_schedule)

        use_async_cloud = bool(getattr(self._pipeline, "has_cloud_asr", False))
        await loop.run_in_executor(
            self._asr_executor,
            lambda: self._pipeline.feed_pcm(  # type: ignore[union-attr]
                pcm,
                on_wakeup=on_wakeup,
                on_utterance_embedding=on_utterance,
                **({"on_cloud_segment": on_cloud_segment} if use_async_cloud else {}),
                on_sleep=on_sleep,
                on_interrupt=on_interrupt,
                on_barge_in_wakeup=on_barge_in_wakeup,
            ),
        )

    async def _process_cloud_segment(
        self,
        conn: Any,
        sess: SpeakerSession,
        epoch: int,
        segment: Any,
        *,
        embedding: list[float] | None,
        vad_end_mono: float,
        speaker_id_ms: float | None,
    ) -> None:
        """Run cloud ASR off the KWS/VAD audio thread so barge-in & VAD stay real-time."""
        if self._pipeline is None:
            return

        def _is_still_valid() -> bool:
            return (
                conn in self._connections
                and self._sessions.get(conn) is sess
                and sess.asr_epoch == epoch
            )

        async with sess.cloud_asr_lock:
            if not _is_still_valid():
                return
            t_req = time.monotonic()
            try:
                text = await asyncio.wait_for(
                    self._pipeline.atranscribe_cloud(segment),
                    timeout=_CLOUD_ASR_TIMEOUT_SEC,
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                logger.warning("Async cloud ASR timed out after %.0fs", _CLOUD_ASR_TIMEOUT_SEC)
                return
            except Exception as exc:
                logger.warning("Async cloud ASR failed: %s", exc)
                return
            if not _is_still_valid():
                logger.debug("Dropping stale cloud ASR result after interrupt/disconnect")
                return
            asr_ms = round((time.monotonic() - t_req) * 1000.0, 1)
            resolved = self._pipeline.resolve_transcribed_text(
                text,
                vad_end_mono=vad_end_mono,
                asr_ms=asr_ms,
                speaker_id_ms=speaker_id_ms,
            )
            if resolved is None or not _is_still_valid():
                return
            kind, payload, metrics = resolved
            if kind == "wakeup":
                logger.info("Cloud ASR wake word detected: %s", payload)
                sess.active_voice = self._resolve_voice_for_keyword(payload)
                sess.mute_native_until = time.monotonic() + 8.0
                asyncio.create_task(self.play_wakeup_chime(conn))
                return
            logger.info("Cloud ASR recognized: %s (%.0fms)", payload, asr_ms)
            asyncio.create_task(
                self._process_local_utterance(
                    conn,
                    payload,
                    embedding=embedding,
                    asr_metrics=metrics,
                )
            )

    async def _handle_barge_in_wakeup(self, conn: Any) -> None:
        sess = self._get_session(conn)
        self._cancel_cloud_asr_tasks(sess)
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
        self._cancel_cloud_asr_tasks(sess)
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
        self._cancel_cloud_asr_tasks(sess)
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

    def _record_spoken(self, sess: SpeakerSession, text: str) -> None:
        now = time.monotonic()
        sess.recent_spoken_texts.append((now, text))
        while len(sess.recent_spoken_texts) > 8 or (
            sess.recent_spoken_texts and (now - sess.recent_spoken_texts[0][0]) > 15.0
        ):
            sess.recent_spoken_texts.popleft()

    def _recent_spoken_list(self, sess: SpeakerSession) -> list[str]:
        now = time.monotonic()
        return [t for ts, t in sess.recent_spoken_texts if (now - ts) <= 15.0]

    async def _process_local_utterance(
        self,
        conn: Any,
        text: str,
        *,
        embedding: list[float] | None = None,
        asr_metrics: dict[str, float] | None = None,
    ) -> None:
        sess = self._get_session(conn)
        if self.is_exit_command(text):
            await self._handle_interrupt(conn)
            await self.speak_sentence(conn, "好的，已退出。", interrupt=True, wait_response=True)
            return
        cleaned = text.strip(" ，。！？,.!?")
        if (
            len(cleaned) <= 1
            or cleaned in FILLER_NOISE_WORDS
            or cleaned.lower() in WAKE_FRAGMENT_WORDS
        ):
            logger.info("Ignoring short filler/noise utterance: %s", text)
            if self._pipeline is not None:
                self._pipeline.refresh_activity()
            return
        pb = self._playbacks.get(conn)
        old_task = sess.active_reply_task
        is_speaking_now = (old_task is not None and not old_task.done()) or (
            pb is not None and pb.is_playing
        )
        in_echo_window = is_speaking_now or (
            (time.monotonic() - sess.last_playback_ended_at) < 1.5
        )
        if in_echo_window and is_self_echo(text, self._recent_spoken_list(sess)):
            logger.info("Ignoring residual TTS self-echo: %s", text)
            return

        matched = match_wakeup_phrase(text, self._config.speech.local_keywords)
        if matched:
            wake_label, rest = matched
            sess.active_voice = self._resolve_voice_for_keyword(wake_label)
            if not rest:
                if is_speaking_now:
                    await self._handle_barge_in_wakeup(conn)
                    return
                if (time.monotonic() - sess.last_chime_at) >= 2.0:
                    await self.play_wakeup_chime(conn)
                if self._pipeline is not None:
                    self._pipeline.refresh_activity()
                return
            text = rest

        speaker_match: SpeakerMatch | None = None
        if self._speaker_store is not None:
            if embedding is not None and self._config.speech.speaker_id_enabled:
                speaker_match = self._speaker_store.identify(
                    embedding, threshold=self._config.speech.speaker_id_threshold
                )
            self._speaker_store.record_utterance(text, embedding, matched=speaker_match)
            if speaker_match is not None:
                logger.info(
                    "Identified speaker: %s (role=%s, score=%.2f)",
                    speaker_match.name,
                    speaker_match.role,
                    speaker_match.score,
                )

        # Full-duplex barge-in: if Nomi is currently speaking or thinking, cancel current reply
        # without resetting session context and immediately answer the new follow-up utterance.
        if is_speaking_now:
            logger.info("Full-duplex follow-up barge-in: %s", text)
            sess.interrupted = True
            if old_task is not None and not old_task.done():
                old_task.cancel()
            for fut in self._pending_rpc.get(conn, {}).values():
                if not fut.done():
                    fut.cancel()
            if pb is not None:
                await pb.stop_playback(stop_native=True)
                sess.last_playback_ended_at = time.monotonic()
            if old_task is not None and not old_task.done():
                try:
                    await old_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

        sess.mute_native_until = time.monotonic() + 10.0
        await self._run_query_and_speak(
            conn,
            text,
            speaker_match=speaker_match,
            asr_metrics=asr_metrics,
            barge_in=is_speaking_now,
        )

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

    async def _run_query_and_speak(
        self,
        conn: Any,
        query: str,
        *,
        speaker_match: SpeakerMatch | None = None,
        asr_metrics: dict[str, float] | None = None,
        barge_in: bool = False,
    ) -> None:
        sess = self._get_session(conn)
        async with sess.query_lock:
            sess.interrupted = False
            task = asyncio.current_task()
            sess.active_reply_task = task
            token = CURRENT_SPEAKER.set(speaker_match)
            pb = self._get_playback(conn)
            now_mono = time.monotonic()
            vad_end_mono = float(asr_metrics["vad_end_mono"]) if (asr_metrics and "vad_end_mono" in asr_metrics) else now_mono
            if vad_end_mono > now_mono:
                vad_end_mono = now_mono
            active_voice = self._resolve_active_voice_override(conn) or self._config.speaker.edge_voice
            trace = TurnTraceBuilder(
                turn_started_mono=vad_end_mono,
                query_started_mono=now_mono,
                asr_ms=asr_metrics.get("asr_ms") if asr_metrics else None,
                speaker_id_ms=asr_metrics.get("speaker_id_ms") if asr_metrics else None,
                model=(
                    self._config.llm.providers.opencode.model
                    if self._config.llm.default_provider == "opencode"
                    else self._config.llm.providers.codex_api.model
                    if self._config.llm.default_provider in {"codex_api", "openai_compatible"}
                    else self._config.llm.providers.gemini_api.model
                    if self._config.llm.default_provider == "gemini_api"
                    else (self._config.llm.providers.codex.model or "codex")
                ),
                voice=active_voice,
                transport="wan_relay" if conn in self._relay_connections else "lan",
                sample_rate=pb.effective_sample_rate,
                barge_in=barge_in,
            )
            trace_token = CURRENT_TURN_TRACE.set(trace)
            try:
                if self._on_stream_query is not None:
                    await self._run_streaming_reply(conn, query)
                    if sess.interrupted and trace.status == "ok":
                        trace.status = "interrupted"
                    return
                assert self._on_query is not None
                reply = await self._on_query(query)
                trace.mark_llm_first_sentence()
                trace.mark_llm_done()
                trace.reply_parts.append(reply)
                trace.sentence_count = 1 if reply.strip() else 0
                logger.info("LLM reply: %s", reply)
                self._record_spoken(sess, reply)
                sess.mute_native_until = time.monotonic() + 8.0
                trace.tts_started_mono = time.monotonic()
                await self.speak_sentence(conn, reply, interrupt=True, wait_response=True)
                trace.tts_done_mono = time.monotonic()
            except asyncio.CancelledError:
                if trace.status == "ok":
                    trace.status = "interrupted"
                logger.info("Reply task cancelled due to interrupt")
                raise
            except Exception as exc:
                trace.status = "error"
                logger.exception("Error handling query %r", query)
                # Detach trace context so the spoken error notice never pollutes turn TTS/TTFA metrics.
                # Speak a short user-friendly message instead of reading raw Python tracebacks/URLs aloud.
                err_tok = CURRENT_TURN_TRACE.set(None)
                try:
                    await self.speak_sentence(conn, "抱歉，刚才处理遇到了一点问题，请稍后再试。", interrupt=True, wait_response=True)
                finally:
                    CURRENT_TURN_TRACE.reset(err_tok)
            finally:
                if self._trace_store is not None:
                    try:
                        self._trace_store.record(
                            query=query,
                            reply="".join(trace.reply_parts),
                            breakdown_ms=trace.build_breakdown(),
                            model=trace.model,
                            voice=trace.voice,
                            speaker_name=speaker_match.name if speaker_match is not None else "未知",
                            speaker_role=speaker_match.role if speaker_match is not None else "",
                            transport=trace.transport,
                            sample_rate=trace.sample_rate,
                            sentence_count=trace.sentence_count,
                            barge_in=trace.barge_in,
                            status=trace.status,
                        )
                    except Exception as exc:
                        logger.debug("Failed persisting turn trace: %s", exc)
                CURRENT_TURN_TRACE.reset(trace_token)
                CURRENT_SPEAKER.reset(token)
                if self._pipeline is not None and not sess.interrupted:
                    self._pipeline.refresh_activity()
                if sess.active_reply_task is task:
                    sess.active_reply_task = None

    async def _run_streaming_reply(self, conn: Any, query: str) -> None:
        assert self._on_stream_query is not None
        sess = self._get_session(conn)
        pb = self._get_playback(conn)
        use_continuous_stream = self._config.speaker.tts_provider != "xiaoai"
        voice_override = self._resolve_active_voice_override(conn)

        # Queue items are `(sentence, optional_prefetch_task, kind)` where kind is
        # `"reply"` | `"ack"` | `"error"`. Sentences 2+ begin Edge-TTS synthesis in the
        # background while sentence 1 is still streaming to ALSA.
        queue: asyncio.Queue[tuple[str, asyncio.Task[bytes] | None, str] | None] = asyncio.Queue()
        all_prefetch_tasks: list[asyncio.Task[bytes]] = []
        prefetch_sem = asyncio.Semaphore(2)
        first_llm_ready = asyncio.Event()

        all_prewarm_tasks: list[asyncio.Task[PrewarmedEdgeTTS | None]] = []

        # Pre-open Edge-TTS WebSocket & speech.config in parallel with LLM generation so the
        # ~600ms TLS+WSS handshake overlaps with LLM thinking instead of adding serially to TTFA.
        prewarm_tts_task: asyncio.Task[PrewarmedEdgeTTS | None] | None = None
        if self._config.speaker.tts_provider == "edge":
            async def _open_prewarm_tts() -> PrewarmedEdgeTTS | None:
                try:
                    return await PrewarmedEdgeTTS.open(connect_timeout=5.0)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("Edge-TTS pre-open skipped: %s", exc)
                    return None

            prewarm_tts_task = asyncio.create_task(_open_prewarm_tts())
            all_prewarm_tasks.append(prewarm_tts_task)

        # Over WAN relay, speculatively stop native audio + open `start_play` under `pb._lock`
        # in parallel with LLM generation so the ~550ms WAN RPC round-trip overlaps LLM think.
        prestart_play_task: asyncio.Task[bool] | None = None
        if use_continuous_stream and conn in self._relay_connections and not pb.stream_open:
            async def _prestart_play() -> bool:
                try:
                    return await pb.prestart_stream(stop_native=True)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("Speculative start_play skipped: %s", exc)
                    return False

            prestart_play_task = asyncio.create_task(_prestart_play())

        async def _take_prewarmed_tts() -> PrewarmedEdgeTTS | None:
            nonlocal prewarm_tts_task
            t = prewarm_tts_task
            if t is None:
                return None
            prewarm_tts_task = None
            if not t.done():
                # Cancel in-flight prewarm without blocking the ready sentence; finally awaits & closes it.
                t.cancel()
                return None
            try:
                res = t.result()
                if res is not None and not res.usable:
                    await res.close()
                    return None
                return res
            except BaseException:
                return None

        async def _prefetch_sentence_pcm(sentence: str) -> bytes:
            async with prefetch_sem:
                buf = bytearray()
                stream = synthesize_tts_pcm_stream(
                    sentence,
                    self._config.speaker,
                    voice_override=voice_override,
                    chunk_bytes=pb.chunk_bytes,
                    sample_rate_override=pb.effective_sample_rate,
                )
                try:
                    async for ch in stream:
                        if ch:
                            buf.extend(ch)
                finally:
                    aclose = getattr(stream, "aclose", None)
                    if callable(aclose):
                        try:
                            await aclose()
                        except Exception:
                            pass
                return bytes(buf)

        ack_task: asyncio.Task[None] | None = None
        if self._config.speaker.tool_ack_enabled:
            ack_phrase, intent_delay = build_conversational_ack(query)
            if ack_phrase:
                cfg_delay = float(self._config.speaker.tool_ack_delay_sec)
                ack_delay = (
                    min(intent_delay, cfg_delay)
                    if intent_delay is not None
                    else max(0.3, cfg_delay)
                )

                async def _ack_watchdog(phrase: str = ack_phrase, delay: float = ack_delay) -> None:
                    try:
                        await asyncio.wait_for(first_llm_ready.wait(), timeout=delay)
                        return
                    except asyncio.TimeoutError:
                        pass
                    if sess.interrupted or first_llm_ready.is_set():
                        return
                    logger.info("Speaking conversational ack while awaiting LLM/tool: %s", phrase)
                    await queue.put((phrase, None, "ack"))

                ack_task = asyncio.create_task(_ack_watchdog())

        async def producer() -> None:
            emitted = 0
            trace = CURRENT_TURN_TRACE.get()
            try:
                async for sentence in self._on_stream_query(query):
                    if not sentence:
                        continue
                    emitted += 1
                    first_llm_ready.set()
                    if ack_task is not None and not ack_task.done():
                        ack_task.cancel()
                    if trace is not None:
                        trace.mark_llm_first_sentence()
                        trace.sentence_count = emitted
                        trace.reply_parts.append(sentence)
                    pf_task: asyncio.Task[bytes] | None = None
                    # First sentence streams chunk-by-chunk for minimum TTFA; subsequent
                    # sentences prefetch in parallel so there is zero pause between sentences.
                    if use_continuous_stream and emitted >= 2 and self._config.speaker.tts_provider == "edge":
                        pf_task = asyncio.create_task(_prefetch_sentence_pcm(sentence))
                        all_prefetch_tasks.append(pf_task)
                    await queue.put((sentence, pf_task, "reply"))
                if trace is not None:
                    trace.mark_llm_done()
            except Exception as exc:
                first_llm_ready.set()
                if ack_task is not None and not ack_task.done():
                    ack_task.cancel()
                if trace is not None:
                    trace.status = "error"
                    trace.mark_llm_done()
                logger.exception("Streaming error for %r", query)
                await queue.put(("抱歉，刚才处理遇到了一点问题，请稍后再试。", None, "error"))
            finally:
                first_llm_ready.set()
                if ack_task is not None and not ack_task.done():
                    ack_task.cancel()
                await queue.put(None)

        prod_task = asyncio.create_task(producer())
        first_spoken = True
        native_stopped = False
        completed_normally = False
        trace = CURRENT_TURN_TRACE.get()
        try:
            while True:
                if sess.interrupted:
                    break
                item = await queue.get()
                if item is None or sess.interrupted:
                    break
                sentence, pf_task, kind = item
                if kind == "ack" and (first_llm_ready.is_set() or not queue.empty()):
                    # Skip obsolete ack when the real reply has already arrived in the queue.
                    continue
                prefetched_pcm: bytes | None = None
                if pf_task is not None:
                    try:
                        prefetched_pcm = await pf_task or None
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning("TTS prefetch failed for %r (%s); falling back to live stream", sentence[:20], exc)
                        prefetched_pcm = None
                if sess.interrupted:
                    break
                logger.info("Streaming TTS sentence (%s): %s", kind, sentence)
                self._record_spoken(sess, sentence)
                sess.mute_native_until = time.monotonic() + 8.0
                should_interrupt_native = not native_stopped
                if kind == "ack":
                    if prestart_play_task is not None:
                        res_list = await asyncio.gather(prestart_play_task, return_exceptions=True)
                        if res_list and res_list[0] is True:
                            native_stopped = True
                        prestart_play_task = None
                    ack_prewarm = await _take_prewarmed_tts()
                    # Start a fresh pre-warmed Edge-TTS WS for the upcoming real reply while ack plays!
                    if self._config.speaker.tts_provider == "edge":
                        async def _reopen_prewarm_tts() -> PrewarmedEdgeTTS | None:
                            try:
                                return await PrewarmedEdgeTTS.open(connect_timeout=5.0)
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                return None

                        prewarm_tts_task = asyncio.create_task(_reopen_prewarm_tts())
                        all_prewarm_tasks.append(prewarm_tts_task)
                    # Speak the quick acknowledgement and close its stream cleanly so it plays
                    # immediately without waiting for the slow tool/LLM call to finish.
                    ack_tok = CURRENT_TURN_TRACE.set(None)
                    t_ack = time.monotonic()
                    try:
                        await self.speak_sentence(
                            conn,
                            sentence,
                            interrupt=not native_stopped,
                            wait_response=True,
                            keep_stream_open=False,
                            prefetched_pcm=None,
                            prewarmed_tts=ack_prewarm,
                        )
                    finally:
                        CURRENT_TURN_TRACE.reset(ack_tok)
                        if ack_prewarm is not None:
                            await ack_prewarm.close()
                    native_stopped = True
                    if trace is not None:
                        trace.ack_ms = round((time.monotonic() - t_ack) * 1000.0, 1)
                    continue
                if prestart_play_task is not None:
                    res_list = await asyncio.gather(prestart_play_task, return_exceptions=True)
                    if res_list and res_list[0] is True:
                        native_stopped = True
                    prestart_play_task = None
                should_interrupt_native = not native_stopped
                if kind == "error":
                    # Detach trace context so speaking the error notice never pollutes turn TTS/TTFA metrics.
                    err_tok = CURRENT_TURN_TRACE.set(None)
                    try:
                        await self.speak_sentence(
                            conn,
                            sentence,
                            interrupt=should_interrupt_native,
                            wait_response=True,
                            keep_stream_open=use_continuous_stream,
                            prefetched_pcm=None,
                        )
                    finally:
                        CURRENT_TURN_TRACE.reset(err_tok)
                else:
                    reply_prewarm = await _take_prewarmed_tts() if first_spoken and prefetched_pcm is None else None
                    if first_spoken and trace is not None and trace.tts_started_mono is None:
                        trace.tts_started_mono = time.monotonic()
                    try:
                        await self.speak_sentence(
                            conn,
                            sentence,
                            interrupt=should_interrupt_native,
                            wait_response=True,
                            keep_stream_open=use_continuous_stream,
                            prefetched_pcm=prefetched_pcm,
                            prewarmed_tts=reply_prewarm,
                        )
                    finally:
                        if reply_prewarm is not None:
                            await reply_prewarm.close()
                native_stopped = True
                first_spoken = False
            if use_continuous_stream and not first_spoken and not sess.interrupted:
                await pb.finish_stream()
            elif use_continuous_stream and first_spoken and not sess.interrupted and pb.stream_open:
                # If speculative start_play opened a stream but producer yielded zero sentences, close it cleanly.
                await pb.abort_stream(unmute=True)
            if trace is not None and trace.tts_started_mono is not None:
                trace.tts_done_mono = time.monotonic()
            completed_normally = True
            await prod_task
        except asyncio.CancelledError:
            if trace is not None:
                if trace.status == "ok":
                    trace.status = "interrupted"
                if trace.tts_started_mono is not None and trace.tts_done_mono is None:
                    trace.tts_done_mono = time.monotonic()
            logger.info("Streaming reply cancelled due to barge-in")
        finally:
            if use_continuous_stream and not completed_normally and not sess.interrupted:
                # Mid-stream exception (not barge-in): close any kept-open stream immediately
                # without padding/draining, and unmute 1ch VAD cleanly.
                if pb.stream_open:
                    await pb.abort_stream(unmute=True)
            if ack_task is not None and not ack_task.done():
                ack_task.cancel()
            if prestart_play_task is not None and not prestart_play_task.done():
                prestart_play_task.cancel()
            for t in all_prewarm_tasks:
                if not t.done():
                    t.cancel()
            for t in all_prefetch_tasks:
                if not t.done():
                    t.cancel()
            if not prod_task.done():
                prod_task.cancel()
            cleanup_tasks: list[asyncio.Task[Any]] = [prod_task, *all_prefetch_tasks, *all_prewarm_tasks]
            if ack_task is not None:
                cleanup_tasks.append(ack_task)
            if prestart_play_task is not None:
                cleanup_tasks.append(prestart_play_task)
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, PrewarmedEdgeTTS):
                    await res.close()
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

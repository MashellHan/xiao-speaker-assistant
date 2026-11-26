"""Persistent thread-safe conversation history store (0600 permissions, max 500 turns)."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

VALID_SOURCES = frozenset({"voice", "cli", "bridge"})
VALID_TRACE_STATUSES = frozenset({"ok", "interrupted", "error"})
VALID_TRANSPORTS = frozenset({"wan_relay", "lan", "bridge", "cli"})
MAX_RECENT_TRACES = 10


@dataclass
class TurnTraceBuilder:
    """Mutable per-turn timing collector bound to a ContextVar across ASR -> LLM -> TTS -> Playback."""

    turn_started_mono: float = field(default_factory=time.monotonic)
    query_started_mono: float = 0.0
    asr_ms: float | None = None
    speaker_id_ms: float | None = None
    llm_first_sentence_mono: float | None = None
    llm_done_mono: float | None = None
    start_play_ms: float | None = None
    tts_first_chunk_ms: float | None = None
    ack_ms: float | None = None
    first_audio_sent_mono: float | None = None
    tts_started_mono: float | None = None
    tts_done_mono: float | None = None
    model: str = ""
    voice: str = ""
    transport: str = "lan"
    sample_rate: int = 24000
    sentence_count: int = 0
    barge_in: bool = False
    status: str = "ok"
    reply_parts: list[str] = field(default_factory=list)

    def mark_llm_first_sentence(self, now: float | None = None) -> None:
        if self.llm_first_sentence_mono is None:
            self.llm_first_sentence_mono = time.monotonic() if now is None else now

    def mark_llm_done(self, now: float | None = None) -> None:
        self.llm_done_mono = time.monotonic() if now is None else now

    def mark_start_play(self, duration_ms: float) -> None:
        if self.start_play_ms is None:
            self.start_play_ms = round(max(0.0, float(duration_ms)), 1)

    def mark_tts_first_chunk(self, duration_ms: float) -> None:
        if self.tts_first_chunk_ms is None:
            self.tts_first_chunk_ms = round(max(0.0, float(duration_ms)), 1)

    def mark_first_audio_sent(self, now: float | None = None) -> None:
        if self.first_audio_sent_mono is None:
            self.first_audio_sent_mono = time.monotonic() if now is None else now

    def build_breakdown(self, finished_mono: float | None = None) -> dict[str, float]:
        end = time.monotonic() if finished_mono is None else finished_mono
        q_start = self.query_started_mono or self.turn_started_mono
        out: dict[str, float] = {}
        if self.asr_ms is not None:
            out["asr_ms"] = round(max(0.0, self.asr_ms), 1)
        if self.speaker_id_ms is not None:
            out["speaker_id_ms"] = round(max(0.0, self.speaker_id_ms), 1)
        pre_query_ms = max(0.0, (q_start - self.turn_started_mono) * 1000.0)
        known_pre_ms = (out.get("asr_ms") or 0.0) + (out.get("speaker_id_ms") or 0.0)
        queue_gap_ms = pre_query_ms - known_pre_ms
        if queue_gap_ms >= 10.0:
            out["queue_ms"] = round(queue_gap_ms, 1)
        if self.llm_first_sentence_mono is not None:
            out["llm_ttft_ms"] = round(max(0.0, (self.llm_first_sentence_mono - q_start) * 1000.0), 1)
        if self.llm_done_mono is not None:
            out["llm_total_ms"] = round(max(0.0, (self.llm_done_mono - q_start) * 1000.0), 1)
        if self.start_play_ms is not None:
            out["start_play_ms"] = round(max(0.0, self.start_play_ms), 1)
        if self.tts_first_chunk_ms is not None:
            out["tts_first_chunk_ms"] = round(max(0.0, self.tts_first_chunk_ms), 1)
        if self.ack_ms is not None:
            out["ack_ms"] = round(max(0.0, self.ack_ms), 1)
        if self.tts_started_mono is not None:
            tts_end = self.tts_done_mono or end
            out["tts_total_ms"] = round(max(0.0, (tts_end - self.tts_started_mono) * 1000.0), 1)
        if self.first_audio_sent_mono is not None:
            out["ttfa_ms"] = round(max(0.0, (self.first_audio_sent_mono - self.turn_started_mono) * 1000.0), 1)
        elif "llm_ttft_ms" in out:
            pre = (out.get("asr_ms") or 0.0) + (out.get("speaker_id_ms") or 0.0)
            out["ttfa_ms"] = round(pre + out["llm_ttft_ms"], 1)
        out["total_ms"] = round(max(0.0, (end - self.turn_started_mono) * 1000.0), 1)
        return out


CURRENT_TURN_TRACE: ContextVar[TurnTraceBuilder | None] = ContextVar("xsa_current_turn_trace", default=None)


@dataclass
class TurnTrace:
    id: str
    timestamp: float
    speaker_name: str
    speaker_role: str
    query: str
    reply: str
    model: str
    voice: str
    transport: str = "lan"
    sample_rate: int = 24000
    sentence_count: int = 0
    barge_in: bool = False
    status: str = "ok"
    breakdown_ms: dict[str, float] = field(default_factory=dict)


class TurnTraceStore:
    """Thread-safe JSON trace store strictly bounded to the most recent `max_entries` (default 10) turns."""

    def __init__(
        self,
        path: str | Path = "~/.local/share/xiao-speaker-assistant/traces.json",
        *,
        max_entries: int = MAX_RECENT_TRACES,
    ) -> None:
        self.path = Path(path).expanduser()
        self._max_entries = max(1, min(MAX_RECENT_TRACES, int(max_entries)))
        self._lock = threading.RLock()
        self._entries: list[TurnTrace] = []
        self._load()

    def _sanitize_breakdown(self, raw: Any) -> dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        allowed = (
            "asr_ms",
            "speaker_id_ms",
            "queue_ms",
            "llm_ttft_ms",
            "llm_total_ms",
            "start_play_ms",
            "tts_first_chunk_ms",
            "ack_ms",
            "tts_total_ms",
            "ttfa_ms",
            "total_ms",
        )
        out: dict[str, float] = {}
        for key in allowed:
            val = raw.get(key)
            if isinstance(val, bool) or val is None:
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(num) and num >= 0.0:
                out[key] = round(num, 1)
        return out

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._entries = []
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._entries = []
                return
            if not isinstance(raw, list):
                self._entries = []
                return
            loaded: list[TurnTrace] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                query = str(item.get("query") or "").strip()
                reply = str(item.get("reply") or "").strip()
                if not query and not reply:
                    continue
                status = str(item.get("status") or "ok").strip().lower()
                if status not in VALID_TRACE_STATUSES:
                    status = "ok"
                transport = str(item.get("transport") or "lan").strip().lower()
                if transport not in VALID_TRANSPORTS:
                    transport = "lan"
                try:
                    sample_rate = max(0, int(item.get("sample_rate") or 0))
                except (TypeError, ValueError):
                    sample_rate = 0
                try:
                    sentence_count = max(0, int(item.get("sentence_count") or 0))
                except (TypeError, ValueError):
                    sentence_count = 0
                loaded.append(
                    TurnTrace(
                        id=str(item.get("id") or uuid.uuid4().hex[:12]),
                        timestamp=float(item.get("timestamp") or 0.0),
                        speaker_name=str(item.get("speaker_name") or "未知"),
                        speaker_role=str(item.get("speaker_role") or ""),
                        query=query[:160],
                        reply=reply[:240],
                        model=str(item.get("model") or ""),
                        voice=str(item.get("voice") or ""),
                        transport=transport,
                        sample_rate=sample_rate,
                        sentence_count=sentence_count,
                        barge_in=bool(item.get("barge_in", False)),
                        status=status,
                        breakdown_ms=self._sanitize_breakdown(item.get("breakdown_ms")),
                    )
                )
            self._entries = loaded[-self._max_entries :]

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        tmp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
        payload = json.dumps([asdict(e) for e in self._entries], ensure_ascii=False, indent=2) + "\n"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(payload.encode("utf-8"))
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, self.path)
        os.chmod(self.path, 0o600)

    def record(
        self,
        *,
        query: str,
        reply: str,
        breakdown_ms: dict[str, float],
        model: str = "",
        voice: str = "",
        speaker_name: str = "未知",
        speaker_role: str = "",
        transport: str = "lan",
        sample_rate: int = 24000,
        sentence_count: int = 0,
        barge_in: bool = False,
        status: str = "ok",
        timestamp: float | None = None,
    ) -> TurnTrace:
        clean_status = status.strip().lower()
        if clean_status not in VALID_TRACE_STATUSES:
            clean_status = "ok"
        clean_transport = transport.strip().lower()
        if clean_transport not in VALID_TRANSPORTS:
            clean_transport = "lan"
        entry = TurnTrace(
            id=uuid.uuid4().hex[:12],
            timestamp=float(time.time() if timestamp is None else timestamp),
            speaker_name=speaker_name.strip() or "未知",
            speaker_role=speaker_role.strip(),
            query=query.strip()[:160],
            reply=reply.strip()[:240],
            model=model.strip(),
            voice=voice.strip(),
            transport=clean_transport,
            sample_rate=max(0, int(sample_rate)),
            sentence_count=max(0, int(sentence_count)),
            barge_in=bool(barge_in),
            status=clean_status,
            breakdown_ms=self._sanitize_breakdown(breakdown_ms),
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]
            self._persist()
            return entry

    def list(self, *, limit: int = MAX_RECENT_TRACES) -> list[TurnTrace]:
        capped = max(1, min(self._max_entries, int(limit)))
        with self._lock:
            return list(reversed(self._entries))[:capped]

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._persist()
            return count


@dataclass
class HistoryEntry:
    id: str
    timestamp: float
    speaker_name: str
    speaker_role: str
    query: str
    reply: str
    latency_ms: float
    model: str
    voice: str
    source: str = "voice"


class ConversationHistoryStore:
    """Thread-safe JSON conversation history store with 0600 permissions."""

    def __init__(
        self,
        path: str | Path = "~/.local/share/xiao-speaker-assistant/history.json",
        *,
        max_entries: int = 500,
    ) -> None:
        self.path = Path(path).expanduser()
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._entries: list[HistoryEntry] = []
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._entries = []
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._entries = []
                return
            if not isinstance(raw, list):
                self._entries = []
                return
            loaded: list[HistoryEntry] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                query = str(item.get("query") or "").strip()
                reply = str(item.get("reply") or "").strip()
                if not query and not reply:
                    continue
                source = str(item.get("source") or "voice").strip().lower()
                if source not in VALID_SOURCES:
                    source = "voice"
                loaded.append(
                    HistoryEntry(
                        id=str(item.get("id") or uuid.uuid4().hex[:12]),
                        timestamp=float(item.get("timestamp") or 0.0),
                        speaker_name=str(item.get("speaker_name") or "未知"),
                        speaker_role=str(item.get("speaker_role") or ""),
                        query=query,
                        reply=reply,
                        latency_ms=max(0.0, float(item.get("latency_ms") or 0.0)),
                        model=str(item.get("model") or ""),
                        voice=str(item.get("voice") or ""),
                        source=source,
                    )
                )
            self._entries = loaded[-self._max_entries :]

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        tmp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
        payload = json.dumps([asdict(e) for e in self._entries], ensure_ascii=False, indent=2) + "\n"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(payload.encode("utf-8"))
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, self.path)
        os.chmod(self.path, 0o600)

    def record(
        self,
        query: str,
        reply: str,
        *,
        latency_ms: float = 0.0,
        model: str = "",
        voice: str = "",
        speaker_name: str = "未知",
        speaker_role: str = "",
        source: str = "voice",
        timestamp: float | None = None,
    ) -> HistoryEntry:
        clean_source = source.strip().lower() or "voice"
        if clean_source not in VALID_SOURCES:
            raise ValueError(f"Unsupported history source {source!r}; expected one of {sorted(VALID_SOURCES)}")
        clean_speaker = speaker_name.strip() or "未知"
        entry = HistoryEntry(
            id=uuid.uuid4().hex[:12],
            timestamp=float(time.time() if timestamp is None else timestamp),
            speaker_name=clean_speaker,
            speaker_role=speaker_role.strip(),
            query=query.strip(),
            reply=reply.strip(),
            latency_ms=round(max(0.0, float(latency_ms)), 1),
            model=model.strip(),
            voice=voice.strip(),
            source=clean_source,
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]
            self._persist()
            return entry

    def list(self, *, limit: int = 50, speaker: str = "") -> list[HistoryEntry]:
        capped = max(1, min(self._max_entries, int(limit)))
        filter_text = speaker.strip().lower()
        with self._lock:
            items = list(reversed(self._entries))
            if filter_text:
                items = [
                    e
                    for e in items
                    if filter_text in e.speaker_name.lower() or filter_text in e.speaker_role.lower()
                ]
            return items[:capped]

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._persist()
            return count

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._entries)
            if total == 0:
                return {
                    "total_turns": 0,
                    "avg_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "turns_today": 0,
                    "speaker_breakdown": {},
                    "model_breakdown": {},
                }

            today = datetime.now().date()
            latencies = sorted(e.latency_ms for e in self._entries)
            avg_lat = round(sum(latencies) / total, 1)
            p95_idx = min(total - 1, max(0, math.ceil(0.95 * total) - 1))
            p95_lat = round(latencies[p95_idx], 1)

            turns_today = 0
            speaker_breakdown: dict[str, int] = {}
            model_breakdown: dict[str, int] = {}
            for entry in self._entries:
                if entry.timestamp > 0 and datetime.fromtimestamp(entry.timestamp).date() == today:
                    turns_today += 1
                spk = entry.speaker_name or "未知"
                speaker_breakdown[spk] = speaker_breakdown.get(spk, 0) + 1
                mdl = entry.model or "unknown"
                model_breakdown[mdl] = model_breakdown.get(mdl, 0) + 1

            return {
                "total_turns": total,
                "avg_latency_ms": avg_lat,
                "p95_latency_ms": p95_lat,
                "turns_today": turns_today,
                "speaker_breakdown": speaker_breakdown,
                "model_breakdown": model_breakdown,
            }

    def daily_turns(self, *, days: int = 14) -> list[dict[str, Any]]:
        """Return per-day turn counts, average latency, and P95 latency for the last `days` local dates."""
        n_days = max(1, min(90, int(days)))
        today = datetime.now().date()
        date_keys = [(today - timedelta(days=i)).isoformat() for i in range(n_days - 1, -1, -1)]
        buckets: dict[str, dict[str, Any]] = {
            d: {"date": d, "turns": 0, "latencies": []} for d in date_keys
        }
        with self._lock:
            for entry in self._entries:
                if entry.timestamp <= 0:
                    continue
                try:
                    ts_sec = entry.timestamp / 1000.0 if entry.timestamp > 1e11 else entry.timestamp
                    d_str = datetime.fromtimestamp(ts_sec).date().isoformat()
                except (OverflowError, OSError, ValueError):
                    continue
                if d_str in buckets:
                    buckets[d_str]["turns"] += 1
                    buckets[d_str]["latencies"].append(entry.latency_ms)
        out: list[dict[str, Any]] = []
        for d in date_keys:
            lats: list[float] = sorted(buckets[d]["latencies"])
            n = len(lats)
            avg_ms = round(sum(lats) / n, 1) if n > 0 else 0.0
            p95_ms = round(lats[min(n - 1, max(0, math.ceil(0.95 * n) - 1))], 1) if n > 0 else 0.0
            out.append(
                {
                    "date": d,
                    "turns": buckets[d]["turns"],
                    "avg_latency_ms": avg_ms,
                    "p95_latency_ms": p95_ms,
                }
            )
        return out


def read_local_opencode_usage(agent_home: str | Path, *, days: int = 14) -> dict[str, Any]:
    """Read token counts and cost totals + daily trend from isolated OpenCode SQLite DB (read-only)."""
    n_days = max(1, min(90, int(days)))
    today = datetime.now().date()
    date_keys = [(today - timedelta(days=i)).isoformat() for i in range(n_days - 1, -1, -1)]
    daily_map: dict[str, dict[str, Any]] = {
        d: {
            "date": d,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cache_read_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
        for d in date_keys
    }
    empty_result: dict[str, Any] = {
        "available": False,
        "total_calls": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "total_cost_usd": 0.0,
        "today_tokens": 0,
        "today_cost_usd": 0.0,
        "daily": list(daily_map.values()),
    }

    db_path = Path(agent_home).expanduser() / "opencode.db"
    if not db_path.is_file():
        return empty_result

    def _safe_int(val: Any) -> int:
        if isinstance(val, bool) or val is None:
            return 0
        try:
            return max(0, int(val))
        except (TypeError, ValueError):
            return 0

    def _safe_float(val: Any) -> float:
        if isinstance(val, bool) or val is None:
            return 0.0
        try:
            out = float(val)
            return out if math.isfinite(out) and out >= 0.0 else 0.0
        except (TypeError, ValueError):
            return 0.0

    conn: sqlite3.Connection | None = None
    try:
        ro_uri = f"{db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(ro_uri, uri=True, timeout=1.5)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='session_message'"
        )
        if cur.fetchone() is None:
            return empty_result

        cur.execute(
            "SELECT time_created, data FROM session_message WHERE type = 'assistant'"
        )
        total_calls = 0
        total_in = 0
        total_out = 0
        total_reason = 0
        total_cache = 0
        total_cost = 0.0

        for ts_ms, raw_data in cur:
            if not raw_data:
                continue
            try:
                obj = json.loads(raw_data)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            tokens_obj = obj.get("tokens")
            cost_val = obj.get("cost")
            tokens_dict = tokens_obj if isinstance(tokens_obj, dict) else {}
            if not tokens_dict and cost_val is None:
                continue

            t_in = _safe_int(tokens_dict.get("input"))
            t_out = _safe_int(tokens_dict.get("output"))
            t_reason = _safe_int(tokens_dict.get("reasoning"))
            cache_obj = tokens_dict.get("cache")
            t_cache = _safe_int(cache_obj.get("read")) if isinstance(cache_obj, dict) else 0
            c_usd = _safe_float(cost_val)
            t_sum = t_in + t_out + t_reason + t_cache

            total_calls += 1
            total_in += t_in
            total_out += t_out
            total_reason += t_reason
            total_cache += t_cache
            total_cost += c_usd

            if isinstance(ts_ms, (int, float)) and not isinstance(ts_ms, bool) and ts_ms > 0:
                try:
                    ts_sec = ts_ms / 1000.0 if ts_ms > 1e11 else float(ts_ms)
                    d_str = datetime.fromtimestamp(ts_sec).date().isoformat()
                except (OverflowError, OSError, ValueError):
                    continue
                if d_str in daily_map:
                    bucket = daily_map[d_str]
                    bucket["calls"] += 1
                    bucket["input_tokens"] += t_in
                    bucket["output_tokens"] += t_out
                    bucket["reasoning_tokens"] += t_reason
                    bucket["cache_read_tokens"] += t_cache
                    bucket["total_tokens"] += t_sum
                    bucket["cost_usd"] += c_usd

        daily_list = []
        for d in date_keys:
            b = daily_map[d]
            daily_list.append({**b, "cost_usd": round(b["cost_usd"], 6)})

        today_bucket = daily_map[today.isoformat()]
        return {
            "available": True,
            "total_calls": total_calls,
            "total_tokens": total_in + total_out + total_reason + total_cache,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "reasoning_tokens": total_reason,
            "cache_read_tokens": total_cache,
            "total_cost_usd": round(total_cost, 6),
            "today_tokens": today_bucket["total_tokens"],
            "today_cost_usd": round(today_bucket["cost_usd"], 6),
            "daily": daily_list,
        }
    except sqlite3.Error:
        return empty_result
    finally:
        if conn is not None:
            conn.close()

"""Persistent thread-safe conversation history store (0600 permissions, max 500 turns)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

VALID_SOURCES = frozenset({"voice", "cli", "bridge"})


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
        tmp_path = self.path.with_suffix(".tmp")
        payload = json.dumps([asdict(e) for e in self._entries], ensure_ascii=False, indent=2) + "\n"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
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

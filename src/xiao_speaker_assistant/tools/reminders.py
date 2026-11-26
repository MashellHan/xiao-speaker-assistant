"""Persistent timer, alarm, and daily morning-briefing scheduler (0600 JSON)."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


@dataclass
class ReminderEntry:
    id: str
    message: str
    kind: str = "reminder"  # "reminder" | "morning_briefing"
    trigger_at: float | None = None
    daily_hhmm: str | None = None
    last_fired_date: str | None = None
    created_at: float = 0.0


def is_quiet_hours(
    local_now: datetime,
    quiet_start_hhmm: str,
    quiet_end_hhmm: str,
) -> bool:
    if not _HHMM_RE.match(quiet_start_hhmm) or not _HHMM_RE.match(quiet_end_hhmm):
        return False
    if quiet_start_hhmm == quiet_end_hhmm:
        return False
    cur = local_now.strftime("%H:%M")
    if quiet_start_hhmm < quiet_end_hhmm:
        return quiet_start_hhmm <= cur < quiet_end_hhmm
    return cur >= quiet_start_hhmm or cur < quiet_end_hhmm


class ReminderStore:
    """Thread-safe persistent reminder store with 0600 file permissions."""

    def __init__(
        self,
        path: str | Path = "~/.local/share/xiao-speaker-assistant/reminders.json",
        *,
        max_entries: int = 100,
    ) -> None:
        self.path = Path(path).expanduser()
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._entries: list[ReminderEntry] = []
        self._retry_after: dict[str, float] = {}
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
            loaded: list[ReminderEntry] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                msg = str(item.get("message") or "").strip()
                if not msg:
                    continue
                raw_trigger = item.get("trigger_at")
                parsed_trigger: float | None = None
                parsed_daily = str(item["daily_hhmm"]) if item.get("daily_hhmm") else None
                if raw_trigger is not None:
                    if isinstance(raw_trigger, str) and _HHMM_RE.match(raw_trigger.strip()):
                        if parsed_daily is None:
                            parsed_daily = raw_trigger.strip()
                    else:
                        try:
                            parsed_trigger = float(raw_trigger)
                        except (TypeError, ValueError):
                            parsed_trigger = None
                loaded.append(
                    ReminderEntry(
                        id=str(item.get("id") or uuid.uuid4().hex[:8]),
                        message=msg,
                        kind=str(item.get("kind") or "reminder"),
                        trigger_at=parsed_trigger,
                        daily_hhmm=parsed_daily,
                        last_fired_date=str(item["last_fired_date"]) if item.get("last_fired_date") else None,
                        created_at=float(item.get("created_at") or 0.0),
                    )
                )
            self._entries = loaded[-self._max_entries :]

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        tmp = self.path.with_suffix(".tmp")
        payload = json.dumps([asdict(e) for e in self._entries], ensure_ascii=False, indent=2) + "\n"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)
        os.chmod(self.path, 0o600)

    def add(
        self,
        message: str,
        *,
        delay_seconds: int | float | None = None,
        trigger_at_iso: str | None = None,
        daily_hhmm: str | None = None,
        kind: str = "reminder",
        now_ts: float | None = None,
        local_now: datetime | None = None,
    ) -> ReminderEntry:
        clean_msg = re.sub(r"\s+", " ", message.strip())
        if not clean_msg or len(clean_msg) > 200:
            raise ValueError("Reminder 'message' must be 1..200 characters.")
        clean_kind = kind if kind in {"reminder", "morning_briefing"} else "reminder"
        ts_now = time.time() if now_ts is None else now_ts
        dt_now = datetime.now() if local_now is None else local_now

        trigger_at: float | None = None
        norm_daily: str | None = None
        last_fired_date: str | None = None

        if daily_hhmm:
            d = daily_hhmm.strip()
            if not _HHMM_RE.match(d):
                raise ValueError(f"daily_hhmm must be HH:MM (24h), got {daily_hhmm!r}")
            norm_daily = d
            # If today's scheduled minute has already passed, schedule first firing for tomorrow.
            if dt_now.strftime("%H:%M") >= norm_daily:
                last_fired_date = dt_now.strftime("%Y-%m-%d")
        elif delay_seconds is not None:
            delay = float(delay_seconds)
            if delay <= 0 or delay > 86400 * 30:
                raise ValueError("delay_seconds must be between 1 and 2592000 (30 days).")
            trigger_at = ts_now + delay
        elif trigger_at_iso:
            parsed = datetime.fromisoformat(trigger_at_iso.strip())
            trigger_at = parsed.timestamp()
            if trigger_at <= ts_now - 60:
                raise ValueError("trigger_at_iso must not be in the past.")
        else:
            raise ValueError("Provide delay_seconds, trigger_at_iso, or daily_hhmm.")

        entry = ReminderEntry(
            id=uuid.uuid4().hex[:8],
            message=clean_msg,
            kind=clean_kind,
            trigger_at=trigger_at,
            daily_hhmm=norm_daily,
            last_fired_date=last_fired_date,
            created_at=ts_now,
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]
            self._persist()
        return entry

    def list_entries(self) -> list[ReminderEntry]:
        with self._lock:
            return list(self._entries)

    def cancel(self, id_or_keyword: str, *, exact_id: bool = False) -> int:
        target = id_or_keyword.strip().lower()
        if not target:
            return 0
        with self._lock:
            before = len(self._entries)
            if exact_id:
                self._entries = [e for e in self._entries if e.id.lower() != target]
            else:
                self._entries = [
                    e
                    for e in self._entries
                    if e.id.lower() != target and target not in e.message.lower()
                ]
            removed = before - len(self._entries)
            if removed:
                self._persist()
            return removed

    def peek_due(
        self,
        *,
        now_ts: float | None = None,
        local_now: datetime | None = None,
        quiet_start_hhmm: str = "23:00",
        quiet_end_hhmm: str = "07:00",
        max_late_sec: float = 900.0,
    ) -> list[ReminderEntry]:
        ts_now = time.time() if now_ts is None else now_ts
        dt_now = datetime.now() if local_now is None else local_now
        today_str = dt_now.strftime("%Y-%m-%d")
        cur_hhmm = dt_now.strftime("%H:%M")
        cur_minutes = dt_now.hour * 60 + dt_now.minute
        quiet = is_quiet_hours(dt_now, quiet_start_hhmm, quiet_end_hhmm)

        due: list[ReminderEntry] = []
        expired_changed = False
        with self._lock:
            remaining: list[ReminderEntry] = []
            for entry in self._entries:
                retry_until = self._retry_after.get(entry.id, 0.0)
                if entry.daily_hhmm:
                    hh, mm = (int(x) for x in entry.daily_hhmm.split(":"))
                    target_minutes = hh * 60 + mm
                    in_window = 0 <= (cur_minutes - target_minutes) <= 10
                    if (
                        not quiet
                        and ts_now >= retry_until
                        and entry.last_fired_date != today_str
                        and cur_hhmm >= entry.daily_hhmm
                        and in_window
                    ):
                        due.append(entry)
                    remaining.append(entry)
                elif entry.trigger_at is not None and ts_now >= entry.trigger_at:
                    if ts_now - entry.trigger_at > max_late_sec:
                        self._retry_after.pop(entry.id, None)
                        expired_changed = True
                        continue
                    if ts_now >= retry_until:
                        due.append(entry)
                    remaining.append(entry)
                else:
                    remaining.append(entry)
            if expired_changed:
                self._entries = remaining
                self._persist()
        return due

    def defer_retry(
        self,
        entry_id: str,
        *,
        delay_sec: float = 10.0,
        now_ts: float | None = None,
    ) -> None:
        ts_now = time.time() if now_ts is None else now_ts
        with self._lock:
            self._retry_after[entry_id] = ts_now + max(0.0, delay_sec)

    def mark_delivered(
        self,
        entry_id: str,
        *,
        local_now: datetime | None = None,
    ) -> bool:
        dt_now = datetime.now() if local_now is None else local_now
        today_str = dt_now.strftime("%Y-%m-%d")
        with self._lock:
            self._retry_after.pop(entry_id, None)
            new_entries: list[ReminderEntry] = []
            found = False
            for entry in self._entries:
                if entry.id != entry_id:
                    new_entries.append(entry)
                    continue
                found = True
                if entry.daily_hhmm:
                    entry.last_fired_date = today_str
                    new_entries.append(entry)
                # One-shot reminders are removed once delivered.
            if found:
                self._entries = new_entries
                self._persist()
            return found

    def pop_due(
        self,
        *,
        now_ts: float | None = None,
        local_now: datetime | None = None,
        quiet_start_hhmm: str = "23:00",
        quiet_end_hhmm: str = "07:00",
    ) -> list[ReminderEntry]:
        due = self.peek_due(
            now_ts=now_ts,
            local_now=local_now,
            quiet_start_hhmm=quiet_start_hhmm,
            quiet_end_hhmm=quiet_end_hhmm,
        )
        for entry in due:
            self.mark_delivered(entry.id, local_now=local_now)
        return due


class ReminderScheduler:
    """Async background loop checking `ReminderStore` and dispatching proactive announcements."""

    def __init__(
        self,
        store: ReminderStore,
        on_due: Callable[[ReminderEntry], Awaitable[bool]],
        *,
        quiet_start_hhmm: str = "23:00",
        quiet_end_hhmm: str = "07:00",
        poll_interval: float = 1.0,
        retry_backoff_sec: float = 10.0,
        enabled: bool = True,
    ) -> None:
        self._store = store
        self._on_due = on_due
        self.quiet_start_hhmm = quiet_start_hhmm
        self.quiet_end_hhmm = quiet_end_hhmm
        self.enabled = enabled
        self._poll_interval = poll_interval
        self._retry_backoff_sec = retry_backoff_sec
        self._task: asyncio.Task[None] | None = None

    async def tick_once(
        self, *, now_ts: float | None = None, local_now: datetime | None = None
    ) -> int:
        if not self.enabled:
            return 0
        due_entries = self._store.peek_due(
            now_ts=now_ts,
            local_now=local_now,
            quiet_start_hhmm=self.quiet_start_hhmm,
            quiet_end_hhmm=self.quiet_end_hhmm,
        )
        dispatched = 0
        for entry in due_entries:
            try:
                ok = await self._on_due(entry)
            except Exception:
                logger.exception("Failed delivering proactive reminder %s", entry.id)
                self._store.defer_retry(
                    entry.id, delay_sec=self._retry_backoff_sec, now_ts=now_ts
                )
                continue
            if ok:
                self._store.mark_delivered(entry.id, local_now=local_now)
                dispatched += 1
            else:
                self._store.defer_retry(
                    entry.id, delay_sec=self._retry_backoff_sec, now_ts=now_ts
                )
        return dispatched

    def start(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
        return self._task

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            await self.tick_once()

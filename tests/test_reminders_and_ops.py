from __future__ import annotations

import asyncio
from datetime import datetime
import json
from pathlib import Path

import pytest

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.tools import (
    McpServer,
    ReminderScheduler,
    ReminderStore,
    is_quiet_hours,
)
from xiao_speaker_assistant.voice.ws_server import SpeakerGateway


def test_reminder_store_countdown_daily_and_quiet_hours(tmp_path: Path) -> None:
    rem_path = tmp_path / "reminders.json"
    store = ReminderStore(rem_path)

    # 1. Countdown timer
    e1 = store.add("看锅", delay_seconds=60, now_ts=1000.0)
    assert (rem_path.stat().st_mode & 0o777) == 0o600
    assert store.pop_due(now_ts=1059.0, local_now=datetime(2026, 11, 25, 10, 0)) == []
    due = store.pop_due(now_ts=1060.0, local_now=datetime(2026, 11, 25, 10, 0))
    assert [e.id for e in due] == [e1.id]
    assert store.list_entries() == []

    # 2. Daily briefing scheduled for 08:30 when created at 10:00 -> waits until tomorrow 08:30
    e2 = store.add(
        "每日早报",
        daily_hhmm="08:30",
        kind="morning_briefing",
        now_ts=2000.0,
        local_now=datetime(2026, 11, 25, 10, 0),
    )
    assert e2.last_fired_date == "2026-11-25"
    assert store.pop_due(now_ts=2100.0, local_now=datetime(2026, 11, 25, 10, 5)) == []

    # Fires next morning at 08:30, once per day
    due_next = store.pop_due(now_ts=90000.0, local_now=datetime(2026, 11, 26, 8, 30))
    assert len(due_next) == 1 and due_next[0].kind == "morning_briefing"
    assert store.pop_due(now_ts=90010.0, local_now=datetime(2026, 11, 26, 8, 31)) == []

    # 3. Quiet hours wrap-around check
    assert is_quiet_hours(datetime(2026, 11, 26, 23, 30), "23:00", "07:00") is True
    assert is_quiet_hours(datetime(2026, 11, 26, 6, 59), "23:00", "07:00") is True
    assert is_quiet_hours(datetime(2026, 11, 26, 7, 0), "23:00", "07:00") is False


def test_mcp_reminder_tools_and_proactive_speak(tmp_path: Path) -> None:
    store = ReminderStore(tmp_path / "reminders.json")
    srv = McpServer(AppConfig(), reminder_store=store, include_local_tools=True)

    res_set = asyncio.run(
        srv.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "reminder_set",
                    "arguments": {"message": "喝水", "delay_seconds": 30},
                },
            }
        )
    )
    assert res_set is not None and res_set["result"]["isError"] is False
    data_set = json.loads(res_set["result"]["content"][0]["text"])
    assert data_set["status"] == "scheduled"

    res_list = asyncio.run(
        srv.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "reminder_list", "arguments": {}}})
    )
    assert res_list is not None
    assert json.loads(res_list["result"]["content"][0]["text"])["count"] == 1

    # Proactive scheduler delivers via SpeakerGateway.proactive_speak
    sent_cmds: list[str] = []

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw) -> None:
            if isinstance(raw, (bytes, bytearray)):
                return
            msg = json.loads(raw)
            req = msg.get("Request") or {}
            cmd = str(req.get("command") or "")
            sent_cmds.append(cmd)
            if cmd in {"start_play", "stop_play"} and isinstance(req.get("id"), str):
                rid = req["id"]
                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.create_task(
                        gw._handle_text_message(self, json.dumps({"Response": {"id": rid, "code": 0}}))
                    )
                )

    async def dummy(q: str) -> str:
        return q

    cfg = AppConfig()
    cfg.speaker.tts_provider = "say"
    gw = SpeakerGateway(cfg, on_query=dummy)

    async def run() -> None:
        conn = FakeConn()
        gw._connections.add(conn)  # type: ignore[arg-type]
        sched = ReminderScheduler(store, lambda e: gw.proactive_speak(f"提醒：{e.message}"))
        count = await sched.tick_once(now_ts=store.list_entries()[0].trigger_at + 1)  # type: ignore[operator]
        assert count == 1

    asyncio.run(run())
    assert "start_play" in sent_cmds and "stop_play" in sent_cmds

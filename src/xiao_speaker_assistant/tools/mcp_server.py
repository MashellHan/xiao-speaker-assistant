"""Stdio & Streamable-HTTP JSON-RPC 2.0 MCP server for Outlook, Codex, Xiaohongshu, Browser, and Mac tools."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote, urlparse

import httpx
from aiohttp import web

from xiao_speaker_assistant.config import (
    AppConfig,
    persist_runtime_settings,
    read_env_secret,
    resolve_binary,
)
from xiao_speaker_assistant.tools.browser import (
    async_fetch_webpage_summary,
    extract_html_summary,
    open_in_edge,
    truncate_text,
    validate_http_url,
)
from xiao_speaker_assistant.tools.confirmation import ConfirmationStore
from xiao_speaker_assistant.tools.history import (
    ConversationHistoryStore,
    read_local_opencode_usage,
)
from xiao_speaker_assistant.tools.mac_local import (
    MacLocalAccessError,
    is_sensitive_path,
    list_recent_files,
    read_local_text_file,
    resolve_allowed_roots,
)
from xiao_speaker_assistant.tools.mcp_client import HttpMcpClient
from xiao_speaker_assistant.tools.memory import MemorySecurityError, MemoryStore
from xiao_speaker_assistant.tools.outlook import OutlookClient, PendingEmailDraft
from xiao_speaker_assistant.tools.reminders import ReminderStore
from xiao_speaker_assistant.voice.speaker_id import SpeakerMatch, SpeakerProfileStore
from xiao_speaker_assistant.voice.voice_catalog import (
    EDGE_ZH_VOICES,
    list_voice_profiles_dicts,
    normalize_edge_rate,
    persist_speaker_voice_config,
    resolve_edge_voice,
)
from xiao_speaker_assistant.web import load_console_html

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "xiao-speaker-assistant", "version": "0.1.0"}

AVAILABLE_MODELS: tuple[dict[str, str], ...] = (
    {"id": "raven/gemini-4-hydrogen#low", "label": "Gemini 4 Hydrogen · Low (默认低延迟 · ~2.5s)"},
    {"id": "raven/gemini-4-hydrogen#medium", "label": "Gemini 4 Hydrogen · Medium"},
    {"id": "raven/gpt-6.2-sol#low", "label": "GPT-6.2 Sol · Low (稳健推荐 · ~2.85s)"},
    {"id": "raven/gpt-6.2-sol#medium", "label": "GPT-6.2 Sol · Medium (~3.5s)"},
    {"id": "raven/gpt-6.1-sol#low", "label": "GPT-6.1 Sol · Low"},
    {"id": "raven/gpt-6.1-sol#medium", "label": "GPT-6.1 Sol · Medium"},
    {"id": "raven/gpt-5.6-sol-fast", "label": "GPT-5.6 Sol Fast (~2.4s)"},
    {"id": "raven/gpt-5.4-nano", "label": "GPT-5.4 Nano (极速 ~1.5s)"},
)
MODEL_ALIASES: dict[str, str] = {
    "hydrogen": "raven/gemini-4-hydrogen#low",
    "hydrogen low": "raven/gemini-4-hydrogen#low",
    "hydrogen低": "raven/gemini-4-hydrogen#low",
    "默认": "raven/gemini-4-hydrogen#low",
    "氢": "raven/gemini-4-hydrogen#low",
    "hydrogen medium": "raven/gemini-4-hydrogen#medium",
    "hydrogen中": "raven/gemini-4-hydrogen#medium",
    "gpt": "raven/gpt-6.2-sol#low",
    "sol": "raven/gpt-6.2-sol#low",
    "gpt6.2": "raven/gpt-6.2-sol#low",
    "gpt 6.2": "raven/gpt-6.2-sol#low",
    "6.2": "raven/gpt-6.2-sol#low",
    "gpt6.2 low": "raven/gpt-6.2-sol#low",
    "gpt6.2 medium": "raven/gpt-6.2-sol#medium",
    "gpt 6.2 medium": "raven/gpt-6.2-sol#medium",
    "gpt6.1": "raven/gpt-6.1-sol#low",
    "gpt 6.1": "raven/gpt-6.1-sol#low",
    "6.1": "raven/gpt-6.1-sol#low",
    "gpt6.1 medium": "raven/gpt-6.1-sol#medium",
    "sol fast": "raven/gpt-5.6-sol-fast",
    "fast": "raven/gpt-5.6-sol-fast",
    "nano": "raven/gpt-5.4-nano",
    "极速": "raven/gpt-5.4-nano",
}
_MODEL_REF_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?:#[A-Za-z0-9._-]+)?$")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_LOOPBACK_REMOTES = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})
_PROXY_HEADER_NAMES = (
    "CF-Connecting-IP",
    "CF-Ray",
    "X-Forwarded-For",
    "X-Forwarded-Host",
    "Forwarded",
)
_CONSOLE_COOKIE_NAME = "xsa_console_session"


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "outlook_list_emails",
        "description": "List recent Outlook emails with subject, sender, timestamp, and preview.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "unread_only": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "outlook_prepare_email",
        "description": (
            "Prepare an Outlook email draft for voice confirmation. "
            "Stores the draft in ConfirmationStore and asks the user to verbally confirm "
            "('确认发送' or '取消') before sending."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Plain-text email body"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "xiaohongshu_search",
        "description": "Search Xiaohongshu notes by keyword via Xiaohongshu MCP.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Search keyword"},
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "xiaohongshu_detail",
        "description": "Fetch full detail of a Xiaohongshu note via Xiaohongshu MCP.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "feed_id": {"type": "string", "description": "Xiaohongshu feed ID"},
                "xsec_token": {"type": "string", "description": "xsec_token from search results"},
            },
            "required": ["feed_id"],
        },
    },
    {
        "name": "browser_fetch_summary",
        "description": (
            "Fetch a web page summary: delegates to Browser MCP when enabled, "
            "otherwise fetches via HTTP and strips HTML tags into a clean text excerpt."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP/HTTPS URL to fetch"},
                "max_chars": {"type": "integer", "minimum": 100, "maximum": 20000, "default": 2000},
            },
            "required": ["url"],
        },
    },
]

LOCAL_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "browser_open_edge",
        "description": "Validate an HTTP/HTTPS URL and open it in Microsoft Edge on macOS.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP/HTTPS URL to open in Microsoft Edge"},
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                    "description": "Validate and return command without launching Edge",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "mac_list_recent_files",
        "description": (
            "List recent non-sensitive files inside allowed macOS user directories "
            "(~/Desktop, ~/Downloads, ~/Documents)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "default": "~/Desktop",
                    "description": "Target directory inside allowed user roots",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            },
        },
    },
    {
        "name": "mac_read_text_file",
        "description": (
            "Read a UTF-8 text file inside allowed macOS user directories, "
            "blocking traversal and sensitive files or keychains."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path inside allowed user roots"},
                "max_chars": {"type": "integer", "minimum": 1, "maximum": 20000, "default": 3000},
            },
            "required": ["path"],
        },
    },
    {
        "name": "mac_disk_usage",
        "description": "Check disk space (total, used, free GB and percentage) on this Mac.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "default": "/",
                    "description": "Filesystem path to inspect (default: /)",
                },
            },
        },
    },
    {
        "name": "codex_analyze",
        "description": (
            "Run local Codex CLI (`codex exec --ephemeral`) for code analysis, "
            "debugging, or technical reasoning on a repository or prompt."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Analysis task or question for Codex"},
                "workdir": {
                    "type": "string",
                    "description": "Optional working directory for Codex analysis",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "weather_query",
        "description": "Query real-time current weather and 2-day forecast for a city (fast, works in China).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name in Chinese or English, e.g. 北京, 上海, Beijing",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web quickly via Bing China for current facts, news, dates, or prices.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query keywords"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "xiaohongshu_check_login",
        "description": "Check whether the Xiaohongshu MCP server is logged in.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "browser_snapshot",
        "description": "Capture an accessibility snapshot of the active page via Playwright Edge MCP.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "memory_save",
        "description": (
            "Save or update a persistent long-term user fact or preference "
            "(e.g. home city, frequent contact email, nickname). Never store secrets or passwords."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short topic key, e.g. 常住城市, 张三邮箱"},
                "value": {"type": "string", "description": "Fact or preference to remember"},
                "category": {"type": "string", "default": "fact"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search saved long-term user memories and preferences.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "default": "", "description": "Optional search keyword"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
        },
    },
    {
        "name": "memory_forget",
        "description": "Delete a saved long-term memory entry by key or id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key_or_id": {"type": "string", "description": "Memory key or ID to remove"},
            },
            "required": ["key_or_id"],
        },
    },
    {
        "name": "speaker_list_voices",
        "description": "List available Edge-TTS Chinese voice profiles (name, ID, gender, style).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "speaker_set_voice",
        "description": (
            "Switch the speaker TTS voice and/or speech rate immediately (e.g. '晓伊', '云希', "
            "'男声', '东北话', '+15%'), optionally persisting to config.yaml."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "voice": {
                    "type": "string",
                    "description": "Voice ID or friendly Chinese name (e.g. 晓伊, 云希, 云扬, 东北话)",
                },
                "rate": {
                    "type": "string",
                    "description": "Optional speech rate percentage, e.g. '+10%' or '-10%'",
                },
                "persist": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "reminder_set",
        "description": (
            "Schedule a proactive voice reminder, countdown timer, or daily morning briefing on the speaker. "
            "Provide one of delay_seconds (e.g. 1200 for 20 min), trigger_at_iso (YYYY-MM-DDTHH:MM:SS), "
            "or daily_hhmm (e.g. '08:30')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "What to announce when due"},
                "delay_seconds": {"type": "integer", "minimum": 1, "description": "Countdown seconds"},
                "trigger_at_iso": {"type": "string", "description": "Local ISO-8601 timestamp"},
                "daily_hhmm": {"type": "string", "description": "Daily 24h HH:MM schedule"},
                "kind": {
                    "type": "string",
                    "enum": ["reminder", "morning_briefing"],
                    "default": "reminder",
                },
            },
            "required": ["message"],
        },
    },
    {
        "name": "reminder_list",
        "description": "List all active scheduled voice reminders, timers, and daily briefings.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "reminder_cancel",
        "description": "Cancel scheduled reminders matching an ID or keyword.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id_or_keyword": {"type": "string", "description": "Reminder ID or message keyword"},
            },
            "required": ["id_or_keyword"],
        },
    },
    {
        "name": "speaker_enroll_voiceprint",
        "description": (
            "Save or update a family member's voiceprint from recent speaker audio. "
            "Use target='previous' when the user says '刚才和你说话的是拳拳', or target='current' "
            "when the user says '我是大雄，记住我的声音'. Supports extensible role, style_hint, "
            "capabilities (e.g. ['storytelling', 'kid_safe_only']), notes, and preferred_voice."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Member name, e.g. 拳拳, 大雄, 爱人, 爷爷"},
                "role": {
                    "type": "string",
                    "default": "family",
                    "description": "Member role: host | spouse | child | elder | guest | custom",
                },
                "style_hint": {"type": "string", "description": "Optional addressing/tone instruction"},
                "capabilities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extensible capability or policy tags for this member",
                },
                "notes": {"type": "string", "description": "Extra preferences or restrictions"},
                "preferred_voice": {"type": "string", "description": "Optional Edge-TTS voice for this member"},
                "target": {
                    "type": "string",
                    "enum": ["previous", "current"],
                    "default": "previous",
                    "description": "'previous' for the previous speaker ('刚才说话的是...'), 'current' for current speaker ('我是...')",
                },
                "keyword": {"type": "string", "description": "Optional phrase snippet spoken by that member"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "speaker_list_voiceprints",
        "description": "List enrolled family member voiceprints, roles, capabilities, and recent utterance history.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "speaker_update_voiceprint",
        "description": "Update an enrolled member's role, style_hint, capabilities, notes, or preferred_voice.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name_or_id": {"type": "string", "description": "Enrolled member name or ID"},
                "role": {"type": "string"},
                "style_hint": {"type": "string"},
                "capabilities": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
                "preferred_voice": {"type": "string"},
            },
            "required": ["name_or_id"],
        },
    },
    {
        "name": "speaker_delete_voiceprint",
        "description": "Delete an enrolled family member voiceprint by name or ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name_or_id": {"type": "string", "description": "Member name or ID to delete"},
            },
            "required": ["name_or_id"],
        },
    },
    {
        "name": "speaker_list_models",
        "description": "List available LLM chat models (e.g. Gemini 4 Hydrogen, GPT-6.2 Sol, GPT-5.4 Nano) and the currently active model.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "speaker_set_model",
        "description": (
            "Switch the active LLM conversation model immediately by model ID or friendly alias "
            "(e.g. 'gpt', 'gpt 6.2', 'hydrogen', 'nano', 'raven/gpt-6.2-sol#low'), "
            "optionally persisting to config.yaml."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model ID or alias such as 'gpt', 'gpt 6.2', 'hydrogen', 'nano', or 'raven/gpt-6.2-sol#low'",
                },
                "persist": {"type": "boolean", "default": True},
            },
            "required": ["model"],
        },
    },
]

ALL_TOOL_DEFINITIONS: list[dict[str, Any]] = [*TOOL_DEFINITIONS, *LOCAL_TOOL_DEFINITIONS]


class McpServer:
    """Stdio & Streamable-HTTP JSON-RPC 2.0 MCP server for Xiao Speaker Assistant tools."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        outlook_client: OutlookClient | None = None,
        xiaohongshu_client: HttpMcpClient | None = None,
        browser_client: HttpMcpClient | None = None,
        confirmation_store: ConfirmationStore | None = None,
        memory_store: MemoryStore | None = None,
        reminder_store: ReminderStore | None = None,
        speaker_store: SpeakerProfileStore | None = None,
        history_store: ConversationHistoryStore | None = None,
        console_password: str | None = None,
        on_memory_changed: Callable[[], None] | None = None,
        on_voice_changed: Callable[[str, str], None] | None = None,
        on_model_changed: Callable[[str], None] | None = None,
        on_wakewords_changed: Callable[[list[str], list[str], bool], None] | None = None,
        on_reminders_config_changed: Callable[[bool, str, str], None] | None = None,
        on_capabilities_changed: Callable[[], None] | None = None,
        config_path: str | Path | None = None,
        speakers_path: str | Path | None = None,
        reminders_path: str | Path | None = None,
        http_client: httpx.AsyncClient | None = None,
        include_local_tools: bool = False,
        mac_allowed_roots: Sequence[str | Path] | None = None,
        edge_dry_run: bool = False,
    ) -> None:
        self._config = config or AppConfig()
        self._outlook = outlook_client or OutlookClient(self._config.tools.outlook)
        self._xiaohongshu = xiaohongshu_client or HttpMcpClient(
            self._config.tools.xiaohongshu.mcp_url
        )
        self._browser_injected = browser_client is not None
        self._browser = browser_client or HttpMcpClient(self._config.tools.browser.mcp_url)
        self.confirmation_store = confirmation_store or ConfirmationStore()
        self.memory_store = memory_store or MemoryStore(
            self._config.memory.file_path
        )
        self._reminders_path = Path(reminders_path).expanduser() if reminders_path is not None else None
        self.reminder_store = reminder_store or ReminderStore(
            self._reminders_path if self._reminders_path is not None else self._config.reminders.file_path
        )
        self.speaker_store = speaker_store or SpeakerProfileStore(
            self._config.speech.speaker_profiles_file
        )
        self.history_store = history_store or ConversationHistoryStore()
        self._console_password = console_password
        self._on_memory_changed = on_memory_changed
        self._on_voice_changed = on_voice_changed
        self._on_model_changed = on_model_changed
        self._on_wakewords_changed = on_wakewords_changed
        self._on_reminders_config_changed = on_reminders_config_changed
        self._on_capabilities_changed = on_capabilities_changed
        self._config_path = config_path
        self._speakers_path = Path(speakers_path).expanduser() if speakers_path is not None else None
        self._http_client = http_client
        self._include_local_tools = include_local_tools
        self._mac_allowed_roots = mac_allowed_roots
        self._edge_dry_run = edge_dry_run
        self._http_sessions: set[str] = set()
        # Track all concurrent in-flight voice/query turns by token -> (speaker, started_at).
        # Fail closed: sensitive tools are allowed ONLY when at least one active turn exists,
        # EVERY concurrent active turn is an authenticated host speaker, and none is expired.
        self._active_voice_turns: dict[str, tuple[SpeakerMatch | None, float]] = {}

    def begin_voice_turn(self, speaker: SpeakerMatch | None) -> str:
        """Register an in-flight spoken/query turn and return a token for `end_voice_turn`."""
        turn_id = uuid.uuid4().hex
        self._active_voice_turns[turn_id] = (speaker, time.monotonic())
        return turn_id

    def end_voice_turn(self, turn_id: str) -> None:
        """Unregister a finished/cancelled turn without disturbing other active turns."""
        self._active_voice_turns.pop(turn_id, None)

    def set_active_voice_speaker(self, speaker: SpeakerMatch | None) -> None:
        """Convenience helper for single-turn callers/tests."""
        self._active_voice_turns.clear()
        if speaker is not None:
            self._active_voice_turns["__direct__"] = (speaker, time.monotonic())

    _HOST_ROLES: frozenset[str] = frozenset({"host", "主人", "owner"})
    _SENSITIVE_VOICEPRINT_TOOLS: frozenset[str] = frozenset(
        {
            "mac_read_text_file",
            "mac_list_recent_files",
            "codex_analyze",
            "outlook_list_emails",
            "outlook_prepare_email",
            "outlook_send_email",
            "speaker_enroll_voiceprint",
            "speaker_update_voiceprint",
            "speaker_delete_voiceprint",
        }
    )

    @classmethod
    def is_host_speaker(cls, speaker: SpeakerMatch | None) -> bool:
        return speaker is not None and speaker.role.strip().lower() in cls._HOST_ROLES

    def _has_enrolled_host_voiceprint(self) -> bool:
        return any(
            str(p.get("role") or "").strip().lower() in self._HOST_ROLES
            for p in self.speaker_store.list_profiles()
        )

    def _check_host_voiceprint_guard(self, tool_name: str) -> dict[str, Any] | None:
        if not self._config.tools.require_host_voiceprint_for_sensitive_tools:
            return None
        if tool_name not in self._SENSITIVE_VOICEPRINT_TOOLS:
            return None
        # Allow bootstrapping the very first host voiceprint when none is enrolled yet.
        if tool_name == "speaker_enroll_voiceprint" and not self._has_enrolled_host_voiceprint():
            return None

        now = time.monotonic()
        # Drop stale turns older than 120s so a leaked/abandoned turn can never keep privilege alive.
        stale_ids = [tid for tid, (_s, ts) in self._active_voice_turns.items() if (now - ts) > 120.0]
        for tid in stale_ids:
            self._active_voice_turns.pop(tid, None)

        if self._active_voice_turns and all(
            self.is_host_speaker(s) for s, _ts in self._active_voice_turns.values()
        ):
            return None

        non_host = next(
            (s for s, _ts in self._active_voice_turns.values() if not self.is_host_speaker(s)),
            None,
        )
        who = f"{non_host.name} ({non_host.role})" if non_host is not None else "未识别声纹"
        return self._tool_result(
            {
                "error": "host_voiceprint_required",
                "tool": tool_name,
                "speaker": who,
                "message": (
                    f"已开启敏感工具主人声纹鉴权，当前说话人（{who}）无权调用 {tool_name}。"
                    "请由已注册为 host 角色的主人亲自语音触发。"
                ),
            },
            is_error=True,
        )

    def reload_config(self, new_config: AppConfig) -> None:
        self._config = new_config
        self._outlook = OutlookClient(new_config.tools.outlook)
        self._xiaohongshu = HttpMcpClient(new_config.tools.xiaohongshu.mcp_url)
        self._browser = HttpMcpClient(new_config.tools.browser.mcp_url)

    @staticmethod
    def _ok(req_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _tool_result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        }

    async def handle_line(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            return json.dumps(self._error(None, -32700, "Parse error"), ensure_ascii=False)
        if not isinstance(message, dict):
            return json.dumps(self._error(None, -32600, "Invalid Request"), ensure_ascii=False)
        response = await self.handle_message(message)
        if response is None:
            return None
        return json.dumps(response, ensure_ascii=False)

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        has_id = "id" in message
        req_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        if not isinstance(method, str):
            return self._error(req_id, -32600, "Invalid Request") if has_id else None

        # JSON-RPC notifications omit `id` and must not receive a response.
        if not has_id:
            logger.debug("Received MCP notification: %s", method)
            return None

        if method == "initialize":
            return self._ok(
                req_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                },
            )
        if method == "ping":
            return self._ok(req_id, {})
        if method == "tools/list":
            tools = (
                ALL_TOOL_DEFINITIONS
                if (
                    self._include_local_tools
                    or self._config.tools.browser.enabled
                    or self._mac_allowed_roots is not None
                )
                else TOOL_DEFINITIONS
            )
            return self._ok(req_id, {"tools": tools})
        if method == "tools/call":
            if not isinstance(params, dict):
                return self._error(req_id, -32602, "Invalid params")
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return self._error(req_id, -32602, "Invalid params: expected tool name and arguments")
            return await self._dispatch_tool_call(req_id, name, arguments)

        return self._error(req_id, -32601, f"Method not found: {method}")

    async def _dispatch_tool_call(
        self, req_id: Any, name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        vp_denied = self._check_host_voiceprint_guard(name)
        if vp_denied is not None:
            return self._ok(req_id, vp_denied)
        if name == "outlook_list_emails":
            return self._ok(req_id, await self._call_outlook_list_emails(args))
        if name in {"outlook_prepare_email", "outlook_send_email"}:
            return self._ok(req_id, await self._call_outlook_prepare_email(args))
        if name == "xiaohongshu_search":
            return self._ok(req_id, await self._call_xiaohongshu_search(args))
        if name == "xiaohongshu_detail":
            return self._ok(req_id, await self._call_xiaohongshu_detail(args))
        if name == "browser_fetch_summary":
            return self._ok(req_id, await self._call_browser_fetch_summary(args))
        if name == "browser_open_edge":
            return self._ok(req_id, await self._call_browser_open_edge(args))
        if name == "mac_list_recent_files":
            return self._ok(req_id, await self._call_mac_list_recent_files(args))
        if name == "mac_read_text_file":
            return self._ok(req_id, await self._call_mac_read_text_file(args))
        if name == "mac_disk_usage":
            return self._ok(req_id, await self._call_mac_disk_usage(args))
        if name == "codex_analyze":
            return self._ok(req_id, await self._call_codex_analyze(args))
        if name == "weather_query":
            return self._ok(req_id, await self._call_weather_query(args))
        if name == "web_search":
            return self._ok(req_id, await self._call_web_search(args))
        if name == "xiaohongshu_check_login":
            return self._ok(req_id, await self._call_xiaohongshu_check_login())
        if name == "browser_snapshot":
            return self._ok(req_id, await self._call_browser_snapshot())
        if name == "memory_save":
            return self._ok(req_id, await self._call_memory_save(args))
        if name == "memory_search":
            return self._ok(req_id, await self._call_memory_search(args))
        if name == "memory_forget":
            return self._ok(req_id, await self._call_memory_forget(args))
        if name == "speaker_list_voices":
            return self._ok(req_id, self._call_speaker_list_voices())
        if name == "speaker_set_voice":
            return self._ok(req_id, await self._call_speaker_set_voice(args))
        if name == "reminder_set":
            return self._ok(req_id, self._call_reminder_set(args))
        if name == "reminder_list":
            return self._ok(req_id, self._call_reminder_list())
        if name == "reminder_cancel":
            return self._ok(req_id, self._call_reminder_cancel(args))
        if name == "speaker_enroll_voiceprint":
            return self._ok(req_id, self._call_speaker_enroll_voiceprint(args))
        if name == "speaker_list_voiceprints":
            return self._ok(req_id, self._call_speaker_list_voiceprints())
        if name == "speaker_update_voiceprint":
            return self._ok(req_id, self._call_speaker_update_voiceprint(args))
        if name == "speaker_delete_voiceprint":
            return self._ok(req_id, self._call_speaker_delete_voiceprint(args))
        if name == "speaker_list_models":
            return self._ok(req_id, self._call_speaker_list_models())
        if name == "speaker_set_model":
            return self._ok(req_id, await self._call_speaker_set_model(args))
        return self._error(req_id, -32602, f"Unknown tool: {name}")

    async def _call_outlook_list_emails(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._config.tools.outlook.enabled:
            return self._tool_result(
                {
                    "error": "outlook_disabled",
                    "message": "Outlook tool is disabled. Set tools.outlook.enabled: true in config.yaml.",
                },
                is_error=True,
            )
        limit = max(1, min(50, int(args.get("limit", 5))))
        unread_only = bool(args.get("unread_only", False))
        try:
            emails = await self._outlook.list_recent_emails(limit=limit, unread_only=unread_only)
        except Exception as exc:
            return self._tool_result(
                {"error": "outlook_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result({"emails": emails, "count": len(emails)})

    async def _call_outlook_prepare_email(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._config.tools.outlook.enabled:
            return self._tool_result(
                {
                    "error": "outlook_disabled",
                    "message": "Outlook tool is disabled. Set tools.outlook.enabled: true in config.yaml.",
                },
                is_error=True,
            )
        to = str(args.get("to") or "").strip()
        subject = str(args.get("subject") or "").strip()
        body = str(args.get("body") or "").strip()
        if not to or not subject or not body:
            return self._tool_result(
                {
                    "error": "invalid_arguments",
                    "message": "Arguments 'to', 'subject', and 'body' are required.",
                },
                is_error=True,
            )

        # Never trust LLM-provided `confirmed` when voice confirmation is required;
        # only ConfirmationStore + explicit spoken confirmation can dispatch send_email.
        if self._config.tools.outlook.require_voice_confirmation_for_send:
            draft = PendingEmailDraft(to=to, subject=subject, body=body, confirmed=False)
            self.confirmation_store.set_pending(draft)
            result_json = await self._outlook.send_email(draft)
            return self._tool_result(result_json, is_error=False)

        draft = PendingEmailDraft(to=to, subject=subject, body=body, confirmed=True)
        try:
            result_json = await self._outlook.send_email(draft)
        except Exception as exc:
            return self._tool_result(
                {"error": "outlook_error", "message": str(exc)},
                is_error=True,
            )
        self.confirmation_store.clear()
        return self._tool_result(result_json, is_error=False)

    def _xiaohongshu_disabled_result(self) -> dict[str, Any]:
        mcp_url = self._config.tools.xiaohongshu.mcp_url
        return self._tool_result(
            {
                "error": "xiaohongshu_mcp_not_enabled",
                "message": (
                    "Xiaohongshu MCP is not enabled. Set tools.xiaohongshu.enabled: true "
                    f"in config.yaml and start the Xiaohongshu MCP server at {mcp_url}."
                ),
                "mcp_url": mcp_url,
            },
            is_error=True,
        )

    async def _call_xiaohongshu_search(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._config.tools.xiaohongshu.enabled:
            return self._xiaohongshu_disabled_result()
        keyword = str(args.get("keyword") or "").strip()
        if not keyword:
            return self._tool_result(
                {"error": "invalid_arguments", "message": "Argument 'keyword' is required."},
                is_error=True,
            )
        try:
            res = await self._xiaohongshu.call_tool("search_feeds", {"keyword": keyword})
        except Exception as exc:
            return self._tool_result(
                {"error": "xiaohongshu_mcp_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result(res)

    async def _call_xiaohongshu_detail(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._config.tools.xiaohongshu.enabled:
            return self._xiaohongshu_disabled_result()
        feed_id = str(args.get("feed_id") or "").strip()
        if not feed_id:
            return self._tool_result(
                {"error": "invalid_arguments", "message": "Argument 'feed_id' is required."},
                is_error=True,
            )
        payload: dict[str, Any] = {"feed_id": feed_id}
        xsec_token = str(args.get("xsec_token") or "").strip()
        if xsec_token:
            payload["xsec_token"] = xsec_token
        try:
            res = await self._xiaohongshu.call_tool("get_feed_detail", payload)
        except Exception as exc:
            return self._tool_result(
                {"error": "xiaohongshu_mcp_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result(res)

    async def _call_xiaohongshu_check_login(self) -> dict[str, Any]:
        if not self._config.tools.xiaohongshu.enabled:
            return self._xiaohongshu_disabled_result()
        try:
            res = await self._xiaohongshu.call_tool("check_login_status", {})
        except Exception as exc:
            return self._tool_result(
                {"error": "xiaohongshu_mcp_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result(res)

    async def _call_browser_snapshot(self) -> dict[str, Any]:
        if not self._config.tools.browser.enabled:
            return self._tool_result(
                {
                    "error": "browser_mcp_not_enabled",
                    "message": "Browser MCP is not enabled. Set tools.browser.enabled: true and run scripts/start_mcp_services.sh.",
                },
                is_error=True,
            )
        try:
            res = await self._browser.call_tool("browser_snapshot", {})
        except Exception as exc:
            return self._tool_result(
                {"error": "browser_mcp_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result(res)

    async def _call_browser_fetch_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_url = str(args.get("url") or "").strip()
        should_resolve_dns = self._http_client is None and not self._browser_injected
        try:
            url = await asyncio.to_thread(
                validate_http_url,
                raw_url,
                block_private_network=True,
                resolve_dns=should_resolve_dns,
            )
        except ValueError as exc:
            return self._tool_result(
                {"error": "invalid_url", "message": str(exc)},
                is_error=True,
            )
        max_chars = max(100, min(20000, int(args.get("max_chars", 2000))))

        if self._config.tools.browser.enabled:
            try:
                res = await self._browser.call_tool("browser_navigate", {"url": url})
            except Exception as exc:
                return self._tool_result(
                    {"error": "browser_mcp_error", "message": str(exc)},
                    is_error=True,
                )
            return self._tool_result({"source": "browser_mcp", "url": url, "result": res})

        try:
            summary = await async_fetch_webpage_summary(
                url,
                max_chars=max_chars,
                client=self._http_client,
                block_private_network=True,
            )
        except Exception as exc:
            return self._tool_result(
                {"error": "http_fetch_error", "message": str(exc), "url": url},
                is_error=True,
            )
        return self._tool_result(summary)

    async def _call_browser_open_edge(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_url = str(args.get("url") or "").strip()
        dry_run = bool(args.get("dry_run", self._edge_dry_run))
        try:
            validated_url = await asyncio.to_thread(
                validate_http_url,
                raw_url,
                block_private_network=True,
                resolve_dns=not dry_run,
            )
            result_json = await asyncio.to_thread(open_in_edge, validated_url, dry_run=dry_run)
        except ValueError as exc:
            return self._tool_result(
                {"error": "invalid_url", "message": str(exc)},
                is_error=True,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            return self._tool_result(
                {"error": "edge_launch_error", "message": str(exc), "url": raw_url},
                is_error=True,
            )
        return self._tool_result(result_json)

    async def _call_mac_list_recent_files(self, args: dict[str, Any]) -> dict[str, Any]:
        directory = str(args.get("directory") or "~/Desktop").strip()
        limit = max(1, min(100, int(args.get("limit", 10))))
        try:
            files = await asyncio.to_thread(
                list_recent_files,
                directory,
                limit,
                allowed_roots=self._mac_allowed_roots,
            )
        except MacLocalAccessError as exc:
            return self._tool_result(
                {"error": "access_denied", "message": str(exc)},
                is_error=True,
            )
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            return self._tool_result(
                {"error": "mac_local_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result({"directory": directory, "files": files, "count": len(files)})

    async def _call_mac_read_text_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = str(args.get("path") or "").strip()
        max_chars = max(1, min(20000, int(args.get("max_chars", 3000))))
        try:
            content = await asyncio.to_thread(
                read_local_text_file,
                path,
                max_chars,
                allowed_roots=self._mac_allowed_roots,
            )
        except MacLocalAccessError as exc:
            return self._tool_result(
                {"error": "access_denied", "message": str(exc)},
                is_error=True,
            )
        except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
            return self._tool_result(
                {"error": "mac_local_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result({"path": path, "content": content, "chars": len(content)})

    async def _call_mac_disk_usage(self, args: dict[str, Any]) -> dict[str, Any]:
        target = str(args.get("path") or "/").strip() or "/"
        try:
            resolved = str(Path(target).expanduser())
            usage = await asyncio.to_thread(shutil.disk_usage, resolved)
        except OSError as exc:
            return self._tool_result(
                {"error": "mac_disk_error", "message": str(exc)},
                is_error=True,
            )
        gib = 1024**3
        total_gb = round(usage.total / gib, 2)
        used_gb = round(usage.used / gib, 2)
        free_gb = round(usage.free / gib, 2)
        free_pct = round((usage.free / usage.total) * 100.0, 1) if usage.total else 0.0
        return self._tool_result(
            {
                "path": resolved,
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": free_gb,
                "free_percent": free_pct,
            }
        )

    def _resolve_codex_workdir(self, raw_workdir: str) -> Path:
        if "\x00" in raw_workdir:
            raise MacLocalAccessError("Codex workdir contains NUL byte.")
        cleaned = raw_workdir.strip()
        if not cleaned:
            raise MacLocalAccessError("A non-empty Codex workdir is required.")

        codex_cfg = self._config.llm.providers.codex
        home_resolved = Path.home().resolve(strict=False)
        overly_broad_roots = {
            Path("/"),
            home_resolved,
            Path("/Users").resolve(strict=False),
            Path("/private").resolve(strict=False),
            Path("/etc").resolve(strict=False),
            Path("/var").resolve(strict=False),
            Path("/tmp").resolve(strict=False),
        }
        allowed_roots: list[Path] = list(resolve_allowed_roots(self._mac_allowed_roots))
        configured_root = Path(codex_cfg.workdir).expanduser().resolve(strict=False)
        if configured_root not in overly_broad_roots:
            allowed_roots.insert(0, configured_root)
        workspace_dir = Path("~/workspace").expanduser().resolve(strict=False)
        if workspace_dir.is_dir():
            allowed_roots.append(workspace_dir)

        expanded = Path(cleaned).expanduser()
        candidate = expanded.resolve(strict=False)
        matching_root = next((r for r in allowed_roots if candidate.is_relative_to(r)), None)
        if matching_root is None:
            raise MacLocalAccessError(
                f"Codex workdir is outside allowed directories: {raw_workdir}"
            )
        if is_sensitive_path(candidate, root=matching_root):
            raise MacLocalAccessError(
                f"Access to sensitive or hidden directory is blocked: {candidate.name}"
            )
        unfollowed_abs = expanded if expanded.is_absolute() else Path.cwd() / expanded
        for root in allowed_roots:
            if unfollowed_abs.is_relative_to(root) and is_sensitive_path(unfollowed_abs, root=root):
                raise MacLocalAccessError(
                    f"Access to sensitive or hidden directory is blocked: {unfollowed_abs.name}"
                )
        if not candidate.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {raw_workdir}")
        return candidate

    async def _call_codex_analyze(self, args: dict[str, Any]) -> dict[str, Any]:
        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            return self._tool_result(
                {"error": "invalid_arguments", "message": "Argument 'prompt' is required."},
                is_error=True,
            )
        codex_cfg = self._config.llm.providers.codex
        raw_workdir = str(args.get("workdir") or "").strip() or codex_cfg.workdir
        try:
            workdir_path = self._resolve_codex_workdir(raw_workdir)
        except MacLocalAccessError as exc:
            return self._tool_result(
                {"error": "access_denied", "message": str(exc)},
                is_error=True,
            )
        except FileNotFoundError as exc:
            return self._tool_result(
                {"error": "invalid_workdir", "message": str(exc)},
                is_error=True,
            )
        try:
            binary = resolve_binary(codex_cfg.binary)
        except RuntimeError as exc:
            return self._tool_result({"error": "codex_missing", "message": str(exc)}, is_error=True)

        fd, out_file = tempfile.mkstemp(prefix="xsa-codex-mcp-", suffix=".txt")
        os.close(fd)
        try:
            cmd = [
                binary,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "-s",
                "read-only",
                "-C",
                str(workdir_path),
                "-o",
                out_file,
            ]
            if codex_cfg.model:
                cmd.extend(["-m", codex_cfg.model])
            cmd.append(prompt)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=codex_cfg.timeout_sec
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return self._tool_result(
                    {"error": "codex_timeout", "message": "Codex analysis timed out."},
                    is_error=True,
                )
            if proc.returncode != 0:
                err = stderr_b.decode("utf-8", errors="replace").strip()
                out = stdout_b.decode("utf-8", errors="replace").strip()
                return self._tool_result(
                    {"error": "codex_failed", "message": err or out[-300:]},
                    is_error=True,
                )
            text = Path(out_file).read_text(encoding="utf-8", errors="replace").strip()
            return self._tool_result(
                {"status": "ok", "workdir": str(workdir_path), "analysis": text[:4000]}
            )
        finally:
            try:
                os.unlink(out_file)
            except OSError:
                pass

    async def _fetch_open_meteo_7d(
        self, client: httpx.AsyncClient, city: str
    ) -> list[dict[str, Any]]:
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh"},
        )
        geo.raise_for_status()
        results = geo.json().get("results") or []
        if not results:
            return []
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        fc = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": "Asia/Shanghai",
            },
        )
        fc.raise_for_status()
        daily = fc.json().get("daily") or {}
        dates = daily.get("time") or []
        mins = daily.get("temperature_2m_min") or []
        maxs = daily.get("temperature_2m_max") or []
        return [
            {"date": d, "min_c": lo, "max_c": hi}
            for d, lo, hi in zip(dates, mins, maxs)
        ]

    async def _call_weather_query(self, args: dict[str, Any]) -> dict[str, Any]:
        city = str(args.get("city") or "").strip()
        if not city:
            return self._tool_result(
                {"error": "invalid_arguments", "message": "Argument 'city' is required."},
                is_error=True,
            )
        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=6.0, follow_redirects=True)
        try:
            wttr_task = client.get(
                f"https://wttr.in/{quote(city, safe='')}",
                params={"format": "j1", "lang": "zh"},
                headers={"User-Agent": "curl/8.5.0"},
            )
            meteo_task = self._fetch_open_meteo_7d(client, city)
            wttr_res, meteo_res = await asyncio.gather(
                wttr_task, meteo_task, return_exceptions=True
            )

            forecast_by_date: dict[str, dict[str, Any]] = {}
            if isinstance(meteo_res, list):
                for day in meteo_res:
                    d_str = str(day.get("date") or "")
                    if d_str:
                        forecast_by_date[d_str] = {
                            "date": d_str,
                            "min_c": day.get("min_c"),
                            "max_c": day.get("max_c"),
                        }

            result: dict[str, Any] = {"city": city}
            if isinstance(wttr_res, httpx.Response) and wttr_res.status_code < 400:
                data = wttr_res.json()
                cur = (data.get("current_condition") or [{}])[0]
                weather_days = data.get("weather") or []
                lang_desc = (cur.get("lang_zh") or cur.get("weatherDesc") or [{}])[0].get(
                    "value", ""
                )
                result.update(
                    {
                        "current_temp_c": cur.get("temp_C"),
                        "feels_like_c": cur.get("FeelsLikeC"),
                        "condition": lang_desc,
                        "humidity_pct": cur.get("humidity"),
                        "wind_kmph": cur.get("windspeedKmph"),
                    }
                )
                for day in weather_days:
                    d_str = str(day.get("date") or "")
                    hourly = day.get("hourly") or []
                    mid = hourly[len(hourly) // 2] if hourly else {}
                    cond = (mid.get("lang_zh") or mid.get("weatherDesc") or [{}])[0].get(
                        "value", ""
                    )
                    entry = forecast_by_date.setdefault(d_str, {"date": d_str})
                    entry.setdefault("min_c", day.get("mintempC"))
                    entry.setdefault("max_c", day.get("maxtempC"))
                    if cond:
                        entry["condition"] = cond

            if not forecast_by_date and isinstance(wttr_res, Exception):
                raise wttr_res

            ordered = [forecast_by_date[k] for k in sorted(forecast_by_date)]
            result["forecast"] = ordered
            if len(ordered) > 0:
                result["today_min_c"] = ordered[0].get("min_c")
                result["today_max_c"] = ordered[0].get("max_c")
            if len(ordered) > 1:
                result["tomorrow_min_c"] = ordered[1].get("min_c")
                result["tomorrow_max_c"] = ordered[1].get("max_c")
            return self._tool_result(result)
        except Exception as exc:
            return self._tool_result(
                {"error": "weather_error", "message": str(exc), "city": city},
                is_error=True,
            )
        finally:
            if owns_client:
                await client.aclose()

    async def _call_web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            return self._tool_result(
                {"error": "invalid_arguments", "message": "Argument 'query' is required."},
                is_error=True,
            )
        limit = max(1, min(8, int(args.get("limit", 5))))
        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=6.0, follow_redirects=True)
        try:
            resp = await client.get(
                "https://www.bing.com/search",
                params={"q": query, "mkt": "zh-CN"},
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                },
            )
            resp.raise_for_status()
            items: list[dict[str, str]] = []
            for block in resp.text.split('<li class="b_algo"')[1:]:
                m_link = re.search(
                    r'<h2[^>]*>[\s\S]*?<a[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>', block
                )
                if not m_link:
                    continue
                url = html.unescape(m_link.group(1))
                title = html.unescape(re.sub(r"<[^>]+>", "", m_link.group(2))).strip()
                m_snip = re.search(r'<p[^>]*>([\s\S]*?)</p>', block)
                snippet = (
                    html.unescape(re.sub(r"<[^>]+>", "", m_snip.group(1))).strip()
                    if m_snip
                    else ""
                )
                if title:
                    items.append({"title": title, "snippet": snippet[:240], "url": url})
                if len(items) >= limit:
                    break
            return self._tool_result({"query": query, "results": items, "count": len(items)})
        except Exception as exc:
            return self._tool_result(
                {"error": "web_search_error", "message": str(exc), "query": query},
                is_error=True,
            )
        finally:
            if owns_client:
                await client.aclose()

    async def _call_memory_save(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._config.memory.enabled:
            return self._tool_result({"error": "memory_disabled", "message": "Long-term memory is disabled."}, is_error=True)
        key = str(args.get("key") or "").strip()
        value = str(args.get("value") or "").strip()
        category = str(args.get("category") or "fact").strip()
        try:
            entry = self.memory_store.save(key, value, category=category)
        except (ValueError, MemorySecurityError) as exc:
            return self._tool_result({"error": "memory_rejected", "message": str(exc)}, is_error=True)
        if self._on_memory_changed is not None:
            self._on_memory_changed()
        return self._tool_result({"status": "saved", "id": entry.id, "key": entry.key, "value": entry.value})

    async def _call_memory_search(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._config.memory.enabled:
            return self._tool_result({"memories": [], "count": 0})
        query = str(args.get("query") or "").strip()
        limit = max(1, min(50, int(args.get("limit", 10))))
        items = self.memory_store.search(query, limit=limit)
        return self._tool_result(
            {
                "memories": [
                    {"id": e.id, "key": e.key, "value": e.value, "category": e.category}
                    for e in items
                ],
                "count": len(items),
            }
        )

    async def _call_memory_forget(self, args: dict[str, Any]) -> dict[str, Any]:
        key_or_id = str(args.get("key_or_id") or "").strip()
        removed = self.memory_store.forget(key_or_id)
        if removed and self._on_memory_changed is not None:
            self._on_memory_changed()
        return self._tool_result({"removed": removed, "key_or_id": key_or_id})

    def _call_speaker_list_voices(self) -> dict[str, Any]:
        return self._tool_result(
            {
                "current_voice": self._config.speaker.edge_voice,
                "current_rate": self._config.speaker.edge_rate,
                "voices": list_voice_profiles_dicts(),
            }
        )

    async def _call_speaker_set_voice(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_voice = str(args.get("voice") or "").strip()
        raw_rate = str(args.get("rate") or "").strip()
        persist = bool(args.get("persist", True))
        if not raw_voice and not raw_rate:
            return self._tool_result(
                {"error": "invalid_arguments", "message": "Provide 'voice' and/or 'rate'."},
                is_error=True,
            )
        new_voice = self._config.speaker.edge_voice
        new_rate = self._config.speaker.edge_rate
        voice_label = new_voice
        try:
            if raw_voice:
                profile = resolve_edge_voice(raw_voice)
                new_voice = profile.id
                voice_label = f"{profile.name} ({profile.style})"
            if raw_rate:
                new_rate = normalize_edge_rate(raw_rate)
        except ValueError as exc:
            return self._tool_result(
                {"error": "invalid_voice_or_rate", "message": str(exc)},
                is_error=True,
            )

        self._config.speaker.edge_voice = new_voice
        self._config.speaker.edge_rate = new_rate
        persisted = False
        if persist and self._config_path is not None:
            await asyncio.to_thread(
                persist_speaker_voice_config,
                self._config_path,
                edge_voice=new_voice,
                edge_rate=new_rate,
            )
            persisted = True
        if self._on_voice_changed is not None:
            self._on_voice_changed(new_voice, new_rate)
        return self._tool_result(
            {
                "status": "updated",
                "voice": new_voice,
                "voice_label": voice_label,
                "rate": new_rate,
                "persisted": persisted,
            }
        )

    def _call_reminder_set(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._config.reminders.enabled:
            return self._tool_result(
                {"error": "reminders_disabled", "message": "Reminders are disabled in config."},
                is_error=True,
            )
        message = str(args.get("message") or "").strip()
        delay_seconds = args.get("delay_seconds")
        trigger_at_iso = args.get("trigger_at_iso")
        daily_hhmm = args.get("daily_hhmm")
        kind = str(args.get("kind") or "reminder").strip()
        try:
            entry = self.reminder_store.add(
                message,
                delay_seconds=int(delay_seconds) if delay_seconds is not None else None,
                trigger_at_iso=str(trigger_at_iso) if trigger_at_iso else None,
                daily_hhmm=str(daily_hhmm) if daily_hhmm else None,
                kind=kind,
            )
        except ValueError as exc:
            return self._tool_result({"error": "invalid_reminder", "message": str(exc)}, is_error=True)
        return self._tool_result(
            {
                "status": "scheduled",
                "id": entry.id,
                "message": entry.message,
                "kind": entry.kind,
                "daily_hhmm": entry.daily_hhmm,
                "trigger_at": entry.trigger_at,
            }
        )

    def _call_reminder_list(self) -> dict[str, Any]:
        items = self.reminder_store.list_entries()
        return self._tool_result(
            {
                "reminders": [
                    {
                        "id": e.id,
                        "message": e.message,
                        "kind": e.kind,
                        "daily_hhmm": e.daily_hhmm,
                        "trigger_at": e.trigger_at,
                    }
                    for e in items
                ],
                "count": len(items),
            }
        )

    def _call_reminder_cancel(self, args: dict[str, Any]) -> dict[str, Any]:
        target = str(args.get("id_or_keyword") or "").strip()
        removed = self.reminder_store.cancel(target)
        return self._tool_result({"removed": removed, "id_or_keyword": target})

    def _call_speaker_enroll_voiceprint(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or "").strip()
        role = str(args.get("role") or "family").strip() or "family"
        style_hint = str(args.get("style_hint") or "").strip()
        notes = str(args.get("notes") or "").strip()
        target = str(args.get("target") or "previous").strip() or "previous"
        keyword = str(args.get("keyword") or "").strip()
        raw_caps = args.get("capabilities")
        capabilities = (
            [str(c).strip() for c in raw_caps if str(c).strip()]
            if isinstance(raw_caps, list)
            else None
        )
        raw_voice = args.get("preferred_voice")
        preferred_voice: str | None = None
        if isinstance(raw_voice, str) and raw_voice.strip():
            try:
                preferred_voice = resolve_edge_voice(raw_voice.strip()).id
            except ValueError as exc:
                return self._tool_result(
                    {"error": "invalid_voice", "message": str(exc)}, is_error=True
                )
        try:
            profile, rec = self.speaker_store.enroll_from_recent(
                name,
                role=role,
                style_hint=style_hint,
                capabilities=capabilities,
                notes=notes,
                preferred_voice=preferred_voice,
                target=target,
                keyword=keyword,
            )
        except ValueError as exc:
            return self._tool_result(
                {"error": "voiceprint_enroll_failed", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result(
            {
                "status": "enrolled",
                "id": profile.id,
                "name": profile.name,
                "role": profile.role,
                "style_hint": profile.style_hint,
                "capabilities": profile.capabilities,
                "notes": profile.notes,
                "preferred_voice": profile.preferred_voice,
                "sample_count": profile.sample_count,
                "captured_utterance": rec.text,
            }
        )

    def _call_speaker_list_voiceprints(self) -> dict[str, Any]:
        profiles = self.speaker_store.list_profiles()
        recent = self.speaker_store.recent_utterances()
        return self._tool_result(
            {
                "profiles": profiles,
                "count": len(profiles),
                "recent_utterances": recent,
            }
        )

    def _call_speaker_update_voiceprint(self, args: dict[str, Any]) -> dict[str, Any]:
        name_or_id = str(args.get("name_or_id") or "").strip()
        if not name_or_id:
            return self._tool_result(
                {"error": "invalid_arguments", "message": "'name_or_id' is required."},
                is_error=True,
            )
        raw_caps = args.get("capabilities")
        capabilities = (
            [str(c).strip() for c in raw_caps if str(c).strip()]
            if isinstance(raw_caps, list)
            else None
        )
        raw_voice = args.get("preferred_voice")
        preferred_voice: str | None = None
        if isinstance(raw_voice, str):
            if raw_voice.strip():
                try:
                    preferred_voice = resolve_edge_voice(raw_voice.strip()).id
                except ValueError as exc:
                    return self._tool_result(
                        {"error": "invalid_voice", "message": str(exc)}, is_error=True
                    )
            else:
                preferred_voice = ""
        updated = self.speaker_store.update_profile(
            name_or_id,
            role=str(args["role"]) if "role" in args and args["role"] is not None else None,
            style_hint=str(args["style_hint"]) if "style_hint" in args and args["style_hint"] is not None else None,
            capabilities=capabilities,
            notes=str(args["notes"]) if "notes" in args and args["notes"] is not None else None,
            preferred_voice=preferred_voice,
        )
        if updated is None:
            return self._tool_result(
                {"error": "not_found", "message": f"No voiceprint found for {name_or_id!r}."},
                is_error=True,
            )
        return self._tool_result(
            {
                "status": "updated",
                "id": updated.id,
                "name": updated.name,
                "role": updated.role,
                "style_hint": updated.style_hint,
                "capabilities": updated.capabilities,
                "notes": updated.notes,
                "preferred_voice": updated.preferred_voice,
            }
        )

    def _call_speaker_delete_voiceprint(self, args: dict[str, Any]) -> dict[str, Any]:
        target = str(args.get("name_or_id") or "").strip()
        removed = self.speaker_store.delete_profile(target)
        return self._tool_result({"removed": removed, "name_or_id": target})

    @classmethod
    def _validate_model_ref(cls, raw_model: str) -> str:
        cleaned = raw_model.strip()
        lowered = re.sub(r"\s+", " ", cleaned.lower())
        if lowered in MODEL_ALIASES:
            return MODEL_ALIASES[lowered]
        compact = lowered.replace(" ", "")
        if compact in MODEL_ALIASES:
            return MODEL_ALIASES[compact]
        if not cleaned or len(cleaned) > 120 or not _MODEL_REF_RE.fullmatch(cleaned):
            raise ValueError(
                f"Invalid model {raw_model!r}; expected 'provider/model[#variant]' or alias like 'gpt', 'gpt 6.2', 'hydrogen', 'nano'."
            )
        return cleaned

    def _call_speaker_list_models(self) -> dict[str, Any]:
        return self._tool_result(
            {
                "current_model": self._config.llm.providers.opencode.model,
                "models": list(AVAILABLE_MODELS),
            }
        )

    async def _call_speaker_set_model(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_model = str(args.get("model") or "").strip()
        persist = bool(args.get("persist", True))
        try:
            new_model = self._validate_model_ref(raw_model)
        except ValueError as exc:
            return self._tool_result(
                {"error": "invalid_model", "message": str(exc)},
                is_error=True,
            )

        self._config.llm.providers.opencode.model = new_model
        persisted = False
        if persist and self._config_path is not None:
            await asyncio.to_thread(
                persist_runtime_settings,
                self._config_path,
                model=new_model,
            )
            persisted = True
        if self._on_model_changed is not None:
            self._on_model_changed(new_model)
        label = next((m["label"] for m in AVAILABLE_MODELS if m["id"] == new_model), new_model)
        return self._tool_result(
            {
                "status": "updated",
                "model": new_model,
                "label": label,
                "persisted": persisted,
            }
        )

    @staticmethod
    def _parse_memory_payload(body: dict[str, Any]) -> tuple[str, str, str]:
        category = str(body.get("category") or "fact").strip() or "fact"
        key = str(body.get("key") or "").strip()
        value = str(body.get("value") or "").strip()
        if key and value:
            return key, value, category
        fact = str(body.get("fact") or "").strip()
        if not fact:
            raise ValueError("Provide either 'fact' or both 'key' and 'value'.")
        for sep in ("：", ":"):
            if sep in fact:
                left, right = fact.split(sep, 1)
                if left.strip() and right.strip():
                    return left.strip()[:80], right.strip(), category
        derived_key = re.sub(r"\s+", " ", fact)[:24].strip() or "偏好"
        return derived_key, fact, category

    def _build_capabilities(self) -> list[dict[str, Any]]:
        cfg = self._config
        outlook_client_id_set = bool(read_env_secret(cfg.tools.outlook.client_id_env))
        outlook_cache_ok = Path(cfg.tools.outlook.token_cache_file).expanduser().is_file()
        edge_app_found = (
            Path("/Applications/Microsoft Edge.app").exists() or shutil.which("msedge") is not None
        )
        try:
            codex_bin = resolve_binary(cfg.llm.providers.codex.binary)
            codex_ready = True
        except RuntimeError:
            codex_bin = ""
            codex_ready = False
        mem_count = len(self.memory_store.search("", limit=100))

        return [
            {
                "id": "weather",
                "name": "Weather 天气查询",
                "enabled": True,
                "status_label": "已就绪",
                "description": "实时天气与未来7天预报（wttr.in + Open-Meteo）",
                "detail": "weather_query · 免密钥直连",
            },
            {
                "id": "web_search",
                "name": "Web Search 联网搜索",
                "enabled": True,
                "status_label": "已就绪",
                "description": "Bing 中文实时新闻、百科与价格检索",
                "detail": "web_search · zh-CN",
            },
            {
                "id": "memory",
                "name": "Memory 长期记忆",
                "enabled": bool(cfg.memory.enabled),
                "status_label": "已启用" if cfg.memory.enabled else "已禁用",
                "description": "持久化个人偏好与常用事实（0600 隔离存储）",
                "detail": f"已存 {mem_count} 条 · 上限注入 {cfg.memory.max_prompt_items} 条",
            },
            {
                "id": "voice_switching",
                "name": "Voice Switching 音色切换",
                "enabled": True,
                "status_label": "已就绪",
                "description": "支持对话与控制台实时切换 Edge-TTS 中文音色与语速",
                "detail": f"{len(EDGE_ZH_VOICES)} 个精选音色 · 当前 {cfg.speaker.edge_voice}",
            },
            {
                "id": "xiaohongshu",
                "name": "Xiaohongshu MCP 小红书",
                "enabled": bool(cfg.tools.xiaohongshu.enabled),
                "status_label": "已启用" if cfg.tools.xiaohongshu.enabled else "未启用",
                "description": "搜索小红书笔记与详情解析",
                "detail": cfg.tools.xiaohongshu.mcp_url,
            },
            {
                "id": "browser",
                "name": "Edge Browser MCP 浏览器",
                "enabled": bool(cfg.tools.browser.enabled or edge_app_found),
                "status_label": "MCP 已启用" if cfg.tools.browser.enabled else ("本地 Edge 就绪" if edge_app_found else "HTTP 摘要可用"),
                "description": "网页正文抓取与 Microsoft Edge 打开/页面快照",
                "detail": cfg.tools.browser.mcp_url if cfg.tools.browser.enabled else "browser_fetch_summary + browser_open_edge",
            },
            {
                "id": "outlook",
                "name": "Outlook 邮件助手",
                "enabled": bool(cfg.tools.outlook.enabled),
                "status_label": "已启用" if cfg.tools.outlook.enabled else "已禁用",
                "description": "读取最近邮件并在语音二次确认后发送邮件",
                "detail": f"client_id={'已配置' if outlook_client_id_set else '未配置'} · cache={'存在' if outlook_cache_ok else '未登录'}",
            },
            {
                "id": "codex",
                "name": "Codex 代码分析",
                "enabled": codex_ready,
                "status_label": "已就绪" if codex_ready else "未检测到 CLI",
                "description": "调用本地只读 Codex CLI 分析代码与技术问题",
                "detail": codex_bin or cfg.llm.providers.codex.binary,
            },
            {
                "id": "mac_local",
                "name": "Mac Disk / Files 本机访问",
                "enabled": True,
                "status_label": "已就绪",
                "description": "受限读取 ~/Desktop, ~/Downloads, ~/Documents 与磁盘空间",
                "detail": "mac_disk_usage · mac_list_recent_files · mac_read_text_file",
            },
        ]

    def _build_overview_payload(self) -> dict[str, Any]:
        cfg = self._config
        current_model = cfg.llm.providers.opencode.model
        models = list(AVAILABLE_MODELS)
        if current_model and all(m["id"] != current_model for m in models):
            models.insert(0, {"id": current_model, "label": f"{current_model} (当前)"})
        voice_name = next(
            (v.name for v in EDGE_ZH_VOICES if v.id == cfg.speaker.edge_voice),
            cfg.speaker.edge_voice,
        )
        return {
            "assistant_name": cfg.persona.assistant_name,
            "model": current_model,
            "default_provider": cfg.llm.default_provider,
            "available_models": models,
            "voice": cfg.speaker.edge_voice,
            "voice_name": voice_name,
            "rate": cfg.speaker.edge_rate,
            "available_voices": list_voice_profiles_dicts(),
            "tts_provider": cfg.speaker.tts_provider,
            "speaker_model": cfg.speaker.model,
            "asr_provider": cfg.speech.asr_provider,
            "asr_model": cfg.speech.asr_model,
            "asr_language": cfg.speech.asr_language,
            "aec_enabled": bool(cfg.speech.aec_enabled),
            "capture_channels": cfg.speech.capture_channels,
            "wakewords": {
                "local": list(cfg.speech.local_keywords),
                "english": list(cfg.speech.en_keywords),
                "en_kws_enabled": bool(cfg.speech.en_kws_enabled),
            },
            "reminders_config": {
                "enabled": bool(cfg.reminders.enabled),
                "quiet_start_hhmm": cfg.reminders.quiet_start_hhmm,
                "quiet_end_hhmm": cfg.reminders.quiet_end_hhmm,
            },
            "toggles": {
                "reminders": bool(cfg.reminders.enabled),
                "memory": bool(cfg.memory.enabled),
                "xiaohongshu": bool(cfg.tools.xiaohongshu.enabled),
                "browser": bool(cfg.tools.browser.enabled),
                "outlook": bool(cfg.tools.outlook.enabled),
                "require_host_voiceprint": bool(cfg.tools.require_host_voiceprint_for_sensitive_tools),
            },
            "config_path": str(self._config_path) if self._config_path else "config.yaml",
            "stats": self.history_store.stats(),
            "daily_turns": self.history_store.daily_turns(days=14),
            "usage": read_local_opencode_usage(cfg.llm.providers.opencode.agent_home, days=14),
            "capabilities": self._build_capabilities(),
        }

    @staticmethod
    def _sanitize_voiceprint(item: dict[str, Any], fallback_id: str = "") -> dict[str, Any]:
        vp_id = str(item.get("id") or fallback_id or item.get("name") or "").strip()
        embeddings = item.get("embeddings") or item.get("embedding")
        sample_count = item.get("sample_count")
        if sample_count is None and isinstance(embeddings, list):
            sample_count = len(embeddings) if embeddings and isinstance(embeddings[0], list) else (1 if embeddings else 0)
        return {
            "id": vp_id,
            "name": str(item.get("name") or vp_id or "未命名"),
            "role": str(item.get("role") or ""),
            "capabilities": list(item.get("capabilities") or []) if isinstance(item.get("capabilities"), list) else [],
            "notes": str(item.get("notes") or ""),
            "preferred_voice": item.get("preferred_voice"),
            "sample_count": int(sample_count or 0),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
        }

    def _read_voiceprints(self) -> list[dict[str, Any]]:
        if self._speakers_path is None:
            return self.speaker_store.list_profiles()
        if not self._speakers_path.is_file():
            return []
        try:
            raw = json.loads(self._speakers_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if isinstance(raw, list):
            return [self._sanitize_voiceprint(x) for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            nested = raw.get("speakers") or raw.get("voiceprints")
            if isinstance(nested, list):
                return [self._sanitize_voiceprint(x) for x in nested if isinstance(x, dict)]
            return [
                self._sanitize_voiceprint(val, fallback_id=str(k))
                for k, val in raw.items()
                if isinstance(val, dict)
            ]
        return []

    def _delete_voiceprint(self, vp_id: str) -> bool:
        target = vp_id.strip()
        if not target:
            return False
        if self._speakers_path is None:
            return self.speaker_store.delete_profile(target)
        if not self._speakers_path.is_file():
            return False
        try:
            raw = json.loads(self._speakers_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        removed = False
        updated: Any = raw
        if isinstance(raw, list):
            kept = [
                x
                for x in raw
                if not (isinstance(x, dict) and (str(x.get("id") or "") == target or str(x.get("name") or "") == target))
            ]
            removed = len(kept) < len(raw)
            updated = kept
        elif isinstance(raw, dict):
            for key in ("speakers", "voiceprints"):
                if isinstance(raw.get(key), list):
                    items = raw[key]
                    kept = [
                        x
                        for x in items
                        if not (isinstance(x, dict) and (str(x.get("id") or "") == target or str(x.get("name") or "") == target))
                    ]
                    if len(kept) < len(items):
                        removed = True
                        updated = {**raw, key: kept}
                    break
            else:
                if target in raw:
                    updated = {k: v for k, v in raw.items() if k != target}
                    removed = True
                else:
                    kept_map = {
                        k: v
                        for k, v in raw.items()
                        if not (isinstance(v, dict) and (str(v.get("id") or "") == target or str(v.get("name") or "") == target))
                    }
                    if len(kept_map) < len(raw):
                        removed = True
                        updated = kept_map

        if not removed:
            return False

        tmp_path = self._speakers_path.with_suffix(".tmp")
        payload = json.dumps(updated, ensure_ascii=False, indent=2) + "\n"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, self._speakers_path)
        os.chmod(self._speakers_path, 0o600)
        return True

    def _read_reminders(self) -> list[dict[str, Any]]:
        if self._reminders_path is None:
            return [asdict(e) for e in self.reminder_store.list_entries()]
        if not self._reminders_path.is_file():
            return []
        try:
            raw = json.loads(self._reminders_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            nested = raw.get("reminders")
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
        return []

    async def start_http_server(
        self,
        host: str = "127.0.0.1",
        port: int = 4397,
        *,
        auth_token: str | None = None,
        console_password: str | None = None,
    ) -> tuple[web.AppRunner, int]:
        """Start an in-process Streamable-HTTP MCP endpoint on `/mcp` and Web Console on `/`."""
        effective_console_password = (
            console_password if console_password is not None else self._console_password
        )
        session_secret = hashlib.sha256(
            (effective_console_password or auth_token or secrets.token_hex(32)).encode("utf-8")
        ).digest()
        login_failures: dict[str, list[float]] = {}

        def _extract_hostname(host_header: str) -> str:
            val = host_header.strip().lower()
            if val.startswith("["):
                end = val.find("]")
                return val[1:end] if end != -1 else val
            return val.split(":", 1)[0]

        def _is_direct_local_request(request: web.Request) -> bool:
            remote = (request.remote or "").strip()
            if remote not in _LOOPBACK_REMOTES:
                return False
            if any(request.headers.get(h) for h in _PROXY_HEADER_NAMES):
                return False
            host_header = request.headers.get("Host", "")
            if host_header and _extract_hostname(host_header) not in _LOOPBACK_HOSTS:
                return False
            return True

        def _make_session_cookie(ttl_sec: int = 7 * 86400) -> str:
            exp = str(int(time.time()) + ttl_sec)
            sig = hmac.new(session_secret, exp.encode("ascii"), hashlib.sha256).hexdigest()
            return f"{exp}.{sig}"

        def _verify_session_cookie(raw_cookie: str) -> bool:
            if "." not in raw_cookie:
                return False
            exp_str, sig = raw_cookie.split(".", 1)
            if not exp_str.isdigit():
                return False
            if int(exp_str) < int(time.time()):
                return False
            expected = hmac.new(session_secret, exp_str.encode("ascii"), hashlib.sha256).hexdigest()
            return hmac.compare_digest(sig, expected)

        def _has_valid_console_auth(request: web.Request) -> bool:
            if _is_direct_local_request(request):
                return True
            cookies = getattr(request, "cookies", {}) or {}
            cookie_val = cookies.get(_CONSOLE_COOKIE_NAME, "")
            if cookie_val and _verify_session_cookie(cookie_val):
                return True
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                supplied = auth_header[7:].strip()
                if effective_console_password and hmac.compare_digest(
                    supplied, effective_console_password
                ):
                    return True
                if auth_token and hmac.compare_digest(supplied, auth_token):
                    return True
            return False

        def _check_security(request: web.Request) -> web.Response | None:
            # Reject proxy-forwarded or non-loopback requests on /mcp so tunnels (e.g. cloudflared)
            # exposing the Web Console never expose internal MCP tool execution.
            if not _is_direct_local_request(request):
                return web.json_response(
                    {"error": "forbidden", "message": "MCP endpoint is local-only"},
                    status=403,
                )
            # Reject any browser cross-origin / no-cors simple requests immediately.
            if request.headers.get("Origin"):
                return web.json_response(
                    {"error": "forbidden", "message": "Cross-origin requests are not allowed"},
                    status=403,
                )
            if auth_token:
                auth_header = request.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    return web.json_response(
                        {"error": "unauthorized", "message": "Missing Bearer token"},
                        status=401,
                    )
                supplied = auth_header[7:].strip()
                if not hmac.compare_digest(supplied, auth_token):
                    return web.json_response(
                        {"error": "unauthorized", "message": "Invalid Bearer token"},
                        status=401,
                    )
            return None

        def _check_console_read_security(request: web.Request) -> web.Response | None:
            if not _has_valid_console_auth(request):
                return web.json_response(
                    {"error": "unauthorized", "message": "Console login required"},
                    status=401,
                )
            return None

        def _check_console_write_security(request: web.Request) -> web.Response | None:
            origin = request.headers.get("Origin")
            if origin:
                origin_host = (urlparse(origin).hostname or "").lower()
                req_host = _extract_hostname(request.headers.get("Host", ""))
                allowed_hosts = set(_LOOPBACK_HOSTS)
                if req_host:
                    allowed_hosts.add(req_host)
                if origin_host not in allowed_hosts:
                    return web.json_response(
                        {"error": "forbidden", "message": "Cross-origin requests are not allowed"},
                        status=403,
                    )
            if not _has_valid_console_auth(request):
                return web.json_response(
                    {"error": "unauthorized", "message": "Console login required"},
                    status=401,
                )
            return None

        async def handle_mcp_post(request: web.Request) -> web.Response:
            denied = _check_security(request)
            if denied is not None:
                return denied
            try:
                raw_body = await request.text()
                message = json.loads(raw_body)
            except Exception:
                err = self._error(None, -32700, "Parse error")
                return web.json_response(err, status=400)

            if not isinstance(message, dict):
                return web.json_response(self._error(None, -32600, "Invalid Request"), status=400)

            session_id = request.headers.get("Mcp-Session-Id") or uuid.uuid4().hex
            self._http_sessions.add(session_id)
            headers = {"Mcp-Session-Id": session_id}

            response = await self.handle_message(message)
            if response is None:
                return web.Response(status=202, headers=headers)
            return web.json_response(response, status=200, headers=headers)

        async def handle_mcp_get(request: web.Request) -> web.Response:
            denied = _check_security(request)
            if denied is not None:
                return denied
            return web.Response(status=405, headers={"Allow": "POST, DELETE"})

        async def handle_mcp_delete(request: web.Request) -> web.Response:
            denied = _check_security(request)
            if denied is not None:
                return denied
            session_id = request.headers.get("Mcp-Session-Id")
            if session_id:
                self._http_sessions.discard(session_id)
            return web.json_response({"closed": True}, status=200)

        async def handle_health(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True, "server": SERVER_INFO})

        async def handle_console_index(_request: web.Request) -> web.Response:
            return web.Response(
                text=load_console_html(),
                content_type="text/html",
                charset="utf-8",
                headers={"Cache-Control": "no-store"},
            )

        async def handle_api_auth_status(request: web.Request) -> web.Response:
            is_local = _is_direct_local_request(request)
            authed = _has_valid_console_auth(request)
            return web.json_response(
                {
                    "authenticated": authed,
                    "auth_required": not is_local,
                    "is_local": is_local,
                }
            )

        async def handle_api_login(request: web.Request) -> web.Response:
            origin = request.headers.get("Origin")
            if origin:
                origin_host = (urlparse(origin).hostname or "").lower()
                req_host = _extract_hostname(request.headers.get("Host", ""))
                allowed_hosts = set(_LOOPBACK_HOSTS)
                if req_host:
                    allowed_hosts.add(req_host)
                if origin_host not in allowed_hosts:
                    return web.json_response(
                        {"error": "forbidden", "message": "Cross-origin login is not allowed"},
                        status=403,
                    )
            if not effective_console_password:
                return web.json_response(
                    {"error": "auth_unavailable", "message": "Console remote password is not configured"},
                    status=503,
                )
            client_key = (
                request.headers.get("CF-Connecting-IP")
                or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                or (request.remote or "unknown")
            )
            now = time.monotonic()
            recent = [t for t in login_failures.get(client_key, []) if (now - t) < 60.0]
            if len(recent) >= 5:
                login_failures[client_key] = recent
                return web.json_response(
                    {"error": "rate_limited", "message": "尝试次数过多，请 1 分钟后再试"},
                    status=429,
                )
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "invalid_json", "message": "Invalid JSON body"}, status=400)
            supplied = str(body.get("password") or "").strip() if isinstance(body, dict) else ""
            if not supplied or not hmac.compare_digest(supplied, effective_console_password):
                recent.append(now)
                login_failures[client_key] = recent
                return web.json_response(
                    {"error": "invalid_password", "message": "访问口令错误"},
                    status=401,
                )
            login_failures.pop(client_key, None)
            resp = web.json_response({"authenticated": True})
            is_https = (
                request.secure
                or request.headers.get("X-Forwarded-Proto", "").lower() == "https"
            )
            resp.set_cookie(
                _CONSOLE_COOKIE_NAME,
                _make_session_cookie(),
                max_age=7 * 86400,
                path="/",
                httponly=True,
                samesite="Lax",
                secure=is_https,
            )
            return resp

        async def handle_api_logout(_request: web.Request) -> web.Response:
            resp = web.json_response({"authenticated": False})
            resp.del_cookie(_CONSOLE_COOKIE_NAME, path="/")
            return resp

        async def handle_api_overview(request: web.Request) -> web.Response:
            denied = _check_console_read_security(request)
            if denied is not None:
                return denied
            return web.json_response(self._build_overview_payload())

        async def handle_api_history_get(request: web.Request) -> web.Response:
            denied = _check_console_read_security(request)
            if denied is not None:
                return denied
            raw_limit = request.query.get("limit", "50")
            try:
                limit = max(1, min(500, int(raw_limit)))
            except ValueError:
                return web.json_response({"error": "invalid_limit", "message": "Invalid limit parameter"}, status=400)
            speaker = request.query.get("speaker", "")
            entries = self.history_store.list(limit=limit, speaker=speaker)
            return web.json_response(
                {
                    "turns": [asdict(e) for e in entries],
                    "count": len(entries),
                    "stats": self.history_store.stats(),
                }
            )

        async def handle_api_history_delete(request: web.Request) -> web.Response:
            denied = _check_console_write_security(request)
            if denied is not None:
                return denied
            cleared = self.history_store.clear()
            return web.json_response({"cleared": cleared, "stats": self.history_store.stats()})

        async def handle_api_memories_get(request: web.Request) -> web.Response:
            denied = _check_console_read_security(request)
            if denied is not None:
                return denied
            q = request.query.get("q", "")
            raw_limit = request.query.get("limit", "100")
            try:
                limit = max(1, min(200, int(raw_limit)))
            except ValueError:
                limit = 100
            items = self.memory_store.search(q, limit=limit) if self._config.memory.enabled else []
            return web.json_response(
                {
                    "memories": [asdict(e) for e in items],
                    "count": len(items),
                    "enabled": bool(self._config.memory.enabled),
                }
            )

        async def handle_api_memories_post(request: web.Request) -> web.Response:
            denied = _check_console_write_security(request)
            if denied is not None:
                return denied
            if not self._config.memory.enabled:
                return web.json_response(
                    {"error": "memory_disabled", "message": "Long-term memory is disabled."},
                    status=400,
                )
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "invalid_json", "message": "Invalid JSON body"}, status=400)
            if not isinstance(body, dict):
                return web.json_response({"error": "invalid_json", "message": "Expected JSON object"}, status=400)
            try:
                key, value, category = self._parse_memory_payload(body)
                entry = self.memory_store.save(key, value, category=category)
            except (ValueError, MemorySecurityError) as exc:
                return web.json_response({"error": "memory_rejected", "message": str(exc)}, status=400)
            if self._on_memory_changed is not None:
                self._on_memory_changed()
            return web.json_response({"memory": asdict(entry)}, status=201)

        async def handle_api_memories_delete(request: web.Request) -> web.Response:
            denied = _check_console_write_security(request)
            if denied is not None:
                return denied
            mem_id = request.match_info.get("id", "").strip()
            removed = self.memory_store.forget(mem_id)
            if not removed:
                return web.json_response({"error": "not_found", "removed": False, "id": mem_id}, status=404)
            if self._on_memory_changed is not None:
                self._on_memory_changed()
            return web.json_response({"removed": True, "id": mem_id})

        async def handle_api_settings_post(request: web.Request) -> web.Response:
            denied = _check_console_write_security(request)
            if denied is not None:
                return denied
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "invalid_json", "message": "Invalid JSON body"}, status=400)
            if not isinstance(body, dict):
                return web.json_response({"error": "invalid_json", "message": "Expected JSON object"}, status=400)

            has_model = "model" in body and str(body.get("model") or "").strip() != ""
            has_voice = "edge_voice" in body and str(body.get("edge_voice") or "").strip() != ""
            has_rate = "edge_rate" in body and str(body.get("edge_rate") or "").strip() != ""
            has_local_kw = "local_keywords" in body and body.get("local_keywords") is not None
            has_en_kw = "en_keywords" in body and body.get("en_keywords") is not None
            has_en_enabled = "en_kws_enabled" in body and body.get("en_kws_enabled") is not None
            if not (has_model or has_voice or has_rate or has_local_kw or has_en_kw or has_en_enabled):
                return web.json_response(
                    {
                        "error": "invalid_arguments",
                        "message": "Provide at least one setting field (model, edge_voice, edge_rate, local_keywords, en_keywords, or en_kws_enabled).",
                    },
                    status=400,
                )

            new_model = self._config.llm.providers.opencode.model
            new_voice = self._config.speaker.edge_voice
            new_rate = self._config.speaker.edge_rate
            new_local_kw = list(self._config.speech.local_keywords)
            new_en_kw = list(self._config.speech.en_keywords)
            new_en_enabled = bool(self._config.speech.en_kws_enabled)

            def _clean_kw_list(raw_val: Any, *, uppercase: bool = False) -> list[str]:
                if isinstance(raw_val, str):
                    parts = re.split(r"[,，、;\n]+", raw_val)
                elif isinstance(raw_val, list):
                    parts = [str(item) for item in raw_val]
                else:
                    raise ValueError("Keywords must be a list of strings or comma-separated string.")
                cleaned: list[str] = []
                for p in parts:
                    s = re.sub(r"\s+", " ", p.strip())
                    if not s:
                        continue
                    if len(s) > 40:
                        raise ValueError("Each wake keyword must be at most 40 characters.")
                    if uppercase:
                        s = s.upper()
                    if s not in cleaned:
                        cleaned.append(s)
                if len(cleaned) > 20:
                    raise ValueError("At most 20 wake keywords are allowed per list.")
                return cleaned

            try:
                if has_model:
                    new_model = self._validate_model_ref(str(body["model"]))
                if has_voice:
                    new_voice = resolve_edge_voice(str(body["edge_voice"])).id
                if has_rate:
                    new_rate = normalize_edge_rate(str(body["edge_rate"]))
                if has_local_kw:
                    new_local_kw = _clean_kw_list(body["local_keywords"], uppercase=False)
                    if not new_local_kw:
                        raise ValueError("At least one Chinese/local wake keyword is required.")
                if has_en_kw:
                    new_en_kw = _clean_kw_list(body["en_keywords"], uppercase=True)
                if has_en_enabled:
                    if not isinstance(body["en_kws_enabled"], bool):
                        raise ValueError("en_kws_enabled must be a boolean.")
                    new_en_enabled = body["en_kws_enabled"]
                if (has_en_kw or has_en_enabled) and new_en_enabled and not new_en_kw:
                    raise ValueError("At least one English wake keyword is required when English KWS is enabled.")
            except ValueError as exc:
                return web.json_response({"error": "invalid_settings", "message": str(exc)}, status=400)

            should_persist = self._config_path is not None
            persisted = False
            if should_persist:
                try:
                    await asyncio.to_thread(
                        persist_runtime_settings,
                        self._config_path,
                        model=new_model if has_model else None,
                        edge_voice=new_voice if has_voice else None,
                        edge_rate=new_rate if has_rate else None,
                        local_keywords=new_local_kw if has_local_kw else None,
                        en_keywords=new_en_kw if has_en_kw else None,
                        en_kws_enabled=new_en_enabled if has_en_enabled else None,
                    )
                    persisted = True
                except OSError as exc:
                    return web.json_response(
                        {"error": "persist_failed", "message": f"Failed saving config.yaml: {exc}"},
                        status=500,
                    )

            self._config.llm.providers.opencode.model = new_model
            self._config.speaker.edge_voice = new_voice
            self._config.speaker.edge_rate = new_rate
            self._config.speech.local_keywords = new_local_kw
            self._config.speech.en_keywords = new_en_kw
            self._config.speech.en_kws_enabled = new_en_enabled

            if (has_voice or has_rate) and self._on_voice_changed is not None:
                self._on_voice_changed(new_voice, new_rate)
            if has_model and self._on_model_changed is not None:
                self._on_model_changed(new_model)
            if (has_local_kw or has_en_kw or has_en_enabled) and self._on_wakewords_changed is not None:
                self._on_wakewords_changed(new_local_kw, new_en_kw, new_en_enabled)

            return web.json_response(
                {
                    "status": "updated",
                    "model": new_model,
                    "edge_voice": new_voice,
                    "edge_rate": new_rate,
                    "wakewords": {
                        "local": new_local_kw,
                        "english": new_en_kw,
                        "en_kws_enabled": new_en_enabled,
                    },
                    "persisted": persisted,
                }
            )

        async def handle_api_voiceprints_get(request: web.Request) -> web.Response:
            denied = _check_console_read_security(request)
            if denied is not None:
                return denied
            items = self._read_voiceprints()
            return web.json_response({"voiceprints": items, "count": len(items)})

        async def handle_api_voiceprints_delete(request: web.Request) -> web.Response:
            denied = _check_console_write_security(request)
            if denied is not None:
                return denied
            vp_id = request.match_info.get("id", "").strip()
            removed = self._delete_voiceprint(vp_id)
            if not removed:
                return web.json_response({"error": "not_found", "removed": False, "id": vp_id}, status=404)
            return web.json_response({"removed": True, "id": vp_id})

        async def handle_api_reminders_get(request: web.Request) -> web.Response:
            denied = _check_console_read_security(request)
            if denied is not None:
                return denied
            items = self._read_reminders()
            return web.json_response(
                {
                    "reminders": items,
                    "count": len(items),
                    "enabled": bool(self._config.reminders.enabled),
                    "quiet_start_hhmm": self._config.reminders.quiet_start_hhmm,
                    "quiet_end_hhmm": self._config.reminders.quiet_end_hhmm,
                }
            )

        async def handle_api_reminders_post(request: web.Request) -> web.Response:
            denied = _check_console_write_security(request)
            if denied is not None:
                return denied
            if not self._config.reminders.enabled:
                return web.json_response(
                    {"error": "reminders_disabled", "message": "Proactive reminders are disabled."},
                    status=400,
                )
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "invalid_json", "message": "Invalid JSON body"}, status=400)
            if not isinstance(body, dict):
                return web.json_response({"error": "invalid_json", "message": "Expected JSON object"}, status=400)

            message = str(body.get("message") or "").strip()
            kind = str(body.get("kind") or "reminder").strip()
            delay_seconds = body.get("delay_seconds")
            trigger_at_iso = body.get("trigger_at_iso")
            daily_hhmm = body.get("daily_hhmm")

            if delay_seconds is not None and delay_seconds != "":
                if isinstance(delay_seconds, bool):
                    return web.json_response(
                        {"error": "invalid_reminder", "message": "delay_seconds must be an integer."},
                        status=400,
                    )
                try:
                    delay_seconds = int(delay_seconds)
                except (TypeError, ValueError):
                    return web.json_response(
                        {"error": "invalid_reminder", "message": "delay_seconds must be an integer."},
                        status=400,
                    )
            else:
                delay_seconds = None

            if trigger_at_iso is not None:
                if not isinstance(trigger_at_iso, str):
                    return web.json_response(
                        {"error": "invalid_reminder", "message": "trigger_at_iso must be an ISO string."},
                        status=400,
                    )
                trigger_at_iso = trigger_at_iso.strip() or None

            if daily_hhmm is not None:
                if not isinstance(daily_hhmm, str):
                    return web.json_response(
                        {"error": "invalid_reminder", "message": "daily_hhmm must be HH:MM string."},
                        status=400,
                    )
                daily_hhmm = daily_hhmm.strip() or None

            try:
                entry = self.reminder_store.add(
                    message,
                    delay_seconds=delay_seconds,
                    trigger_at_iso=trigger_at_iso,
                    daily_hhmm=daily_hhmm,
                    kind=kind,
                )
            except ValueError as exc:
                return web.json_response({"error": "invalid_reminder", "message": str(exc)}, status=400)
            return web.json_response({"reminder": asdict(entry)}, status=201)

        async def handle_api_reminders_delete(request: web.Request) -> web.Response:
            denied = _check_console_write_security(request)
            if denied is not None:
                return denied
            rem_id = request.match_info.get("id", "").strip()
            removed = self.reminder_store.cancel(rem_id, exact_id=True)
            if not removed:
                return web.json_response({"error": "not_found", "removed": False, "id": rem_id}, status=404)
            return web.json_response({"removed": True, "id": rem_id, "count": removed})

        async def handle_api_capabilities_post(request: web.Request) -> web.Response:
            denied = _check_console_write_security(request)
            if denied is not None:
                return denied
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "invalid_json", "message": "Invalid JSON body"}, status=400)
            if not isinstance(body, dict):
                return web.json_response({"error": "invalid_json", "message": "Expected JSON object"}, status=400)

            hhmm_re = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
            updates: dict[str, Any] = {}
            for bool_field in (
                "reminders_enabled",
                "memory_enabled",
                "xiaohongshu_enabled",
                "browser_enabled",
                "outlook_enabled",
                "require_host_voiceprint",
            ):
                if bool_field in body and body[bool_field] is not None:
                    if not isinstance(body[bool_field], bool):
                        return web.json_response(
                            {"error": "invalid_settings", "message": f"{bool_field} must be a boolean."},
                            status=400,
                        )
                    updates[bool_field] = body[bool_field]

            for time_field in ("quiet_start_hhmm", "quiet_end_hhmm"):
                if time_field in body and body[time_field] is not None:
                    val = str(body[time_field]).strip()
                    if not hhmm_re.match(val):
                        return web.json_response(
                            {"error": "invalid_settings", "message": f"{time_field} must be HH:MM (24h)."},
                            status=400,
                        )
                    updates[time_field] = val

            if not updates:
                return web.json_response(
                    {"error": "invalid_arguments", "message": "Provide at least one capability or quiet-hours field."},
                    status=400,
                )

            persisted = False
            if self._config_path is not None:
                try:
                    await asyncio.to_thread(
                        persist_runtime_settings,
                        self._config_path,
                        **updates,
                    )
                    persisted = True
                except OSError as exc:
                    return web.json_response(
                        {"error": "persist_failed", "message": f"Failed saving config.yaml: {exc}"},
                        status=500,
                    )

            if "reminders_enabled" in updates:
                self._config.reminders.enabled = updates["reminders_enabled"]
            if "quiet_start_hhmm" in updates:
                self._config.reminders.quiet_start_hhmm = updates["quiet_start_hhmm"]
            if "quiet_end_hhmm" in updates:
                self._config.reminders.quiet_end_hhmm = updates["quiet_end_hhmm"]
            if "memory_enabled" in updates:
                self._config.memory.enabled = updates["memory_enabled"]
            if "xiaohongshu_enabled" in updates:
                self._config.tools.xiaohongshu.enabled = updates["xiaohongshu_enabled"]
            if "browser_enabled" in updates:
                self._config.tools.browser.enabled = updates["browser_enabled"]
            if "outlook_enabled" in updates:
                self._config.tools.outlook.enabled = updates["outlook_enabled"]
            if "require_host_voiceprint" in updates:
                self._config.tools.require_host_voiceprint_for_sensitive_tools = updates[
                    "require_host_voiceprint"
                ]

            if (
                any(k in updates for k in ("reminders_enabled", "quiet_start_hhmm", "quiet_end_hhmm"))
                and self._on_reminders_config_changed is not None
            ):
                self._on_reminders_config_changed(
                    self._config.reminders.enabled,
                    self._config.reminders.quiet_start_hhmm,
                    self._config.reminders.quiet_end_hhmm,
                )
            if self._on_capabilities_changed is not None:
                self._on_capabilities_changed()

            return web.json_response(
                {
                    "status": "updated",
                    "persisted": persisted,
                    "reminders_config": {
                        "enabled": bool(self._config.reminders.enabled),
                        "quiet_start_hhmm": self._config.reminders.quiet_start_hhmm,
                        "quiet_end_hhmm": self._config.reminders.quiet_end_hhmm,
                    },
                    "toggles": {
                        "reminders": bool(self._config.reminders.enabled),
                        "memory": bool(self._config.memory.enabled),
                        "xiaohongshu": bool(self._config.tools.xiaohongshu.enabled),
                        "browser": bool(self._config.tools.browser.enabled),
                        "outlook": bool(self._config.tools.outlook.enabled),
                        "require_host_voiceprint": bool(
                            self._config.tools.require_host_voiceprint_for_sensitive_tools
                        ),
                    },
                    "capabilities": self._build_capabilities(),
                }
            )

        app = web.Application()
        app.router.add_post("/mcp", handle_mcp_post)
        app.router.add_get("/mcp", handle_mcp_get)
        app.router.add_delete("/mcp", handle_mcp_delete)
        app.router.add_get("/health", handle_health)

        app.router.add_get("/", handle_console_index)
        app.router.add_get("/api/auth/status", handle_api_auth_status)
        app.router.add_post("/api/login", handle_api_login)
        app.router.add_post("/api/logout", handle_api_logout)
        app.router.add_get("/api/overview", handle_api_overview)
        app.router.add_get("/api/history", handle_api_history_get)
        app.router.add_delete("/api/history", handle_api_history_delete)
        app.router.add_get("/api/memories", handle_api_memories_get)
        app.router.add_post("/api/memories", handle_api_memories_post)
        app.router.add_delete("/api/memories/{id}", handle_api_memories_delete)
        app.router.add_post("/api/settings", handle_api_settings_post)
        app.router.add_post("/api/capabilities", handle_api_capabilities_post)
        app.router.add_get("/api/voiceprints", handle_api_voiceprints_get)
        app.router.add_delete("/api/voiceprints/{id}", handle_api_voiceprints_delete)
        app.router.add_get("/api/reminders", handle_api_reminders_get)
        app.router.add_post("/api/reminders", handle_api_reminders_post)
        app.router.add_delete("/api/reminders/{id}", handle_api_reminders_delete)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        actual_port = port
        if site._server and site._server.sockets:  # type: ignore[attr-defined]
            actual_port = int(site._server.sockets[0].getsockname()[1])  # type: ignore[attr-defined]
        logger.info("Unified HTTP MCP & Web Console listening on http://%s:%d/ (MCP at /mcp)", host, actual_port)
        return runner, actual_port

    async def serve_stdio(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        in_stream = stdin or sys.stdin
        out_stream = stdout or sys.stdout
        while True:
            line = await asyncio.to_thread(in_stream.readline)
            if not line:
                break
            response = await self.handle_line(line)
            if response is not None:
                out_stream.write(response + "\n")
                out_stream.flush()

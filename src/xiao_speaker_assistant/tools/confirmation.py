"""In-memory voice confirmation store for sensitive actions like sending emails."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Callable

from xiao_speaker_assistant.tools.outlook import PendingEmailDraft

_STRIP_CHARS = " \t\r\n，。！？、；：,.!?;:'\"`~～"

_CANCEL_EXACT = frozenset(
    {
        "取消",
        "取消发送",
        "别发",
        "别发了",
        "不要发",
        "不要发送",
        "不用发",
        "不用了",
        "不发",
        "不发了",
        "不发送",
        "不确认",
        "算了",
        "作废",
        "撤回",
        "停止",
        "放弃",
        "不需要",
        "不用",
        "不",
        "no",
        "nope",
        "cancel",
        "abort",
        "stop",
        "never mind",
        "nevermind",
        "don't send",
        "do not send",
    }
)

_CANCEL_PHRASES = (
    "取消发送",
    "取消",
    "别发",
    "不要发",
    "不用发",
    "不发了",
    "不发送",
    "不确认",
    "算了",
    "作废",
    "don't send",
    "do not send",
    "never mind",
    "nevermind",
    "cancel",
)

_CONFIRM_EXACT = frozenset(
    {
        "确认",
        "确认发送",
        "确定",
        "确定发送",
        "发送",
        "发吧",
        "发出去",
        "好",
        "好的",
        "好呀",
        "好啊",
        "可以",
        "行",
        "嗯",
        "是的",
        "是",
        "对",
        "没错",
        "没问题",
        "同意",
        "准许",
        "yes",
        "yeah",
        "yep",
        "confirm",
        "confirmed",
        "send",
        "send it",
        "ok",
        "okay",
        "sure",
        "go ahead",
        "do it",
    }
)

_CONFIRM_PHRASES = (
    "确认发送",
    "确定发送",
    "send it",
    "go ahead",
)


def _normalize(text: str) -> str:
    return text.strip(_STRIP_CHARS).lower()


def is_cancel_utterance(text: str) -> bool:
    """Return True if the spoken utterance clearly cancels a pending action."""
    norm = _normalize(text)
    if not norm:
        return False
    if norm in _CANCEL_EXACT:
        return True
    return any(phrase in norm for phrase in _CANCEL_PHRASES)


def is_confirm_utterance(text: str) -> bool:
    """Return True if the spoken utterance clearly confirms a pending action."""
    if is_cancel_utterance(text):
        return False
    norm = _normalize(text)
    if not norm:
        return False
    if norm in _CONFIRM_EXACT:
        return True
    if any(phrase in norm for phrase in _CONFIRM_PHRASES):
        return True
    parts = [p.strip(_STRIP_CHARS) for p in re.split(r"[,，、\s]+", norm) if p.strip(_STRIP_CHARS)]
    return bool(parts) and len(parts) <= 3 and all(p in _CONFIRM_EXACT for p in parts)


@dataclass
class _PendingEntry:
    draft: PendingEmailDraft
    expires_at: float


class ConfirmationStore:
    """TTL-backed single-slot store for a pending email draft awaiting voice confirmation."""

    def __init__(
        self,
        ttl_sec: float = 120.0,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._ttl_sec = ttl_sec
        self._clock = clock or time.monotonic
        self._pending: _PendingEntry | None = None

    def set_pending(
        self,
        draft: PendingEmailDraft,
        ttl_sec: float | None = None,
    ) -> PendingEmailDraft:
        ttl = self._ttl_sec if ttl_sec is None else ttl_sec
        stored = replace(draft, confirmed=False)
        self._pending = _PendingEntry(
            draft=stored,
            expires_at=self._clock() + ttl,
        )
        return stored

    def peek(self) -> PendingEmailDraft | None:
        if self._pending is None:
            return None
        if self._clock() >= self._pending.expires_at:
            self._pending = None
            return None
        return self._pending.draft

    def confirm_and_pop(self) -> PendingEmailDraft | None:
        draft = self.peek()
        if draft is None:
            return None
        self._pending = None
        return replace(draft, confirmed=True)

    def clear(self) -> None:
        self._pending = None

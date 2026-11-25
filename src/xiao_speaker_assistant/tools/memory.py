"""Persistent file-backed long-term memory store (0600, rejects secrets/credentials)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

_SECRET_PATTERN = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|keychain|credential|bearer|密码|口令|密钥|钥匙串)",
    re.IGNORECASE,
)


class MemorySecurityError(ValueError):
    """Raised when attempting to persist sensitive credentials in long-term memory."""


@dataclass
class MemoryEntry:
    id: str
    key: str
    value: str
    category: str = "fact"
    updated_at: float = 0.0


class MemoryStore:
    """Thread-safe JSON long-term memory store with 0600 permissions."""

    def __init__(
        self,
        path: str | Path = "~/.local/share/xiao-speaker-assistant/memory.json",
        *,
        max_entries: int = 100,
    ) -> None:
        self.path = Path(path).expanduser()
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._entries: list[MemoryEntry] = []
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
            loaded: list[MemoryEntry] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                k = str(item.get("key") or "").strip()
                v = str(item.get("value") or "").strip()
                if not k or not v:
                    continue
                loaded.append(
                    MemoryEntry(
                        id=str(item.get("id") or uuid.uuid4().hex[:8]),
                        key=k,
                        value=v,
                        category=str(item.get("category") or "fact"),
                        updated_at=float(item.get("updated_at") or 0.0),
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

    def save(self, key: str, value: str, *, category: str = "fact") -> MemoryEntry:
        clean_key = key.strip()
        clean_val = value.strip()
        clean_cat = category.strip() or "fact"
        if not clean_key or not clean_val:
            raise ValueError("Both 'key' and 'value' must be non-empty.")
        if _SECRET_PATTERN.search(clean_key) or _SECRET_PATTERN.search(clean_val):
            raise MemorySecurityError(
                "Refusing to store passwords, tokens, keys, or Keychain secrets in long-term memory."
            )

        with self._lock:
            norm_key = clean_key.lower()
            now = time.time()
            for idx, entry in enumerate(self._entries):
                if entry.key.lower() == norm_key:
                    updated = MemoryEntry(
                        id=entry.id,
                        key=clean_key,
                        value=clean_val,
                        category=clean_cat,
                        updated_at=now,
                    )
                    self._entries[idx] = updated
                    self._persist()
                    return updated

            new_entry = MemoryEntry(
                id=uuid.uuid4().hex[:8],
                key=clean_key,
                value=clean_val,
                category=clean_cat,
                updated_at=now,
            )
            self._entries.append(new_entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]
            self._persist()
            return new_entry

    def search(self, query: str = "", *, limit: int = 10) -> list[MemoryEntry]:
        q = query.strip().lower()
        with self._lock:
            items = list(reversed(self._entries))
            if not q:
                return items[:limit]
            tokens = [t for t in re.split(r"\s+", q) if t]
            matched = [
                e
                for e in items
                if any(
                    tok in e.key.lower() or tok in e.value.lower() or tok in e.category.lower()
                    for tok in tokens
                )
            ]
            return matched[:limit]

    def forget(self, key_or_id: str) -> bool:
        target = key_or_id.strip().lower()
        if not target:
            return False
        with self._lock:
            before = len(self._entries)
            self._entries = [
                e
                for e in self._entries
                if e.id.lower() != target and e.key.lower() != target
            ]
            removed = len(self._entries) < before
            if removed:
                self._persist()
            return removed

    def format_for_prompt(self, limit: int = 12) -> str:
        entries = self.search("", limit=limit)
        if not entries:
            return ""
        lines = ["用户长期记忆（已确认的个人偏好与常用事实，可直接参考）："]
        for e in entries:
            lines.append(f"- {e.key}: {e.value}")
        return "\n".join(lines)

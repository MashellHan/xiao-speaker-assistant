"""Async config.yaml hot-reload watcher."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

from xiao_speaker_assistant.config import AppConfig, load_config

logger = logging.getLogger(__name__)

ReloadCallback = Callable[[AppConfig], Awaitable[None] | None]


class ConfigWatcher:
    """Polls `config.yaml` mtime and invokes `on_reload(new_config)` without restart."""

    def __init__(
        self,
        config_path: str | Path | None,
        on_reload: ReloadCallback,
        *,
        poll_interval: float = 1.0,
    ) -> None:
        self._path = Path(config_path) if config_path else Path("config.yaml")
        self._on_reload = on_reload
        self._poll_interval = poll_interval
        self._last_mtime: float | None = self._read_mtime()
        self._task: asyncio.Task[None] | None = None

    def _read_mtime(self) -> float | None:
        try:
            return self._path.stat().st_mtime if self._path.exists() else None
        except OSError:
            return None

    async def check_once(self) -> bool:
        current = self._read_mtime()
        if current == self._last_mtime:
            return False
        try:
            new_cfg = load_config(self._path)
        except Exception as exc:
            logger.warning("Ignoring invalid config reload from %s: %s", self._path, exc)
            self._last_mtime = current
            return False
        self._last_mtime = current
        try:
            res = self._on_reload(new_cfg)
            if asyncio.iscoroutine(res):
                await res
            logger.info("Hot-reloaded configuration from %s", self._path)
            return True
        except Exception:
            logger.exception("Error applying hot-reloaded configuration from %s", self._path)
            return False

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
            await self.check_once()

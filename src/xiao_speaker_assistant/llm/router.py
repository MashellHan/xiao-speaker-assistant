"""LLM router supporting local `opencode` CLI, `codex` CLI, and streaming sentence TTS."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from xiao_speaker_assistant.config import LLMConfig, read_env_secret
from xiao_speaker_assistant.llm.opencode_server import (
    OpenCodeServerManager,
    OpenCodeV2Client,
    SessionNotFoundError,
)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]

SENTENCE_END_RE = re.compile(r"([。！？!?；;\n]+)")


def format_for_speech(text: str, max_chars: int = 220) -> str:
    """Strip markdown/links so speaker TTS reads naturally and concisely."""
    cleaned = re.sub(r"```[\s\S]*?```", " ", text)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"[#*`_~>]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 1)].rstrip("，。；,; ") + "。"


class SentenceStreamBuffer:
    """Buffers incremental LLM text and emits complete spoken sentences early."""

    def __init__(self, *, min_chars: int = 18, max_total_chars: int = 260) -> None:
        self._buf = ""
        self._min_chars = min_chars
        self._max_total = max_total_chars
        self._emitted_chars = 0

    def feed(self, chunk: str) -> list[str]:
        if not chunk or self._emitted_chars >= self._max_total:
            return []
        self._buf += chunk
        out: list[str] = []
        search_from = 0

        while self._emitted_chars < self._max_total:
            m = SENTENCE_END_RE.search(self._buf, search_from)
            if not m:
                break
            end = m.end()
            candidate = format_for_speech(self._buf[:end], max_chars=140)
            if len(candidate) < self._min_chars:
                search_from = end
                continue
            self._buf = self._buf[end:]
            search_from = 0
            if candidate:
                remaining = self._max_total - self._emitted_chars
                if len(candidate) > remaining:
                    candidate = format_for_speech(candidate, max_chars=remaining)
                out.append(candidate)
                self._emitted_chars += len(candidate)
        return out

    def flush(self) -> list[str]:
        if self._emitted_chars >= self._max_total or not self._buf.strip():
            self._buf = ""
            return []
        remaining = self._max_total - self._emitted_chars
        tail = format_for_speech(self._buf, max_chars=remaining)
        self._buf = ""
        if not tail:
            return []
        self._emitted_chars += len(tail)
        return [tail]


def parse_opencode_jsonl(stdout: str) -> str:
    """Extract assistant text parts from `opencode run --format json` output."""
    chunks: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "text":
            part = event.get("part") or {}
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n".join(chunks).strip()


def resolve_binary(name_or_path: str) -> str:
    found = shutil.which(name_or_path)
    if found:
        return found
    local_bin = Path("~/.local/bin").expanduser() / name_or_path
    if local_bin.is_file():
        return str(local_bin)
    raise RuntimeError(f"CLI binary not found: {name_or_path}")


class LLMRouter:
    def __init__(
        self,
        config: LLMConfig,
        client: httpx.AsyncClient | None = None,
        *,
        mcp_url: str = "http://127.0.0.1:4397/mcp",
        mcp_auth_token: str | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._mcp_url = mcp_url
        self._mcp_auth_token = mcp_auth_token
        self._opencode_mgr = OpenCodeServerManager(config.providers.opencode)
        self._opencode_v2: OpenCodeV2Client | None = None
        self.opencode_session_id: str | None = None
        self.history: list[tuple[str, str]] = []

    async def is_opencode_server_healthy(self) -> bool:
        if self._client is not None:
            return True
        return await self._opencode_mgr.is_healthy()

    async def start(
        self,
        *,
        mcp_url: str | None = None,
        mcp_auth_token: str | None = None,
    ) -> None:
        if mcp_url:
            self._mcp_url = mcp_url
        if mcp_auth_token is not None:
            self._mcp_auth_token = mcp_auth_token
        cfg = self._config.providers.opencode
        if self._config.default_provider != "opencode" or not cfg.server_url.strip():
            return
        if self._client is not None:
            self._opencode_v2 = OpenCodeV2Client(
                cfg.server_url, client=self._client, timeout_sec=cfg.timeout_sec
            )
            return
        base_url, password = await self._opencode_mgr.ensure_running(
            mcp_url=self._mcp_url,
            system_prompt=self._build_system_prompt(),
            mcp_auth_token=self._mcp_auth_token,
        )
        self._opencode_v2 = OpenCodeV2Client(
            base_url, password=password, timeout_sec=cfg.timeout_sec
        )

    async def close(self) -> None:
        await self._opencode_mgr.stop()

    async def interrupt_active(self) -> None:
        if self._opencode_v2 is not None and self.opencode_session_id:
            await self._opencode_v2.interrupt(self.opencode_session_id)

    def reset_session(self) -> None:
        self.opencode_session_id = None
        self.history.clear()

    def _format_with_history(self, user_text: str) -> str:
        if not self.history:
            return user_text
        recent = self.history[-6:]
        lines = ["最近对话上下文："]
        for u, a in recent:
            lines.append(f"用户：{u}")
            lines.append(f"助手：{a}")
        lines.append(f"用户最新提问：{user_text}")
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        return (
            f"你是运行在用户 Mac 本机上的家庭语音助手 Nomi，回答将通过智能音箱朗读。"
            f"你可以直接调用 xsa_tools 中的本机工具来回答问题：包括查询 Mac 磁盘空间（mac_disk_usage）、"
            f"读取本机允许目录的文件（mac_list_recent_files / mac_read_text_file）、"
            f"读取 Outlook 邮件（outlook_list_emails）、准备发邮件草稿（outlook_prepare_email）、"
            f"搜索小红书（xiaohongshu_search / xiaohongshu_detail）、网页摘要与用 Edge 打开网页（browser_fetch_summary / browser_open_edge）、"
            f"以及调用本地 Codex 做代码或技术分析（codex_analyze）。"
            f"准备发送邮件时只能调用 outlook_prepare_email 生成待确认草稿，并请用户口头说“确认发送”或“取消”，切勿声称邮件已发送。"
            f"不要回答“我无法访问你的电脑”或“还在学习中”。"
            f"无论用户使用中文、英文还是中英混合提问，请一律使用自然流畅的简体中文回答（专有名词可保留英文）。"
            f"请直接给出核心结论，简短口语化，不要使用 Markdown、代码块或 URL，"
            f"尽量控制在 100 字以内（不超过 {self._config.spoken_max_chars} 字）。"
        )

    async def ask(
        self,
        user_text: str,
        *,
        provider: str | None = None,
        tools: list[ToolDefinition] | None = None,
        executor: ToolExecutor | None = None,
        max_rounds: int = 4,
    ) -> str:
        chosen = provider or self._config.default_provider
        system_prompt = self._build_system_prompt()
        if chosen == "opencode":
            if self._config.providers.opencode.server_url.strip():
                raw = await self._ask_opencode_v2(system_prompt, user_text)
            else:
                raw = await self._ask_opencode_cli(system_prompt, user_text)
        elif chosen == "codex":
            raw = await self._ask_codex_cli(system_prompt, self._format_with_history(user_text))
        elif chosen == "codex_api":
            raw = await self._ask_codex_api(
                system_prompt, self._format_with_history(user_text), tools=tools or [], executor=executor, max_rounds=max_rounds
            )
        elif chosen == "gemini_api":
            raw = await self._ask_gemini_api(
                system_prompt, self._format_with_history(user_text), tools=tools or [], executor=executor, max_rounds=max_rounds
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {chosen}")
        formatted = format_for_speech(raw, max_chars=self._config.spoken_max_chars)
        self.history.append((user_text, formatted))
        if len(self.history) > 8:
            self.history = self.history[-8:]
        return formatted

    async def stream_ask(
        self,
        user_text: str,
        *,
        provider: str | None = None,
        tools: list[ToolDefinition] | None = None,
        executor: ToolExecutor | None = None,
        max_rounds: int = 4,
    ) -> AsyncIterator[str]:
        """Yield spoken-formatted sentences incrementally as the LLM responds."""
        chosen = provider or self._config.default_provider
        system_prompt = self._build_system_prompt()
        splitter = SentenceStreamBuffer(max_total_chars=self._config.spoken_max_chars)

        if chosen == "opencode":
            collected: list[str] = []
            stream_it = (
                self._stream_opencode_v2(system_prompt, user_text)
                if self._config.providers.opencode.server_url.strip()
                else self._stream_opencode_cli(system_prompt, user_text)
            )
            async for piece in stream_it:
                for sent in splitter.feed(piece):
                    collected.append(sent)
                    yield sent
            for sent in splitter.flush():
                collected.append(sent)
                yield sent
            if collected:
                self.history.append((user_text, "".join(collected)))
                if len(self.history) > 8:
                    self.history = self.history[-8:]
            return

        if chosen == "codex_api" and not tools:
            async for piece in self._stream_codex_api(system_prompt, user_text):
                for sent in splitter.feed(piece):
                    yield sent
            for sent in splitter.flush():
                yield sent
            return

        if chosen == "gemini_api" and not tools:
            async for piece in self._stream_gemini_api(system_prompt, user_text):
                for sent in splitter.feed(piece):
                    yield sent
            for sent in splitter.flush():
                yield sent
            return

        reply = await self.ask(
            user_text,
            provider=chosen,
            tools=tools,
            executor=executor,
            max_rounds=max_rounds,
        )
        for sent in splitter.feed(reply):
            yield sent
        for sent in splitter.flush():
            yield sent

    async def _ensure_v2_client(self, system_prompt: str) -> OpenCodeV2Client:
        if self._client is None and self._opencode_mgr.has_dead_process():
            self._opencode_v2 = None
            self.opencode_session_id = None
        if self._opencode_v2 is not None:
            return self._opencode_v2
        cfg = self._config.providers.opencode
        if self._client is not None:
            self._opencode_v2 = OpenCodeV2Client(
                cfg.server_url, client=self._client, timeout_sec=cfg.timeout_sec
            )
            return self._opencode_v2
        base_url, password = await self._opencode_mgr.ensure_running(
            mcp_url=self._mcp_url,
            system_prompt=system_prompt,
            mcp_auth_token=self._mcp_auth_token,
        )
        self._opencode_v2 = OpenCodeV2Client(
            base_url, password=password, timeout_sec=cfg.timeout_sec
        )
        return self._opencode_v2

    async def _stream_opencode_v2(self, system_prompt: str, user_text: str) -> AsyncIterator[str]:
        cfg = self._config.providers.opencode

        for attempt in range(2):
            v2 = await self._ensure_v2_client(system_prompt)
            if not self.opencode_session_id:
                self.opencode_session_id = await v2.create_session(
                    model=cfg.model,
                    directory=str(self._opencode_mgr.agent_home),
                    agent="nomi",
                    title="xsa-voice",
                )
            sid = self.opencode_session_id
            got_text = False
            try:
                async for delta in v2.stream_prompt(sid, user_text):
                    if delta:
                        got_text = True
                        yield delta
                if not got_text:
                    yield "抱歉，这次没有拿到回复。"
                return
            except (SessionNotFoundError, httpx.TransportError):
                self.opencode_session_id = None
                if self._client is None:
                    self._opencode_v2 = None
                if attempt == 0:
                    continue
                raise
            except (asyncio.CancelledError, GeneratorExit):
                asyncio.create_task(v2.interrupt(sid))
                raise

    async def _ask_opencode_v2(self, system_prompt: str, user_text: str) -> str:
        chunks: list[str] = []
        async for piece in self._stream_opencode_v2(system_prompt, user_text):
            chunks.append(piece)
        return "".join(chunks).strip() or "抱歉，这次没有拿到回复。"

    async def _stream_opencode_cli(self, system_prompt: str, user_text: str) -> AsyncIterator[str]:
        cfg = self._config.providers.opencode
        binary = resolve_binary(cfg.binary)
        if self.opencode_session_id:
            prompt = user_text
            cmd = [
                binary,
                "run",
                "--model",
                cfg.model,
                "--format",
                "json",
                "--session",
                self.opencode_session_id,
            ]
        else:
            prompt = f"{system_prompt}\n\n用户语音问题：{user_text}"
            cmd = [
                binary,
                "run",
                "--model",
                cfg.model,
                "--format",
                "json",
                "--archived",
                "--title",
                "xsa-voice",
            ]
        if cfg.auto_approve:
            cmd.append("--auto")
        cmd.append(prompt)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None

        got_text = False
        try:
            async def read_lines() -> AsyncIterator[str]:
                nonlocal got_text
                while True:
                    raw_line = await proc.stdout.readline()  # type: ignore[union-attr]
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sid = event.get("sessionID")
                    if isinstance(sid, str) and sid.startswith("ses_"):
                        self.opencode_session_id = sid
                    if event.get("type") == "text":
                        part = event.get("part") or {}
                        text = part.get("text")
                        if isinstance(text, str) and text.strip():
                            got_text = True
                            yield text

            async for chunk in read_lines():
                yield chunk
            await asyncio.wait_for(proc.wait(), timeout=cfg.timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("opencode CLI timed out")

        if proc.returncode != 0:
            err = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"opencode CLI failed ({proc.returncode}): {err}")
        if not got_text:
            yield "抱歉，这次没有拿到回复。"

    async def _ask_opencode_cli(self, system_prompt: str, user_text: str) -> str:
        chunks: list[str] = []
        async for piece in self._stream_opencode_cli(system_prompt, user_text):
            chunks.append(piece)
        return "\n".join(chunks).strip() or "抱歉，这次没有拿到回复。"

    async def _ask_codex_cli(self, system_prompt: str, user_text: str) -> str:
        cfg = self._config.providers.codex
        binary = resolve_binary(cfg.binary)
        prompt = f"{system_prompt}\n\n用户语音问题：{user_text}"
        workdir = str(Path(cfg.workdir).expanduser().resolve())

        fd, out_file = tempfile.mkstemp(prefix="xsa-codex-", suffix=".txt")
        os.close(fd)
        try:
            cmd = [
                binary,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "-C",
                workdir,
                "-o",
                out_file,
            ]
            if cfg.model:
                cmd.extend(["-m", cfg.model])
            cmd.append(prompt)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=cfg.timeout_sec)
            except asyncio.TimeoutError:
                proc.kill()
                raise RuntimeError("codex CLI timed out")
            if proc.returncode != 0:
                err = stderr_b.decode("utf-8", errors="replace").strip()
                out = stdout_b.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"codex CLI failed ({proc.returncode}): {err or out[-300:]}")
            text = Path(out_file).read_text(encoding="utf-8", errors="replace").strip()
            return text or "抱歉，这次没有拿到回复。"
        finally:
            try:
                os.unlink(out_file)
            except OSError:
                pass

    async def _stream_codex_api(self, system_prompt: str, user_text: str) -> AsyncIterator[str]:
        cfg = self._config.providers.codex_api
        api_key = read_env_secret(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env {cfg.api_key_env}")

        url = cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": cfg.model,
            "stream": True,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        }
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=45.0)
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    data = json.loads(body)
                    delta = data["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        finally:
            if owns_client:
                await client.aclose()

    async def _stream_gemini_api(self, system_prompt: str, user_text: str) -> AsyncIterator[str]:
        cfg = self._config.providers.gemini_api
        api_key = read_env_secret(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env {cfg.api_key_env}")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.model}:streamGenerateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        }
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=45.0)
        try:
            async with client.stream("POST", url, params={"key": api_key, "alt": "sse"}, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:].strip())
                    parts = data["candidates"][0]["content"]["parts"]
                    for part in parts:
                        txt = part.get("text")
                        if txt:
                            yield txt
        finally:
            if owns_client:
                await client.aclose()

    async def _ask_codex_api(
        self,
        system_prompt: str,
        user_text: str,
        *,
        tools: list[ToolDefinition],
        executor: ToolExecutor | None,
        max_rounds: int,
    ) -> str:
        cfg = self._config.providers.codex_api
        api_key = read_env_secret(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env {cfg.api_key_env}")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=45.0)
        try:
            for _ in range(max_rounds):
                payload: dict[str, Any] = {"model": cfg.model, "messages": messages}
                if openai_tools:
                    payload["tools"] = openai_tools
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls or executor is None:
                    return msg.get("content") or ""
                messages.append(msg)
                for call in tool_calls:
                    fn = call["function"]
                    args = json.loads(fn.get("arguments") or "{}")
                    result = await executor(fn["name"], args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result,
                        }
                    )
            return "处理超时，请稍后再试。"
        finally:
            if owns_client:
                await client.aclose()

    async def _ask_gemini_api(
        self,
        system_prompt: str,
        user_text: str,
        *,
        tools: list[ToolDefinition],
        executor: ToolExecutor | None,
        max_rounds: int,
    ) -> str:
        cfg = self._config.providers.gemini_api
        api_key = read_env_secret(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env {cfg.api_key_env}")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.model}:generateContent"
        contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": user_text}]}]
        gemini_tools = (
            [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                        for t in tools
                    ]
                }
            ]
            if tools
            else []
        )

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=45.0)
        try:
            for _ in range(max_rounds):
                payload: dict[str, Any] = {
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": contents,
                }
                if gemini_tools:
                    payload["tools"] = gemini_tools
                resp = await client.post(url, params={"key": api_key}, json=payload)
                resp.raise_for_status()
                parts = resp.json()["candidates"][0]["content"]["parts"]
                fn_calls = [p["functionCall"] for p in parts if "functionCall" in p]
                if not fn_calls or executor is None:
                    return " ".join(p.get("text", "") for p in parts).strip()
                contents.append({"role": "model", "parts": parts})
                response_parts = []
                for call in fn_calls:
                    result = await executor(call["name"], call.get("args") or {})
                    response_parts.append(
                        {
                            "functionResponse": {
                                "name": call["name"],
                                "response": {"result": result},
                            }
                        }
                    )
                contents.append({"role": "user", "parts": response_parts})
            return "处理超时，请稍后再试。"
        finally:
            if owns_client:
                await client.aclose()

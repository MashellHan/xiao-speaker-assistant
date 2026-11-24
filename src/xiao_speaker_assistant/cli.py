"""CLI entrypoint for Xiao Speaker Assistant."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import secrets
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from xiao_speaker_assistant.config import AppConfig, load_config, read_env_secret
from xiao_speaker_assistant.llm import LLMRouter, ToolDefinition
from xiao_speaker_assistant.llm.opencode_server import load_or_create_mcp_token
from xiao_speaker_assistant.llm.router import resolve_binary
from xiao_speaker_assistant.tools import (
    ConfirmationStore,
    HttpMcpClient,
    McpServer,
    OutlookClient,
    PendingEmailDraft,
    is_cancel_utterance,
    is_confirm_utterance,
)
from xiao_speaker_assistant.voice import SpeakerGateway, run_with_relay
from xiao_speaker_assistant.voice.models_setup import (
    download_models,
    find_paraformer_dir,
    find_sense_voice_dir,
)


def build_tool_registry(
    config: AppConfig,
    confirmation_store: ConfirmationStore | None = None,
) -> tuple[list[ToolDefinition], dict[str, Any]]:
    tools: list[ToolDefinition] = []
    clients: dict[str, Any] = {}
    if confirmation_store is not None:
        clients["confirmations"] = confirmation_store

    if config.tools.outlook.enabled and read_env_secret(config.tools.outlook.client_id_env):
        outlook = OutlookClient(config.tools.outlook)
        clients["outlook"] = outlook
        tools.extend(
            [
                ToolDefinition(
                    name="outlook_list_emails",
                    description="读取最近的 Outlook 邮件标题、发件人和摘要",
                    parameters={
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "default": 5},
                            "unread_only": {"type": "boolean", "default": False},
                        },
                    },
                ),
                ToolDefinition(
                    name="outlook_prepare_email",
                    description="准备 Outlook 邮件草稿并请求用户语音确认发送",
                    parameters={
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"],
                    },
                ),
            ]
        )

    if config.tools.xiaohongshu.enabled:
        clients["xiaohongshu"] = HttpMcpClient(config.tools.xiaohongshu.mcp_url)
        tools.append(
            ToolDefinition(
                name="xiaohongshu_search",
                description="在小红书搜索笔记并返回标题与摘要",
                parameters={
                    "type": "object",
                    "properties": {"keyword": {"type": "string"}},
                    "required": ["keyword"],
                },
            )
        )

    return tools, clients


async def execute_tool(clients: dict[str, Any], name: str, args: dict[str, Any]) -> str:
    if name == "outlook_list_emails":
        outlook: OutlookClient = clients["outlook"]
        items = await outlook.list_recent_emails(
            limit=int(args.get("limit", 5)),
            unread_only=bool(args.get("unread_only", False)),
        )
        return json.dumps(items, ensure_ascii=False)
    if name in {"outlook_prepare_email", "outlook_send_email"}:
        outlook = clients["outlook"]
        store: ConfirmationStore | None = clients.get("confirmations")
        draft = PendingEmailDraft(
            to=args["to"],
            subject=args["subject"],
            body=args["body"],
            confirmed=False,
        )
        if store is not None and outlook._config.require_voice_confirmation_for_send:
            store.set_pending(draft)
        elif not outlook._config.require_voice_confirmation_for_send:
            draft = PendingEmailDraft(
                to=args["to"],
                subject=args["subject"],
                body=args["body"],
                confirmed=True,
            )
        return await outlook.send_email(draft)
    if name == "xiaohongshu_search":
        mcp: HttpMcpClient = clients["xiaohongshu"]
        res = await mcp.call_tool("search_feeds", {"keyword": args["keyword"]})
        return json.dumps(res, ensure_ascii=False)
    raise ValueError(f"Unknown tool: {name}")


def _check_http_reachable(url: str) -> tuple[bool, str]:
    try:
        resp = httpx.get(url, timeout=1.5)
        return True, f"HTTP {resp.status_code}"
    except Exception as exc:
        return False, str(exc)


def cmd_doctor(config: AppConfig) -> int:
    print("Xiao Speaker Assistant configuration check:")
    print(f"  Speaker model:    {config.speaker.model} (ws://{config.speaker.ws_host}:{config.speaker.ws_port})")
    print(f"  TTS provider:     {config.speaker.tts_provider} ({config.speaker.tts_sample_rate} Hz)")
    print(
        f"  Speech pipeline:  {config.speech.asr_provider} / {config.speech.asr_model} "
        f"(pcm={config.speech.capture_pcm}, ch={config.speech.capture_channels}, aec={config.speech.aec_enabled})"
    )
    print(f"  Default provider: {config.llm.default_provider}")

    for label, binary in (
        ("opencode CLI (raven/gemini)", config.llm.providers.opencode.binary),
        ("codex CLI", config.llm.providers.codex.binary),
        ("ffmpeg (edge-tts PCM)", "ffmpeg"),
    ):
        try:
            path = resolve_binary(binary)
            extra = ""
            if binary == config.llm.providers.opencode.binary:
                ver = subprocess.run(
                    [path, "--version"], capture_output=True, text=True, timeout=5, check=False
                )
                if ver.returncode == 0 and ver.stdout.strip():
                    extra = f" ({ver.stdout.strip()})"
            print(f"  [OK] {label}: {path}{extra}")
        except RuntimeError:
            print(f"  [MISSING] {label}: {binary}")

    models_dir = Path(config.speech.models_dir).expanduser()
    required_files = ["silero_vad.onnx", "tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"]
    missing_models = [f for f in required_files if not (models_dir / f).is_file()]
    if config.speech.asr_model == "paraformer":
        try:
            find_paraformer_dir(models_dir)
        except FileNotFoundError:
            missing_models.append("paraformer/model.int8.onnx")
    elif config.speech.asr_model == "sense_voice":
        try:
            find_sense_voice_dir(models_dir)
        except FileNotFoundError:
            missing_models.append("sense-voice/model.int8.onnx")
    if missing_models:
        print(f"  [MISSING] Speech models in {models_dir}: {', '.join(missing_models)}")
    else:
        print(f"  [OK] Speech models ready in {models_dir}")

    print(f"  OpenCode model:   {config.llm.providers.opencode.model}")
    print(f"  OpenCode V2 URL:  {config.llm.providers.opencode.server_url}")
    print(f"  OpenCode home:    {config.llm.providers.opencode.agent_home}")
    print(f"  Unified MCP HTTP: http://127.0.0.1:{config.tools.mcp_http_port}/mcp")

    client_id_present = bool(read_env_secret(config.tools.outlook.client_id_env))
    cache_path = Path(config.tools.outlook.token_cache_file).expanduser()
    cache_status = "missing"
    if cache_path.exists():
        mode = stat.S_IMODE(cache_path.stat().st_mode)
        cache_status = f"present (mode={oct(mode)})"
    print(
        f"  Outlook tool:     {'enabled' if config.tools.outlook.enabled else 'disabled'} "
        f"(client_id={'set' if client_id_present else 'unset'}, cache={cache_path} [{cache_status}])"
    )

    if config.tools.xiaohongshu.enabled:
        ok, detail = _check_http_reachable(config.tools.xiaohongshu.mcp_url)
        tag = "OK" if ok else "UNREACHABLE"
        print(f"  Xiaohongshu MCP:  [{tag}] {config.tools.xiaohongshu.mcp_url} ({detail})")
    else:
        print(f"  Xiaohongshu MCP:  disabled ({config.tools.xiaohongshu.mcp_url})")

    edge_exists = Path("/Applications/Microsoft Edge.app").exists() or shutil.which("msedge") is not None
    print(
        f"  Browser / Edge:   mcp={'enabled' if config.tools.browser.enabled else 'disabled'}, "
        f"edge_app={'found' if edge_exists else 'not found'}"
    )
    return 0


async def _serve_async(config: AppConfig, args: argparse.Namespace) -> None:
    confirmations = ConfirmationStore()
    agent_home = Path(config.llm.providers.opencode.agent_home).expanduser().resolve()
    mcp_token = load_or_create_mcp_token(agent_home)
    mcp_server = McpServer(
        config,
        confirmation_store=confirmations,
        include_local_tools=True,
    )
    mcp_runner, mcp_port = await mcp_server.start_http_server(
        host="127.0.0.1",
        port=config.tools.mcp_http_port,
        auth_token=mcp_token,
    )
    mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"

    router = LLMRouter(config.llm, mcp_url=mcp_url, mcp_auth_token=mcp_token)
    effective_provider = args.provider or config.llm.default_provider
    if effective_provider == "opencode":
        await router.start(mcp_url=mcp_url, mcp_auth_token=mcp_token)

    tools, clients = build_tool_registry(config, confirmation_store=confirmations)
    outlook_client = clients.get("outlook") or OutlookClient(config.tools.outlook)

    async def handle_query(query: str) -> str:
        if confirmations.peek() is not None:
            if is_confirm_utterance(query):
                draft = confirmations.confirm_and_pop()
                if draft is not None:
                    await outlook_client.send_email(draft)
                    return f"好的，已发送邮件给 {draft.to}。"
            elif is_cancel_utterance(query):
                confirmations.clear()
                return "好的，已取消发送邮件。"

        return await router.ask(
            query,
            provider=args.provider,
            tools=tools,
            executor=lambda name, t_args: execute_tool(clients, name, t_args),
        )

    async def stream_query(query: str):
        if confirmations.peek() is not None:
            if is_confirm_utterance(query):
                draft = confirmations.confirm_and_pop()
                if draft is not None:
                    await outlook_client.send_email(draft)
                    yield f"好的，已发送邮件给 {draft.to}。"
                    return
            elif is_cancel_utterance(query):
                confirmations.clear()
                yield "好的，已取消发送邮件。"
                return

        async for sentence in router.stream_ask(
            query,
            provider=args.provider,
            tools=tools,
            executor=lambda name, t_args: execute_tool(clients, name, t_args),
        ):
            yield sentence

    def reset_all() -> None:
        router.reset_session()
        confirmations.clear()

    try:
        if args.relay:
            await run_with_relay(
                config,
                on_query=handle_query,
                relay_url=args.relay,
                on_stream_query=stream_query,
                on_session_reset=reset_all,
                token_env=args.relay_token_env,
                also_serve_local=True,
            )
        else:
            gateway = SpeakerGateway(
                config,
                on_query=handle_query,
                on_stream_query=stream_query,
                on_session_reset=reset_all,
            )
            await gateway.serve_forever()
    finally:
        await router.close()
        await mcp_runner.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xsa", description="Xiao Speaker Assistant CLI")
    parser.add_argument("--config", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check config and local CLI backends")
    sub.add_parser("outlook-login", help="Sign in to Outlook using device code + private file cache")
    sub.add_parser("download-models", help="Download VAD + KWS + SenseVoice models for `Hi Nomi` wakeup")
    sub.add_parser(
        "mcp-server",
        help="Run stdio JSON-RPC 2.0 MCP server for Outlook, Codex, Xiaohongshu, Browser, and Mac tools",
    )

    ask_p = sub.add_parser("ask", help="Test a single question via opencode or codex CLI")
    ask_p.add_argument("prompt", help="Question text")
    ask_p.add_argument("--provider", choices=["opencode", "codex", "codex_api", "gemini_api"])

    serve_p = sub.add_parser("serve", help="Run the speaker WebSocket gateway")
    serve_p.add_argument("--host", help="Override WebSocket listen host")
    serve_p.add_argument("--port", type=int, help="Override WebSocket listen port")
    serve_p.add_argument(
        "--asr",
        choices=["sense_voice", "xiaoai_asr"],
        help="Override speech recognition backend (`sense_voice` enables `Hi Nomi` local wakeup)",
    )
    serve_p.add_argument(
        "--asr-model",
        choices=["sense_voice", "paraformer", "doubao_flash", "openai_whisper"],
        help="Select ASR engine when `--asr sense_voice` is active",
    )
    serve_p.add_argument(
        "--relay",
        help="Connect outbound to an Azure/remote WebSocket relay (e.g. wss://relay.example.com)",
    )
    serve_p.add_argument(
        "--relay-token-env",
        default="XSA_RELAY_TOKEN",
        help="Environment variable holding the relay Bearer token (default: XSA_RELAY_TOKEN)",
    )
    serve_p.add_argument(
        "--provider",
        choices=["opencode", "codex", "codex_api", "gemini_api"],
        help="Override default LLM provider",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)

    if args.command == "doctor":
        return cmd_doctor(config)
    if args.command == "outlook-login":
        OutlookClient(config.tools.outlook).login_device_flow()
        print("OK: Outlook token saved to private file cache.")
        return 0
    if args.command == "download-models":
        download_models(Path(config.speech.models_dir).expanduser(), config.speech.local_keywords)
        return 0
    if args.command == "mcp-server":
        asyncio.run(McpServer(config, include_local_tools=True).serve_stdio())
        return 0
    if args.command == "ask":
        async def _ask_once() -> str:
            effective = args.provider or config.llm.default_provider
            router = LLMRouter(config.llm)
            mcp_runner = None
            try:
                if (
                    effective == "opencode"
                    and config.llm.providers.opencode.server_url.strip()
                    and not await router.is_opencode_server_healthy()
                ):
                    mcp_token = secrets.token_urlsafe(24)
                    mcp_srv = McpServer(config, include_local_tools=True)
                    mcp_runner, mcp_port = await mcp_srv.start_http_server(
                        host="127.0.0.1",
                        port=0,
                        auth_token=mcp_token,
                    )
                    await router.start(
                        mcp_url=f"http://127.0.0.1:{mcp_port}/mcp",
                        mcp_auth_token=mcp_token,
                    )
                return await router.ask(args.prompt, provider=args.provider)
            finally:
                await router.close()
                if mcp_runner is not None:
                    await mcp_runner.cleanup()

        print(asyncio.run(_ask_once()))
        return 0
    if args.command == "serve":
        if args.host:
            config.speaker.ws_host = args.host
        if args.port:
            config.speaker.ws_port = args.port
        if args.asr:
            config.speech.asr_provider = args.asr
        if args.asr_model:
            config.speech.asr_model = args.asr_model
        asyncio.run(_serve_async(config, args))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

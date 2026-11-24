"""CLI entrypoint for Xiao Speaker Assistant."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from xiao_speaker_assistant.config import AppConfig, load_config, read_env_secret
from xiao_speaker_assistant.llm import LLMRouter, ToolDefinition
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
from xiao_speaker_assistant.voice.models_setup import download_models


def build_tool_registry(config: AppConfig) -> tuple[list[ToolDefinition], dict[str, Any]]:
    tools: list[ToolDefinition] = []
    clients: dict[str, Any] = {}

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
                    name="outlook_send_email",
                    description="通过 Outlook 发送邮件；未经过用户口头确认时必须先返回待确认摘要",
                    parameters={
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                            "confirmed": {"type": "boolean", "default": False},
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
    if name == "outlook_send_email":
        outlook = clients["outlook"]
        draft = PendingEmailDraft(
            to=args["to"],
            subject=args["subject"],
            body=args["body"],
            confirmed=bool(args.get("confirmed", False)),
        )
        return await outlook.send_email(draft)
    if name == "xiaohongshu_search":
        mcp: HttpMcpClient = clients["xiaohongshu"]
        res = await mcp.call_tool("search_feeds", {"keyword": args["keyword"]})
        return json.dumps(res, ensure_ascii=False)
    raise ValueError(f"Unknown tool: {name}")


def cmd_doctor(config: AppConfig) -> int:
    print("Xiao Speaker Assistant configuration check:")
    print(f"  Speaker model:    {config.speaker.model} (ws://{config.speaker.ws_host}:{config.speaker.ws_port})")
    print(f"  Default provider: {config.llm.default_provider}")

    for label, binary in (
        ("opencode CLI (raven/gemini)", config.llm.providers.opencode.binary),
        ("codex CLI", config.llm.providers.codex.binary),
    ):
        try:
            path = resolve_binary(binary)
            print(f"  [OK] {label}: {path}")
        except RuntimeError:
            print(f"  [MISSING] {label}: {binary}")

    print(f"  opencode model:   {config.llm.providers.opencode.model}")
    print(f"  Outlook cache:    {config.tools.outlook.token_cache_file}")
    print(f"  Xiaohongshu MCP:  {'enabled' if config.tools.xiaohongshu.enabled else 'disabled'}")
    print(f"  Browser MCP:      {'enabled' if config.tools.browser.enabled else 'disabled'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xsa", description="Xiao Speaker Assistant CLI")
    parser.add_argument("--config", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check config and local CLI backends")
    sub.add_parser("outlook-login", help="Sign in to Outlook using device code + private file cache")
    sub.add_parser("download-models", help="Download VAD + KWS + SenseVoice models for `Hi Nomi` wakeup")
    sub.add_parser(
        "mcp-server",
        help="Run stdio JSON-RPC 2.0 MCP server for Outlook, Xiaohongshu, and Browser",
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
        choices=["sense_voice", "paraformer"],
        help="Select offline ASR model when `--asr sense_voice` is active",
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
        asyncio.run(McpServer(config).serve_stdio())
        return 0
    if args.command == "ask":
        router = LLMRouter(config.llm)
        reply = asyncio.run(router.ask(args.prompt, provider=args.provider))
        print(reply)
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
        router = LLMRouter(config.llm)
        tools, clients = build_tool_registry(config)
        confirmations = ConfirmationStore()

        async def handle_query(query: str) -> str:
            if confirmations.peek() is not None:
                if is_confirm_utterance(query):
                    draft = confirmations.confirm_and_pop()
                    if draft is not None and "outlook" in clients:
                        await clients["outlook"].send_email(draft)
                        return f"好的，已发送邮件给 {draft.to}。"
                elif is_cancel_utterance(query):
                    confirmations.clear()
                    return "好的，已取消发送邮件。"

            chosen = args.provider
            if chosen is None and ("codex" in query.lower()):
                chosen = "codex"
            return await router.ask(
                query,
                provider=chosen,
                tools=tools,
                executor=lambda name, t_args: execute_tool(clients, name, t_args),
            )

        if args.relay:
            asyncio.run(
                run_with_relay(
                    config,
                    on_query=handle_query,
                    relay_url=args.relay,
                    token_env=args.relay_token_env,
                    also_serve_local=True,
                )
            )
        else:
            gateway = SpeakerGateway(config, on_query=handle_query)
            asyncio.run(gateway.serve_forever())
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

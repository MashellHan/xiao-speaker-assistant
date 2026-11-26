#!/usr/bin/env python3
"""CLI helper to deploy and manage `/data/xiaoai-agent/` on an OH2P speaker via SSH."""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from xiao_speaker_assistant.voice.speaker_deployer import (
        DEFAULT_KEY_PATH,
        DEFAULT_SSH_USER,
        DeployResult,
        SpeakerDeployer,
        build_boot_script,
        validate_server_url,
    )
except ModuleNotFoundError:
    _sd_path = SRC_DIR / "xiao_speaker_assistant" / "voice" / "speaker_deployer.py"
    _sd_spec = importlib.util.spec_from_file_location("speaker_deployer", _sd_path)
    assert _sd_spec is not None and _sd_spec.loader is not None
    _sd_mod = importlib.util.module_from_spec(_sd_spec)
    sys.modules[_sd_spec.name] = _sd_mod
    _sd_spec.loader.exec_module(_sd_mod)
    DEFAULT_KEY_PATH = _sd_mod.DEFAULT_KEY_PATH
    DEFAULT_SSH_USER = _sd_mod.DEFAULT_SSH_USER
    DeployResult = _sd_mod.DeployResult
    SpeakerDeployer = _sd_mod.SpeakerDeployer
    build_boot_script = _sd_mod.build_boot_script
    validate_server_url = _sd_mod.validate_server_url


def default_known_hosts() -> Path:
    return REPO_ROOT / ".local" / "known_hosts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy and manage /data/xiaoai-agent/ on OH2P over key-only SSH.",
    )
    parser.add_argument(
        "--host",
        help="Speaker IP or hostname (e.g. 192.168.1.50 or oh2p.local).",
    )
    parser.add_argument(
        "--user",
        default=DEFAULT_SSH_USER,
        help=f"SSH user (default: {DEFAULT_SSH_USER}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=22,
        help="SSH port (default: 22).",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=DEFAULT_KEY_PATH,
        help=f"SSH private key path (default: {DEFAULT_KEY_PATH}).",
    )
    parser.add_argument(
        "--known-hosts",
        type=Path,
        default=default_known_hosts(),
        help="Known hosts file path (default: .local/known_hosts).",
    )
    parser.add_argument(
        "--server-url",
        help="Write ws:// (LAN) or wss:// (Azure relay) URL to /data/xiaoai-agent/server.txt.",
    )
    parser.add_argument(
        "--install-boot",
        action="store_true",
        help="Install BusyBox-compatible /data/xiaoai-agent/boot.sh on the speaker.",
    )
    parser.add_argument(
        "--install-aec-alsa",
        action="store_true",
        help="Install /data/xiaoai-agent/asound_xsa.conf exposing 2ch (mic + ch3 loopback) pcm.xsa_aec.",
    )
    parser.add_argument(
        "--client-bin",
        type=Path,
        help="Upload local open-xiaoai client binary to /data/xiaoai-agent/client.",
    )
    parser.add_argument(
        "--tunnel-host",
        help="Configure encrypted Dropbear SSH tunnel to Azure VM host (e.g. relay.mxhan.com).",
    )
    parser.add_argument(
        "--tunnel-port",
        type=int,
        default=2222,
        help="SSH tunnel port on Azure VM (default: 2222).",
    )
    parser.add_argument(
        "--tunnel-user",
        default="xsa-tunnel",
        help="Restricted SSH tunnel user on Azure VM (default: xsa-tunnel).",
    )
    parser.add_argument(
        "--tunnel-key",
        type=Path,
        help="Upload local Dropbear-format private key to /data/xiaoai-agent/tunnel_id_dropbear.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check speaker processes (ps), /data free space, and /tmp/xiaoai-agent.log.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart /data/xiaoai-agent/boot.sh using `start-stop-daemon -S -b`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned SSH commands without connecting to the speaker.",
    )
    parser.add_argument(
        "--print-boot-script",
        action="store_true",
        help="Print the generated BusyBox boot.sh script and exit.",
    )
    return parser


def _print_result(label: str, result: DeployResult) -> None:
    if result.dry_run:
        print(f"[DRY-RUN] {label}:")
        print("  " + " ".join(shlex.quote(part) for part in result.command))
        return
    if result.stdout:
        sys.stdout.write(result.stdout)
        if not result.stdout.endswith("\n"):
            sys.stdout.write("\n")
    if result.stderr:
        sys.stderr.write(result.stderr)
        if not result.stderr.endswith("\n"):
            sys.stderr.write("\n")


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., Any] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.print_boot_script:
        sys.stdout.write(build_boot_script())
        return 0

    has_action = any(
        [
            args.client_bin is not None,
            args.install_boot,
            args.install_aec_alsa,
            args.tunnel_host is not None,
            args.tunnel_key is not None,
            args.server_url is not None,
            args.restart,
            args.status,
        ]
    )
    if not has_action:
        parser.error(
            "Specify at least one action: --server-url, --install-boot, --install-aec-alsa, --client-bin, --restart, --status, or --print-boot-script."
        )

    if not args.host:
        parser.error("--host is required for speaker operations.")

    if args.server_url is not None:
        try:
            validate_server_url(args.server_url)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    deployer_kwargs: dict[str, Any] = {
        "user": args.user,
        "key_path": args.key,
        "known_hosts": args.known_hosts,
        "port": args.port,
        "dry_run": args.dry_run,
    }
    if runner is not None:
        deployer_kwargs["runner"] = runner
    deployer = SpeakerDeployer(args.host, **deployer_kwargs)

    steps: list[tuple[str, Callable[[], DeployResult]]] = []
    if args.client_bin is not None:
        steps.append(("upload-client", lambda: deployer.upload_client(args.client_bin)))
    if args.install_boot:
        steps.append(("install-boot", deployer.install_boot_script))
    if args.install_aec_alsa:
        steps.append(("install-aec-alsa", deployer.install_aec_alsa))
    if args.tunnel_key is not None:
        steps.append(("upload-tunnel-key", lambda: deployer.upload_tunnel_key(args.tunnel_key)))
    if args.tunnel_host is not None:
        steps.append(
            (
                "install-tunnel-config",
                lambda: deployer.install_tunnel_config(
                    args.tunnel_host,
                    tunnel_port=args.tunnel_port,
                    tunnel_user=args.tunnel_user,
                ),
            )
        )
    if args.server_url is not None:
        steps.append(("update-server-url", lambda: deployer.update_server_url(args.server_url)))
    if args.restart:
        steps.append(("restart", deployer.restart))
    if args.status:
        steps.append(("status", deployer.status))

    for label, action in steps:
        try:
            result = action()
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR ({label}): {exc}", file=sys.stderr)
            return 1
        _print_result(label, result)
        if result.returncode != 0:
            print(f"ERROR ({label}): SSH exited with status {result.returncode}", file=sys.stderr)
            return result.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())

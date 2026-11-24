#!/usr/bin/env python3
"""Build, install, or uninstall the macOS LaunchAgent for Xiao Speaker Assistant."""

from __future__ import annotations

import argparse
import plistlib
import stat
import sys
from pathlib import Path
from typing import Any

LAUNCHAGENT_LABEL = "com.mashellhan.xiao-speaker-assistant"
DEFAULT_PLIST_PATH = Path(f"~/Library/LaunchAgents/{LAUNCHAGENT_LABEL}.plist")
DEFAULT_ENV_FILE = Path("~/.config/xiao-speaker-assistant/env")


def default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def validate_env_file(env_file: Path) -> Path:
    expanded = env_file.expanduser().resolve()
    if not expanded.is_file():
        raise FileNotFoundError(f"Env file not found: {expanded}")
    mode = stat.S_IMODE(expanded.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"Env file {expanded} must not be group/world-accessible (mode={oct(mode)}; run `chmod 600`)"
        )
    return expanded


def build_program_arguments(
    gateway_script: Path,
    *,
    asr: str | None = None,
    asr_model: str | None = None,
    relay: str | None = None,
    keep_awake: bool = False,
) -> list[str]:
    args: list[str] = []
    if keep_awake:
        args.extend(["/usr/bin/caffeinate", "-s"])
    args.append(str(gateway_script))
    if asr:
        args.extend(["--asr", asr])
    if asr_model:
        args.extend(["--asr-model", asr_model])
    if relay:
        args.extend(["--relay", relay])
    return args


def build_launchagent_dict(
    *,
    repo_root: Path | None = None,
    asr: str | None = None,
    asr_model: str | None = None,
    relay: str | None = None,
    keep_awake: bool = False,
    env_file: Path | None = None,
) -> dict[str, Any]:
    root = (repo_root or default_repo_root()).resolve()
    gateway_script = root / "scripts" / "start_mac_gateway.sh"
    log_path = root / ".local" / "launchagent.log"
    program_args = build_program_arguments(
        gateway_script,
        asr=asr,
        asr_model=asr_model,
        relay=relay,
        keep_awake=keep_awake,
    )
    local_bin = Path.home() / ".local" / "bin"
    path_env = f"{local_bin}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    env_vars: dict[str, str] = {"PATH": path_env}
    if env_file is not None:
        env_vars["XSA_ENV_FILE"] = str(env_file.expanduser().resolve())
    return {
        "Label": LAUNCHAGENT_LABEL,
        "ProgramArguments": program_args,
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
        "EnvironmentVariables": env_vars,
    }


def render_launchagent_plist(
    *,
    repo_root: Path | None = None,
    asr: str | None = None,
    asr_model: str | None = None,
    relay: str | None = None,
    keep_awake: bool = False,
    env_file: Path | None = None,
) -> bytes:
    payload = build_launchagent_dict(
        repo_root=repo_root,
        asr=asr,
        asr_model=asr_model,
        relay=relay,
        keep_awake=keep_awake,
        env_file=env_file,
    )
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=False)


def install_launchagent(
    plist_bytes: bytes,
    plist_path: Path = DEFAULT_PLIST_PATH,
    *,
    log_path: Path | None = None,
    force: bool = False,
) -> Path:
    target = plist_path.expanduser()
    if target.exists() and not force:
        raise FileExistsError(f"LaunchAgent plist already exists at {target}; pass --force to overwrite.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(plist_bytes)
    target.chmod(0o644)
    return target


def uninstall_launchagent(plist_path: Path = DEFAULT_PLIST_PATH) -> bool:
    target = plist_path.expanduser()
    if target.exists() or target.is_symlink():
        target.unlink()
        return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, install, or uninstall the macOS LaunchAgent for Xiao Speaker Assistant.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated plist XML without writing files (default).",
    )
    mode.add_argument(
        "--install",
        action="store_true",
        help=f"Write plist to {DEFAULT_PLIST_PATH}.",
    )
    mode.add_argument(
        "--uninstall",
        action="store_true",
        help=f"Remove plist at {DEFAULT_PLIST_PATH}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing plist when using --install.",
    )
    parser.add_argument(
        "--asr",
        choices=["sense_voice", "xiaoai_asr"],
        help="Speech recognition backend passed to start_mac_gateway.sh.",
    )
    parser.add_argument(
        "--asr-model",
        choices=["sense_voice", "paraformer", "doubao_flash", "openai_whisper"],
        help="ASR model passed to start_mac_gateway.sh.",
    )
    parser.add_argument(
        "--relay",
        help="Outbound WebSocket relay URL (e.g. wss://relay.example.com) passed to start_mac_gateway.sh.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=f"Optional 0600 shell env file (e.g. {DEFAULT_ENV_FILE}) sourced by start_mac_gateway.sh.",
    )
    parser.add_argument(
        "--keep-awake",
        action="store_true",
        help="Wrap start_mac_gateway.sh with `caffeinate -s` to prevent system sleep on AC power.",
    )
    parser.add_argument(
        "--plist-path",
        type=Path,
        default=DEFAULT_PLIST_PATH,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.uninstall:
        removed = uninstall_launchagent(args.plist_path)
        target = args.plist_path.expanduser()
        if removed:
            print(f"Removed LaunchAgent plist: {target}")
        else:
            print(f"LaunchAgent plist not found: {target}")
        return 0

    validated_env: Path | None = None
    if args.env_file is not None:
        try:
            validated_env = validate_env_file(args.env_file)
        except (FileNotFoundError, PermissionError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    repo_root = (args.repo_root or default_repo_root()).resolve()
    plist_bytes = render_launchagent_plist(
        repo_root=repo_root,
        asr=args.asr,
        asr_model=args.asr_model,
        relay=args.relay,
        keep_awake=args.keep_awake,
        env_file=validated_env,
    )

    if args.install:
        log_path = repo_root / ".local" / "launchagent.log"
        try:
            target = install_launchagent(
                plist_bytes,
                args.plist_path,
                log_path=log_path,
                force=args.force,
            )
        except FileExistsError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Installed LaunchAgent plist: {target}")
        print(f"To load now: launchctl load -w {target}")
        return 0

    sys.stdout.write(plist_bytes.decode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

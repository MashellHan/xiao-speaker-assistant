from __future__ import annotations

import importlib.util
import plistlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from xiao_speaker_assistant.voice.speaker_deployer import (
    SpeakerDeployer,
    build_boot_script,
    build_deploy_boot_script_command,
    build_restart_command,
    build_ssh_command,
    build_status_command,
    build_update_server_url_command,
    validate_server_url,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script_module(name: str, rel_path: str) -> ModuleType:
    script_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launchagent_mod = _load_script_module("xsa_install_launchagent", "scripts/install_launchagent.py")
deploy_cli_mod = _load_script_module("xsa_deploy_speaker_client", "scripts/deploy_speaker_client.py")


def test_launchagent_plist_generation_defaults_and_flags(tmp_path: Path) -> None:
    plist_bytes = launchagent_mod.render_launchagent_plist(
        repo_root=tmp_path,
        asr="sense_voice",
        asr_model="paraformer",
        relay="wss://relay.example.com",
        keep_awake=False,
    )
    data = plistlib.loads(plist_bytes)

    assert data["Label"] == "com.mashellhan.xiao-speaker-assistant"
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is True
    assert data["WorkingDirectory"] == str(tmp_path.resolve())
    expected_log = str(tmp_path.resolve() / ".local" / "launchagent.log")
    assert data["StandardOutPath"] == expected_log
    assert data["StandardErrorPath"] == expected_log

    expected_gateway = str(tmp_path.resolve() / "scripts" / "start_mac_gateway.sh")
    assert data["ProgramArguments"] == [
        expected_gateway,
        "--asr",
        "sense_voice",
        "--asr-model",
        "paraformer",
        "--relay",
        "wss://relay.example.com",
    ]


def test_launchagent_keep_awake_wraps_caffeinate_s(tmp_path: Path) -> None:
    plist_bytes = launchagent_mod.render_launchagent_plist(
        repo_root=tmp_path,
        asr="xiaoai_asr",
        keep_awake=True,
    )
    data = plistlib.loads(plist_bytes)
    prog_args = data["ProgramArguments"]

    assert prog_args[0] == "/usr/bin/caffeinate"
    assert prog_args[1] == "-s"
    assert prog_args[2] == str(tmp_path.resolve() / "scripts" / "start_mac_gateway.sh")
    assert prog_args[3:] == ["--asr", "xiaoai_asr"]


def test_launchagent_cli_dry_run_install_force_and_uninstall(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    plist_target = tmp_path / "LaunchAgents" / "com.mashellhan.xiao-speaker-assistant.plist"

    # Default (no --install/--uninstall) acts as --dry-run and prints valid XML plist.
    rc = launchagent_mod.main(
        [
            "--repo-root",
            str(fake_repo),
            "--plist-path",
            str(plist_target),
            "--asr",
            "sense_voice",
            "--keep-awake",
        ]
    )
    assert rc == 0
    assert not plist_target.exists()
    out = capsys.readouterr().out
    parsed = plistlib.loads(out.encode("utf-8"))
    assert parsed["Label"] == "com.mashellhan.xiao-speaker-assistant"
    assert parsed["ProgramArguments"][:2] == ["/usr/bin/caffeinate", "-s"]

    # --install writes plist and creates .local log parent directory.
    rc = launchagent_mod.main(
        [
            "--install",
            "--repo-root",
            str(fake_repo),
            "--plist-path",
            str(plist_target),
            "--relay",
            "wss://relay.example.com",
        ]
    )
    assert rc == 0
    assert plist_target.is_file()
    assert (fake_repo / ".local").is_dir()
    installed = plistlib.loads(plist_target.read_bytes())
    assert "--relay" in installed["ProgramArguments"]
    capsys.readouterr()

    # Refuses overwrite without --force.
    plist_target.write_text("sentinel", encoding="utf-8")
    rc = launchagent_mod.main(
        [
            "--install",
            "--repo-root",
            str(fake_repo),
            "--plist-path",
            str(plist_target),
        ]
    )
    assert rc == 1
    assert plist_target.read_text(encoding="utf-8") == "sentinel"
    assert "already exists" in capsys.readouterr().err

    # --force overwrites cleanly.
    rc = launchagent_mod.main(
        [
            "--install",
            "--force",
            "--repo-root",
            str(fake_repo),
            "--plist-path",
            str(plist_target),
        ]
    )
    assert rc == 0
    assert plistlib.loads(plist_target.read_bytes())["Label"] == "com.mashellhan.xiao-speaker-assistant"
    capsys.readouterr()

    # --uninstall removes the file and is idempotent.
    assert launchagent_mod.main(["--uninstall", "--plist-path", str(plist_target)]) == 0
    assert not plist_target.exists()
    assert launchagent_mod.main(["--uninstall", "--plist-path", str(plist_target)]) == 0


def test_boot_script_busybox_features_and_shell_syntax(tmp_path: Path) -> None:
    script = build_boot_script()
    assert script.startswith("#!/bin/sh\n")
    assert "ping -c 1 '223.5.5.5'" in script or "ping -c 1 223.5.5.5" in script
    assert "tr -d '\\r\\n '" in script
    assert "while true; do" in script
    assert "/data/xiaoai-agent" in script
    assert "/tmp/xiaoai-agent.log" in script

    script_file = tmp_path / "boot.sh"
    script_file.write_text(script, encoding="utf-8")
    syntax_check = subprocess.run(
        ["/bin/sh", "-n", str(script_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert syntax_check.returncode == 0, syntax_check.stderr


def test_build_ssh_command_options_and_target() -> None:
    cmd = build_ssh_command(
        "192.168.1.50",
        "~/.ssh/xiaoai_oh2p_rsa",
        ".local/known_hosts",
        "df -h /data",
    )
    assert cmd[0] == "ssh"
    assert "-i" in cmd
    key_idx = cmd.index("-i")
    assert cmd[key_idx + 1] == str(Path("~/.ssh/xiaoai_oh2p_rsa").expanduser())
    assert "IdentitiesOnly=yes" in cmd
    assert "BatchMode=yes" in cmd
    assert "UserKnownHostsFile=.local/known_hosts" in cmd
    assert "HostKeyAlgorithms=+ssh-rsa" in cmd
    assert "PubkeyAcceptedAlgorithms=+ssh-rsa" in cmd
    assert cmd[-2] == "root@192.168.1.50"
    assert cmd[-1] == "df -h /data"

    custom = build_ssh_command(
        "admin@oh2p.local",
        "/tmp/key",
        "/tmp/kh",
        "ps",
        port=2222,
    )
    assert custom[-4:-2] == ["-p", "2222"]
    assert custom[-2] == "admin@oh2p.local"


def test_server_url_validation_and_remote_commands() -> None:
    assert validate_server_url(" ws://192.168.1.10:4399 ") == "ws://192.168.1.10:4399"
    assert (
        validate_server_url("wss://relay.example.com/ws/speaker")
        == "wss://relay.example.com/ws/speaker"
    )
    for bad in ("", "   ", "http://192.168.1.10:4399", "ws://bad host/path", "wss://"):
        with pytest.raises(ValueError):
            validate_server_url(bad)

    lan_cmd = build_update_server_url_command("ws://192.168.1.10:4399")
    assert "/data/xiaoai-agent/server.txt" in lan_cmd
    assert "ws://192.168.1.10:4399" in lan_cmd

    relay_cmd = build_update_server_url_command("wss://relay.example.com/ws/speaker")
    assert "wss://relay.example.com/ws/speaker" in relay_cmd

    boot_cmd = build_deploy_boot_script_command()
    assert "/data/xiaoai-agent/boot.sh" in boot_cmd
    assert "chmod 755" in boot_cmd
    assert "start_tunnel_if_configured" in boot_cmd
    assert "tunnel.conf" in boot_cmd

    from xiao_speaker_assistant.voice.speaker_deployer import (
        build_deploy_tunnel_config_command,
        build_upload_tunnel_key_command,
    )

    tun_cmd = build_deploy_tunnel_config_command("relay.example.com", tunnel_port=2222)
    assert "TUNNEL_HOST=relay.example.com" in tun_cmd
    assert "TUNNEL_PORT=2222" in tun_cmd
    assert "chmod 600" in tun_cmd

    tun_key_cmd = build_upload_tunnel_key_command()
    assert "tunnel_id_dropbear" in tun_key_cmd
    assert "chmod 600" in tun_key_cmd

    status_cmd = build_status_command()
    assert "ps" in status_cmd
    assert "df -h /data" in status_cmd
    assert "/tmp/xiaoai-agent.log" in status_cmd
    assert status_cmd.count("token=REDACTED") >= 2

    restart_cmd = build_restart_command()
    assert "start-stop-daemon -S -b" in restart_cmd
    assert "/data/xiaoai-agent/boot.sh" in restart_cmd


def test_speaker_deployer_dry_run_never_executes_ssh(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[Any] = []

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise AssertionError("subprocess runner must not be called in dry-run mode")

    fake_bin = tmp_path / "client"
    fake_bin.write_bytes(b"ELF-dummy")
    known_hosts = tmp_path / ".local" / "known_hosts"

    rc = deploy_cli_mod.main(
        [
            "--host",
            "192.168.1.88",
            "--known-hosts",
            str(known_hosts),
            "--client-bin",
            str(fake_bin),
            "--install-boot",
            "--server-url",
            "wss://relay.example.com/ws/speaker",
            "--restart",
            "--status",
            "--dry-run",
        ],
        runner=fail_if_called,
    )
    assert rc == 0
    assert calls == []
    assert not known_hosts.parent.exists()
    out = capsys.readouterr().out
    assert "[DRY-RUN] upload-client:" in out
    assert "[DRY-RUN] install-boot:" in out
    assert "[DRY-RUN] update-server-url:" in out
    assert "[DRY-RUN] restart:" in out
    assert "[DRY-RUN] status:" in out
    assert "HostKeyAlgorithms=+ssh-rsa" in out
    assert "PubkeyAcceptedAlgorithms=+ssh-rsa" in out


def test_speaker_deployer_executes_ssh_commands_in_order(tmp_path: Path) -> None:
    recorded: list[tuple[list[str], bytes | None]] = []

    def fake_runner(
        cmd: list[str],
        *,
        input: bytes | None = None,
        stdout: Any = None,
        stderr: Any = None,
        check: bool = False,
    ) -> SimpleNamespace:
        recorded.append((cmd, input))
        return SimpleNamespace(returncode=0, stdout=b"ok\n", stderr=b"")

    fake_bin = tmp_path / "client"
    fake_bin.write_bytes(b"\x7fELF-test")
    known_hosts = tmp_path / "state" / "known_hosts"

    deployer = SpeakerDeployer(
        "192.168.1.99",
        key_path=tmp_path / "xiaoai_oh2p_rsa",
        known_hosts=known_hosts,
        dry_run=False,
        runner=fake_runner,
    )

    assert deployer.upload_client(fake_bin).returncode == 0
    assert deployer.install_boot_script().returncode == 0
    assert deployer.update_server_url("ws://192.168.1.2:4399").returncode == 0
    assert deployer.restart().returncode == 0
    assert deployer.status().returncode == 0

    assert known_hosts.parent.is_dir()
    assert len(recorded) == 5
    assert recorded[0][1] == b"\x7fELF-test"
    assert "cat > /data/xiaoai-agent/client.tmp" in recorded[0][0][-1]
    assert "/data/xiaoai-agent/boot.sh" in recorded[1][0][-1]
    assert "ws://192.168.1.2:4399" in recorded[2][0][-1]
    assert "start-stop-daemon -S -b" in recorded[3][0][-1]
    assert "df -h /data" in recorded[4][0][-1]


def test_deploy_cli_print_boot_script_and_validation(capsys: pytest.CaptureFixture[str]) -> None:
    rc = deploy_cli_mod.main(["--print-boot-script"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "#!/bin/sh" in out
    assert "ping -c 1 223.5.5.5" in out

    bad_rc = deploy_cli_mod.main(
        ["--host", "192.168.1.88", "--server-url", "http://not-ws", "--dry-run"]
    )
    assert bad_rc == 2
    assert "ws:// or wss://" in capsys.readouterr().err

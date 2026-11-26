"""Pure-SSH deployment and management helper for `/data/xiaoai-agent/` on OH2P."""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

DEFAULT_SSH_USER = "root"
DEFAULT_KEY_PATH = Path("~/.ssh/xiaoai_oh2p_rsa")
DEFAULT_KNOWN_HOSTS = Path(".local/known_hosts")
DEFAULT_REMOTE_DIR = "/data/xiaoai-agent"
DEFAULT_LOG_PATH = "/tmp/xiaoai-agent.log"
DEFAULT_PID_FILE = "/tmp/xiaoai-agent-boot.pid"
DEFAULT_TUNNEL_PID_FILE = "/tmp/xiaoai-agent-tunnel.pid"
DEFAULT_TUNNEL_SUP_PID_FILE = "/tmp/xiaoai-agent-tunnel-sup.pid"
_TUNNEL_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,252}$")
_TUNNEL_USER_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class DeployResult:
    command: list[str]
    remote_cmd: str
    returncode: int
    stdout: str
    stderr: str
    dry_run: bool = False


def validate_server_url(server_url: str) -> str:
    cleaned = server_url.strip()
    if not cleaned:
        raise ValueError("Server URL must not be empty.")
    if any(ch.isspace() for ch in cleaned):
        raise ValueError(f"Server URL must not contain whitespace: {server_url!r}")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError(
            f"Server URL must be a valid ws:// or wss:// WebSocket URL, got: {server_url!r}"
        )
    return cleaned


def build_boot_script(
    *,
    agent_dir: str = DEFAULT_REMOTE_DIR,
    binary_name: str = "client",
    server_file: str = "server.txt",
    tunnel_conf_file: str = "tunnel.conf",
    log_file: str = DEFAULT_LOG_PATH,
    pid_file: str = DEFAULT_PID_FILE,
    tunnel_pid_file: str = DEFAULT_TUNNEL_PID_FILE,
    tunnel_sup_pid_file: str = DEFAULT_TUNNEL_SUP_PID_FILE,
    ping_target: str = "223.5.5.5",
    max_net_wait_attempts: int = 60,
    restart_delay: int = 5,
) -> str:
    """Return a BusyBox `/bin/sh` compatible `boot.sh` for `/data/xiaoai-agent/`."""
    return f"""#!/bin/sh
AGENT_DIR={shlex.quote(agent_dir)}
CLIENT_BIN="${{AGENT_DIR}}/{binary_name}"
SERVER_FILE="${{AGENT_DIR}}/{server_file}"
TUNNEL_CONF="${{AGENT_DIR}}/{tunnel_conf_file}"
LOG_FILE={shlex.quote(log_file)}
PID_FILE={shlex.quote(pid_file)}
TUNNEL_PID_FILE={shlex.quote(tunnel_pid_file)}
TUNNEL_SUP_PID_FILE={shlex.quote(tunnel_sup_pid_file)}
TUNNEL_SUP_PID=""

echo $$ > "$PID_FILE"
cd "$AGENT_DIR" || exit 1

cleanup_tunnel() {{
    if [ -n "$TUNNEL_SUP_PID" ]; then
        kill "$TUNNEL_SUP_PID" 2>/dev/null || true
        TUNNEL_SUP_PID=""
    fi
    if [ -f "$TUNNEL_SUP_PID_FILE" ]; then
        kill "$(cat "$TUNNEL_SUP_PID_FILE")" 2>/dev/null || true
        rm -f "$TUNNEL_SUP_PID_FILE"
    fi
    if [ -f "$TUNNEL_PID_FILE" ]; then
        kill "$(cat "$TUNNEL_PID_FILE")" 2>/dev/null || true
        rm -f "$TUNNEL_PID_FILE"
    fi
}}
trap cleanup_tunnel EXIT
trap 'cleanup_tunnel; exit 0' INT TERM

i=0
while [ "$i" -lt {max_net_wait_attempts} ]; do
    if ping -c 1 {shlex.quote(ping_target)} >/dev/null 2>&1; then
        break
    fi
    i=$((i + 1))
    sleep 2
done

# Run Dropbear SSH tunnel in its own independent supervisor loop so a Wi-Fi blip or
# dead link immediately auto-reconnects even while open-xiaoai-client stays running.
run_tunnel_supervisor() {{
    CHILD_SSH=""
    trap 'if [ -n "$CHILD_SSH" ]; then kill "$CHILD_SSH" 2>/dev/null || true; fi; rm -f "$TUNNEL_PID_FILE"; exit 0' INT TERM EXIT
    while true; do
        if [ ! -f "$TUNNEL_CONF" ]; then
            sleep {restart_delay}
            continue
        fi
        TUNNEL_HOST=""
        TUNNEL_PORT="2222"
        TUNNEL_USER="xsa-tunnel"
        TUNNEL_LOCAL_PORT="8765"
        TUNNEL_REMOTE_PORT="8765"
        TUNNEL_KEY="${{AGENT_DIR}}/tunnel_id_dropbear"
        # shellcheck disable=SC1090
        . "$TUNNEL_CONF"
        if [ -z "$TUNNEL_HOST" ] || [ ! -f "$TUNNEL_KEY" ]; then
            sleep {restart_delay}
            continue
        fi
        echo "$(date): starting encrypted SSH tunnel -> ${{TUNNEL_USER}}@${{TUNNEL_HOST}}:${{TUNNEL_PORT}} (127.0.0.1:${{TUNNEL_LOCAL_PORT}} -> 127.0.0.1:${{TUNNEL_REMOTE_PORT}})" >> "$LOG_FILE"
        # Note: Dropbear -y performs TOFU (persisting first host key to ~/.ssh/known_hosts and rejecting changes).
        # -K 10 sends SSH keepalives every 10s; Dropbear disconnects dead links after 3 missed replies (~30s)
        # without killing healthy idle -N -L tunnels (unlike -I, which is an inactivity timeout).
        ssh -y -N -K 10 -i "$TUNNEL_KEY" -p "$TUNNEL_PORT" \\
            -L "127.0.0.1:${{TUNNEL_LOCAL_PORT}}:127.0.0.1:${{TUNNEL_REMOTE_PORT}}" \\
            "${{TUNNEL_USER}}@${{TUNNEL_HOST}}" >> "$LOG_FILE" 2>&1 &
        CHILD_SSH=$!
        echo "$CHILD_SSH" > "$TUNNEL_PID_FILE"
        wait "$CHILD_SSH"
        TUN_RC=$?
        CHILD_SSH=""
        rm -f "$TUNNEL_PID_FILE"
        echo "$(date): SSH tunnel exited ($TUN_RC), reconnecting in 3s" >> "$LOG_FILE"
        sleep 3
    done
}}

start_tunnel_if_configured() {{
    if [ ! -f "$TUNNEL_CONF" ]; then
        return 0
    fi
    if [ -n "$TUNNEL_SUP_PID" ] && kill -0 "$TUNNEL_SUP_PID" 2>/dev/null; then
        return 0
    fi
    cleanup_tunnel
    run_tunnel_supervisor &
    TUNNEL_SUP_PID=$!
    echo "$TUNNEL_SUP_PID" > "$TUNNEL_SUP_PID_FILE"
    sleep 2
    return 0
}}

while true; do
    SERVER_URL=""
    if [ -f "$SERVER_FILE" ]; then
        SERVER_URL=$(tr -d '\\r\\n ' < "$SERVER_FILE")
    fi

    if [ -z "$SERVER_URL" ]; then
        echo "$(date): waiting for $SERVER_FILE" >> "$LOG_FILE"
        sleep {restart_delay}
        continue
    fi

    if [ ! -x "$CLIENT_BIN" ]; then
        echo "$(date): missing executable $CLIENT_BIN" >> "$LOG_FILE"
        sleep {restart_delay}
        continue
    fi

    start_tunnel_if_configured

    echo "$(date): starting {binary_name} -> $(printf '%s' "$SERVER_URL" | sed 's/token=[^&#]*/token=REDACTED/g')" >> "$LOG_FILE"
    if [ -f "$AGENT_DIR/asound_xsa.conf" ]; then
        export ALSA_CONFIG_PATH="$AGENT_DIR/asound_xsa.conf"
    else
        unset ALSA_CONFIG_PATH
    fi
    "$CLIENT_BIN" "$SERVER_URL" >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    echo "$(date): {binary_name} exited ($EXIT_CODE), restarting in {restart_delay}s" >> "$LOG_FILE"
    sleep {restart_delay}
done
"""


def build_ssh_command(
    host: str,
    key_path: str | Path,
    known_hosts: str | Path,
    remote_cmd: str,
    *,
    user: str = DEFAULT_SSH_USER,
    port: int = 22,
    connect_timeout: int = 10,
) -> list[str]:
    """Build an OpenSSH argv list compatible with Dropbear RSA keys on OH2P."""
    target_host = host.strip()
    if not target_host:
        raise ValueError("Host must not be empty.")
    target = target_host if "@" in target_host else f"{user}@{target_host}"
    expanded_key = str(Path(key_path).expanduser())
    expanded_known_hosts = str(Path(known_hosts).expanduser())

    cmd = [
        "ssh",
        "-i",
        expanded_key,
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={expanded_known_hosts}",
        "-o",
        "HostKeyAlgorithms=+ssh-rsa",
        "-o",
        "PubkeyAcceptedAlgorithms=+ssh-rsa",
        "-o",
        f"ConnectTimeout={connect_timeout}",
    ]
    if port != 22:
        cmd.extend(["-p", str(port)])
    cmd.extend([target, remote_cmd])
    return cmd


def build_update_server_url_command(
    server_url: str,
    *,
    remote_dir: str = DEFAULT_REMOTE_DIR,
) -> str:
    valid_url = validate_server_url(server_url)
    q_dir = shlex.quote(remote_dir)
    q_file = shlex.quote(f"{remote_dir}/server.txt")
    q_url = shlex.quote(valid_url)
    return (
        f"mkdir -p {q_dir} && "
        f"(umask 077 && printf '%s\\n' {q_url} > {q_file}) && "
        f"chmod 600 {q_file}"
    )


def build_asound_xsa_conf() -> str:
    """Return ALSA config defining `pcm.xsa_aec` (ch0=mic, ch1=hw:0,3 ch3 speaker loopback)."""
    return """</usr/share/alsa/alsa.conf>

pcm.xsa_aec {
    type plug
    slave {
        pcm "Capture"
        channels 4
    }
    ttable.0.0 1
    ttable.1.3 1
}
"""


def build_deploy_asound_command(
    *,
    remote_dir: str = DEFAULT_REMOTE_DIR,
) -> str:
    conf = build_asound_xsa_conf()
    marker = "__XSA_ASOUND_EOF__"
    q_dir = shlex.quote(remote_dir)
    q_file = shlex.quote(f"{remote_dir}/asound_xsa.conf")
    return (
        f"mkdir -p {q_dir} && "
        f"cat <<'{marker}' > {q_file}\n"
        f"{conf}"
        f"{marker}\n"
        f"chmod 644 {q_file}"
    )


def build_deploy_boot_script_command(
    boot_script: str | None = None,
    *,
    remote_dir: str = DEFAULT_REMOTE_DIR,
) -> str:
    script = boot_script if boot_script is not None else build_boot_script(agent_dir=remote_dir)
    marker = "__XSA_BOOT_EOF__"
    if marker in script:
        raise ValueError("Boot script unexpectedly contains heredoc marker.")
    q_dir = shlex.quote(remote_dir)
    q_boot = shlex.quote(f"{remote_dir}/boot.sh")
    body = script if script.endswith("\n") else f"{script}\n"
    return (
        f"mkdir -p {q_dir} && "
        f"cat <<'{marker}' > {q_boot}\n"
        f"{body}"
        f"{marker}\n"
        f"chmod 755 {q_boot}"
    )


def build_deploy_tunnel_config_command(
    tunnel_host: str,
    *,
    tunnel_port: int = 2222,
    tunnel_user: str = "xsa-tunnel",
    local_port: int = 8765,
    remote_port: int = 8765,
    remote_dir: str = DEFAULT_REMOTE_DIR,
) -> str:
    """Write `/data/xiaoai-agent/tunnel.conf` (0600) for Dropbear encrypted SSH tunneling."""
    clean_host = tunnel_host.strip()
    if not clean_host or not _TUNNEL_HOST_RE.match(clean_host):
        raise ValueError(f"Invalid tunnel host: {tunnel_host!r}")
    clean_user = tunnel_user.strip()
    if not clean_user or not _TUNNEL_USER_RE.match(clean_user):
        raise ValueError(f"Invalid tunnel user: {tunnel_user!r}")
    for label, p in (("tunnel_port", tunnel_port), ("local_port", local_port), ("remote_port", remote_port)):
        if isinstance(p, bool) or not (1 <= int(p) <= 65535):
            raise ValueError(f"Invalid {label}: {p}")
    q_dir = shlex.quote(remote_dir)
    q_conf = shlex.quote(f"{remote_dir}/tunnel.conf")
    conf_body = (
        f"TUNNEL_HOST={shlex.quote(clean_host)}\n"
        f"TUNNEL_PORT={int(tunnel_port)}\n"
        f"TUNNEL_USER={shlex.quote(clean_user)}\n"
        f"TUNNEL_LOCAL_PORT={int(local_port)}\n"
        f"TUNNEL_REMOTE_PORT={int(remote_port)}\n"
    )
    marker = "__XSA_TUNNEL_CONF_EOF__"
    return (
        f"mkdir -p {q_dir} && "
        f"(umask 077 && cat <<'{marker}' > {q_conf}\n"
        f"{conf_body}"
        f"{marker}\n"
        f") && chmod 600 {q_conf}"
    )


def build_upload_tunnel_key_command(
    *,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    key_filename: str = "tunnel_id_dropbear",
) -> str:
    """Upload a Dropbear binary private key (0600) to `/data/xiaoai-agent/tunnel_id_dropbear`."""
    q_dir = shlex.quote(remote_dir)
    q_tmp = shlex.quote(f"{remote_dir}/{key_filename}.tmp")
    q_dst = shlex.quote(f"{remote_dir}/{key_filename}")
    return (
        f"mkdir -p {q_dir} && "
        f"(umask 077 && cat > {q_tmp}) && "
        f"chmod 600 {q_tmp} && "
        f"mv {q_tmp} {q_dst} && "
        f"chmod 600 {q_dst}"
    )


def build_upload_client_command(
    *,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    binary_name: str = "client",
) -> str:
    q_dir = shlex.quote(remote_dir)
    q_tmp = shlex.quote(f"{remote_dir}/{binary_name}.tmp")
    q_bin = shlex.quote(f"{remote_dir}/{binary_name}")
    return (
        f"mkdir -p {q_dir} && "
        f"cat > {q_tmp} && "
        f"chmod 755 {q_tmp} && "
        f"mv {q_tmp} {q_bin}"
    )


def build_status_command(
    *,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    log_path: str = DEFAULT_LOG_PATH,
    tail_lines: int = 40,
) -> str:
    q_dir = shlex.quote(remote_dir)
    q_server = shlex.quote(f"{remote_dir}/server.txt")
    q_tunnel = shlex.quote(f"{remote_dir}/tunnel.conf")
    q_log = shlex.quote(log_path)
    return (
        'echo "=== Processes ===" && '
        '(ps w 2>/dev/null || ps) | grep -E "xiaoai-agent|client|boot\\.sh|tunnel_id_dropbear" | grep -v grep | sed "s/token=[^&# ]*/token=REDACTED/g" || true; '
        'echo "=== /data Free Space ===" && '
        'df -h /data && '
        'echo "=== server.txt ===" && '
        f'(sed "s/token=[^&#]*/token=REDACTED/g" {q_server} 2>/dev/null || echo "(missing in {q_dir})") && '
        'echo "=== tunnel.conf ===" && '
        f'(cat {q_tunnel} 2>/dev/null || echo "(not configured)") && '
        f'echo "=== {log_path} ===" && '
        f'(tail -n {tail_lines} {q_log} 2>/dev/null || echo "(no log)")'
    )


def build_restart_command(
    *,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    pid_file: str = DEFAULT_PID_FILE,
    tunnel_pid_file: str = DEFAULT_TUNNEL_PID_FILE,
    tunnel_sup_pid_file: str = DEFAULT_TUNNEL_SUP_PID_FILE,
    binary_name: str = "client",
) -> str:
    q_boot = shlex.quote(f"{remote_dir}/boot.sh")
    q_pid = shlex.quote(pid_file)
    q_tun_sup_pid = shlex.quote(tunnel_sup_pid_file)
    q_tun_pid = shlex.quote(tunnel_pid_file)
    q_bin = shlex.quote(binary_name)
    return (
        f"if [ -f {q_pid} ]; then "
        f"kill $(cat {q_pid}) 2>/dev/null || true; "
        f"rm -f {q_pid}; "
        "fi; "
        f"if [ -f {q_tun_sup_pid} ]; then "
        f"kill $(cat {q_tun_sup_pid}) 2>/dev/null || true; "
        f"rm -f {q_tun_sup_pid}; "
        "fi; "
        f"if [ -f {q_tun_pid} ]; then "
        f"kill $(cat {q_tun_pid}) 2>/dev/null || true; "
        f"rm -f {q_tun_pid}; "
        "fi; "
        f"killall {q_bin} 2>/dev/null || true; "
        "sleep 1; "
        f"start-stop-daemon -S -b -x /bin/sh -- {q_boot}"
    )


class SpeakerDeployer:
    """Manage `/data/xiaoai-agent/` on an OH2P speaker exclusively over SSH."""

    def __init__(
        self,
        host: str,
        *,
        user: str = DEFAULT_SSH_USER,
        key_path: str | Path = DEFAULT_KEY_PATH,
        known_hosts: str | Path = DEFAULT_KNOWN_HOSTS,
        remote_dir: str = DEFAULT_REMOTE_DIR,
        port: int = 22,
        dry_run: bool = False,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.host = host
        self.user = user
        self.key_path = Path(key_path)
        self.known_hosts = Path(known_hosts)
        self.remote_dir = remote_dir
        self.port = port
        self.dry_run = dry_run
        self._runner = runner

    def build_command(self, remote_cmd: str) -> list[str]:
        return build_ssh_command(
            self.host,
            self.key_path,
            self.known_hosts,
            remote_cmd,
            user=self.user,
            port=self.port,
        )

    def run_remote(
        self,
        remote_cmd: str,
        *,
        stdin_bytes: bytes | None = None,
    ) -> DeployResult:
        cmd = self.build_command(remote_cmd)
        if self.dry_run:
            return DeployResult(
                command=cmd,
                remote_cmd=remote_cmd,
                returncode=0,
                stdout="",
                stderr="",
                dry_run=True,
            )

        self.known_hosts.expanduser().parent.mkdir(parents=True, exist_ok=True)
        proc = self._runner(
            cmd,
            input=stdin_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout = (
            proc.stdout.decode("utf-8", errors="replace")
            if isinstance(proc.stdout, bytes)
            else (proc.stdout or "")
        )
        stderr = (
            proc.stderr.decode("utf-8", errors="replace")
            if isinstance(proc.stderr, bytes)
            else (proc.stderr or "")
        )
        return DeployResult(
            command=cmd,
            remote_cmd=remote_cmd,
            returncode=int(proc.returncode),
            stdout=stdout,
            stderr=stderr,
            dry_run=False,
        )

    def update_server_url(self, server_url: str) -> DeployResult:
        return self.run_remote(
            build_update_server_url_command(server_url, remote_dir=self.remote_dir)
        )

    def install_boot_script(self, boot_script: str | None = None) -> DeployResult:
        return self.run_remote(
            build_deploy_boot_script_command(boot_script, remote_dir=self.remote_dir)
        )

    def install_aec_alsa(self) -> DeployResult:
        return self.run_remote(build_deploy_asound_command(remote_dir=self.remote_dir))

    def upload_tunnel_key(self, local_key_path: Path) -> DeployResult:
        key_file = local_key_path.expanduser()
        if not key_file.is_file():
            raise FileNotFoundError(f"Tunnel key file not found: {key_file}")
        payload = None if self.dry_run else key_file.read_bytes()
        return self.run_remote(
            build_upload_tunnel_key_command(remote_dir=self.remote_dir),
            stdin_bytes=payload,
        )

    def install_tunnel_config(
        self,
        tunnel_host: str,
        *,
        tunnel_port: int = 2222,
        tunnel_user: str = "xsa-tunnel",
        local_port: int = 8765,
        remote_port: int = 8765,
    ) -> DeployResult:
        return self.run_remote(
            build_deploy_tunnel_config_command(
                tunnel_host,
                tunnel_port=tunnel_port,
                tunnel_user=tunnel_user,
                local_port=local_port,
                remote_port=remote_port,
                remote_dir=self.remote_dir,
            )
        )

    def upload_client(self, local_binary: Path) -> DeployResult:
        binary_path = local_binary.expanduser()
        if not binary_path.is_file():
            raise FileNotFoundError(f"Client binary not found: {binary_path}")
        payload = None if self.dry_run else binary_path.read_bytes()
        return self.run_remote(
            build_upload_client_command(remote_dir=self.remote_dir),
            stdin_bytes=payload,
        )

    def status(self) -> DeployResult:
        return self.run_remote(build_status_command(remote_dir=self.remote_dir))

    def restart(self) -> DeployResult:
        return self.run_remote(build_restart_command(remote_dir=self.remote_dir))

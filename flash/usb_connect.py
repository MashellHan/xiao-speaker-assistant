#!/usr/bin/env python3
"""Catch the ~1.6s Amlogic WorldCup USB window on power cycle and run `identify 7`."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from usb_probe import UsbDevice, parse_ioreg_usb

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".local"


def query_usb() -> dict[str, list[UsbDevice]]:
    res = subprocess.run(
        ["/usr/sbin/ioreg", "-p", "IOUSB", "-w0", "-l", "-r", "-c", "IOUSBHostDevice"],
        capture_output=True,
        text=True,
        timeout=3,
        check=True,
    )
    grouped: dict[str, list[UsbDevice]] = {
        "normal": [],
        "worldcup": [],
        "other_amlogic": [],
    }
    for dev in parse_ioreg_usb(res.stdout):
        if dev.mode == "normal-oh2p":
            grouped["normal"].append(dev)
        elif dev.mode == "amlogic-worldcup-flash":
            grouped["worldcup"].append(dev)
        elif dev.mode == "amlogic-other":
            grouped["other_amlogic"].append(dev)
    return grouped


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually wait and run `update identify 7`")
    parser.add_argument("--duration", type=float, default=120.0, help="Wait timeout in seconds (3..300)")
    args = parser.parse_args()

    if not 3.0 <= args.duration <= 300.0:
        parser.error("--duration must be between 3 and 300")

    if not args.execute:
        print("DRY RUN: would wait for normal-oh2p -> amlogic-worldcup-flash and run `.local/update identify 7`.")
        print("Run with `--execute` when ready.")
        return 0

    update_bin = LOCAL / "update"
    tool_meta = LOCAL / "tool.json"
    if not update_bin.is_file() or not tool_meta.is_file():
        print("STOP: run `python3 flash/get_flash_tool.py` first.")
        return 1
    if sha256_file(update_bin) != json.loads(tool_meta.read_text())["sha256"]:
        print("STOP: .local/update SHA-256 mismatch.")
        return 1

    grouped = query_usb()
    if len(grouped["normal"]) != 1 or grouped["worldcup"] or grouped["other_amlogic"]:
        print(f"STOP: expected exactly one normal OH2P device, got: {grouped}")
        return 1

    port = grouped["normal"][0].location_id
    print(
        f"READY_TO_PLUG (port={port}): 请拔掉音箱 12V DC 电源插头，停 3 秒后插回（Type-C 线不要拔）。",
        flush=True,
    )

    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        grouped = query_usb()
        if len(grouped["normal"]) > 1 or grouped["other_amlogic"] or len(grouped["worldcup"]) > 1:
            print(f"STOP: USB device ambiguity: {grouped}")
            return 1
        if grouped["worldcup"]:
            wc = grouped["worldcup"][0]
            if wc.location_id != port:
                print(f"STOP: WorldCup appeared on different port {wc.location_id} (expected {port})")
                return 1
            proc = subprocess.run(
                [str(update_bin), "identify", "7"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=8,
                check=False,
            )
            log_path = LOCAL / "identify.log"
            with log_path.open("ab") as log:
                os.fchmod(log.fileno(), 0o600)
                log.write(proc.stdout)

            if b"No [WorldCup" in proc.stdout:
                print(
                    "STOP: 刷机设备未能连接 Mac（可能是 macOS 弹出“允许配件连接”拦截了窗口）。"
                    "请在 系统设置 -> 隐私与安全性 将“允许配件连接”临时改为解锁时自动允许后重试。"
                )
                return 1
            if proc.returncode != 0 or b"This firmware version is" not in proc.stdout or b"ERR" in proc.stdout:
                print("STOP: `update identify 7` failed; check .local/identify.log")
                return 1

            usb_meta = LOCAL / "usb.json"
            usb_meta.write_text(json.dumps({"port": port}, indent=2), encoding="utf-8")
            usb_meta.chmod(0o600)
            print(f"OK: captured WorldCup flash mode on port {port}.")
            print("NEXT: python3 flash/backup_partitions.py --execute")
            return 0
        if grouped["normal"] and grouped["normal"][0].location_id != port:
            print("STOP: normal OH2P changed USB port.")
            return 1
        time.sleep(0.05)

    print("STOP: timed out waiting for WorldCup flash window.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

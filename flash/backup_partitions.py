#!/usr/bin/env python3
"""Read-only double-read backup of boot0/boot1/system0/system1 from OH2P in WorldCup mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

from usb_connect import query_usb, sha256_file

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".local"
SYSTEM_PART_BYTES = 0x02800000  # 40 MiB


def ensure_worldcup_snapshot(port: str) -> None:
    grouped = query_usb()
    if (
        len(grouped["worldcup"]) != 1
        or grouped["worldcup"][0].location_id != port
        or grouped["normal"]
        or grouped["other_amlogic"]
    ):
        raise RuntimeError(f"WorldCup must be unique at {port}; got {grouped}")


def run_update(update_bin: Path, port: str, *cmd: str | Path, allow_fail: bool = False) -> tuple[int, str]:
    ensure_worldcup_snapshot(port)
    proc = subprocess.run(
        [str(update_bin), *[str(c) for c in cmd]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    text = proc.stdout.decode(errors="replace")
    with (LOCAL / "update.log").open("ab") as log:
        os.fchmod(log.fileno(), 0o600)
        log.write(proc.stdout)
    ensure_worldcup_snapshot(port)
    if not allow_fail and (proc.returncode != 0 or any(w in text.lower() for w in ("err", "fail", "timeout"))):
        raise RuntimeError(f"update command failed: {cmd}")
    return proc.returncode, text


def detect_boot_part(update_bin: Path, port: str) -> str:
    results: list[bool] = []
    for expr in ("test 1 = 1", "test 1 = 0", "test ${boot_part} = boot0", "test ${boot_part} = boot1"):
        rc, text = run_update(update_bin, port, "bulkcmd", "     " + expr, allow_fail=True)
        ok = rc == 0 and "bulkInReply success" in text and "failed" not in text.lower()
        bad = "failed" in text.lower() and not ok
        if not (ok or bad):
            raise RuntimeError(f"Unknown U-Boot test output for `{expr}`")
        results.append(ok)
    if results[:2] != [True, False] or results[2:] not in ([True, False], [False, True]):
        raise RuntimeError(f"boot_part calibration failed: {results}")
    return "boot0" if results[2] else "boot1"


def read_partition(update_bin: Path, port: str, part: str, size: int, dest: Path) -> None:
    if dest.exists():
        raise RuntimeError(f"Backup file already exists (refusing to overwrite): {dest}")
    run_update(update_bin, port, "mread", "store", part, "normal", hex(size), dest)
    dest.chmod(0o600)
    if dest.stat().st_size != size:
        raise RuntimeError(f"Read size mismatch for {dest.name}: {dest.stat().st_size} != {size}")


def android_boot_size(header_bytes: bytes) -> int:
    if header_bytes[:8] != b"ANDROID!":
        raise RuntimeError("Invalid Android boot header magic")
    ks, _ka, rs, _ra, ss, _sa, _ta, pg, dt, _un = struct.unpack_from("<10I", header_bytes, 8)
    if pg != 2048 or max(ks, rs, ss, dt) > 10_000_000:
        raise RuntimeError(f"Unexpected boot header dimensions: pg={pg}")
    return pg + sum((x + pg - 1) // pg * pg for x in (ks, rs, ss, dt)) + 512


def reset_to_normal(update_bin: Path, port: str) -> None:
    ensure_worldcup_snapshot(port)
    with (LOCAL / "update.log").open("ab") as log:
        subprocess.run(
            [str(update_bin), "bulkcmd", "     reset"],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=20,
            check=False,
        )
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        grouped = query_usb()
        if len(grouped["normal"]) == 1 and not grouped["worldcup"]:
            return
        time.sleep(0.2)
    raise RuntimeError("Speaker did not return to normal USB mode within 60s")


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Read partitions from OH2P")
    parser.add_argument("--reset-after", action="store_true", help="Reset speaker back to normal mode after backup")
    parser.add_argument("--reset-only", action="store_true", help="Only reset speaker back to normal mode")
    args = parser.parse_args()

    if not args.execute:
        print("DRY RUN: would calibrate boot_part and double-read boot0/boot1/system0/system1 (read-only).")
        print("Run with `--execute` (optionally `--reset-after`) to run.")
        return 0

    update_bin = LOCAL / "update"
    if sha256_file(update_bin) != json.loads((LOCAL / "tool.json").read_text())["sha256"]:
        print("STOP: .local/update SHA-256 mismatch.")
        return 1
    port = json.loads((LOCAL / "usb.json").read_text())["port"]

    if args.reset_only:
        reset_to_normal(update_bin, port)
        print("OK: speaker reset to normal mode.")
        return 0

    active_boot = detect_boot_part(update_bin, port)
    (LOCAL / "boot-part.json").write_text(json.dumps({"boot_part": active_boot}, indent=2), encoding="utf-8")
    (LOCAL / "boot-part.json").chmod(0o600)
    print(f"Detected active boot_part: {active_boot}")

    backup_dir = LOCAL / "backup"
    backup_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(backup_dir, 0o700)

    hashes: dict[str, str] = {}
    empty_partitions: list[str] = []
    for part in ("boot0", "boot1", "system0", "system1"):
        p1 = backup_dir / f"{part}.1.bin"
        p2 = backup_dir / f"{part}.2.bin"

        if p1.exists() and p2.exists():
            h1, h2 = sha256_file(p1), sha256_file(p2)
            if h1 != h2:
                raise RuntimeError(f"Existing backup hash mismatch on {part}: {h1} != {h2}")
            hashes[p1.name] = h1
            hashes[p2.name] = h2
            print(f"  [OK] {part} already backed up sha256={h1[:16]}...")
            continue

        if part.startswith("boot"):
            head_path = backup_dir / f"{part}.head"
            if not head_path.exists():
                read_partition(update_bin, port, part, 4096, head_path)
            head_bytes = head_path.read_bytes()
            if part != active_boot and (all(b == 0xFF for b in head_bytes) or all(b == 0x00 for b in head_bytes)):
                print(f"  [SKIP] inactive {part} is empty/erased")
                empty_partitions.append(part)
                continue
            size = android_boot_size(head_bytes)
        else:
            size = SYSTEM_PART_BYTES

        print(f"Reading {part} twice ({size} bytes)...", flush=True)
        read_partition(update_bin, port, part, size, p1)
        read_partition(update_bin, port, part, size, p2)
        h1, h2 = sha256_file(p1), sha256_file(p2)
        if h1 != h2:
            raise RuntimeError(f"Double-read SHA-256 mismatch on {part}: {h1} != {h2}")
        hashes[p1.name] = h1
        hashes[p2.name] = h2
        print(f"  [OK] {part} verified sha256={h1[:16]}...")

    manifest = LOCAL / "backup.json"
    manifest.write_text(
        json.dumps(
            {"boot_part": active_boot, "empty_partitions": empty_partitions, "files": hashes},
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest.chmod(0o600)

    if args.reset_after:
        reset_to_normal(update_bin, port)
        print("OK: speaker reset back to normal mode.")

    print("OK: read-only backup complete.")
    print("NEXT: python3 flash/extract_active_rootfs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

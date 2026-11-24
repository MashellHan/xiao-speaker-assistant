#!/usr/bin/env python3
"""Write patched rootfs ONCE to the active system partition and verify readback SHA-256."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from backup_partitions import detect_boot_part, read_partition, reset_to_normal, run_update
from usb_connect import sha256_file

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".local"

SYSTEM_FOR_BOOT = {"boot0": "system0", "boot1": "system1"}


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-accept-brick-risk",
        action="store_true",
        help="Actually write patched SquashFS once to the active system partition",
    )
    args = parser.parse_args()

    patch_meta_path = LOCAL / "patch.json"
    backup_meta_path = LOCAL / "backup.json"
    if not patch_meta_path.is_file() or not backup_meta_path.is_file():
        print("STOP: run backup_partitions.py, extract_active_rootfs.py, and make_patch.py first.")
        return 1

    patch_meta = json.loads(patch_meta_path.read_text(encoding="utf-8"))
    backup_meta = json.loads(backup_meta_path.read_text(encoding="utf-8"))
    image = LOCAL / "root-patched.squashfs"
    size = int(patch_meta["size"])

    if not args.i_accept_brick_risk:
        print(
            f"DRY RUN: would verify boot_part={backup_meta['boot_part']}, "
            f"write {image.name} ({size} bytes) to {patch_meta['target_system']}, "
            f"read back {size} bytes to verify SHA-256, then reset."
        )
        print("Pass `--i-accept-brick-risk` after capturing WorldCup mode with usb_connect.py.")
        return 0

    update_bin = LOCAL / "update"
    if sha256_file(update_bin) != json.loads((LOCAL / "tool.json").read_text())["sha256"]:
        print("STOP: .local/update SHA-256 mismatch.")
        return 1

    if sha256_file(image) != patch_meta["sha256"] or image.stat().st_size != size:
        print("STOP: patched image hash/size mismatch.")
        return 1

    for name, expected in backup_meta["files"].items():
        if sha256_file(LOCAL / "backup" / name) != expected:
            print(f"STOP: backup file {name} integrity check failed.")
            return 1

    port = json.loads((LOCAL / "usb.json").read_text())["port"]
    active_boot = detect_boot_part(update_bin, port)
    if active_boot != backup_meta["boot_part"]:
        print(f"STOP: boot_part changed ({active_boot} != {backup_meta['boot_part']}); refusing to write.")
        return 1

    target_system = SYSTEM_FOR_BOOT[active_boot]
    if target_system != patch_meta["target_system"]:
        print(f"STOP: target_system mismatch ({target_system} != {patch_meta['target_system']}).")
        return 1

    print(f"Writing {image.name} ({size} bytes) to {target_system}...", flush=True)
    run_update(update_bin, port, "partition", target_system, image)

    readback = LOCAL / "flash-readback.bin"
    if readback.exists():
        readback.unlink()
    print(f"Reading back {size} bytes from {target_system}...", flush=True)
    read_partition(update_bin, port, target_system, size, readback)
    if sha256_file(readback) != patch_meta["sha256"]:
        print("STOP: READBACK SHA-256 MISMATCH! Do NOT reset or unplug blindly.")
        return 1

    print("Readback SHA-256 matches 100%. Resetting speaker to normal mode...", flush=True)
    reset_to_normal(update_bin, port)
    print("OK: flash verified and speaker booted cleanly!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

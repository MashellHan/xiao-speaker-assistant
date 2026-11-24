#!/usr/bin/env python3
"""Offline helper: slice exact SquashFS rootfs from active backup and unpack for inspection."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".local"

SYSTEM_FOR_BOOT = {"boot0": "system0", "boot1": "system1"}
COMPRESSION_NAMES = {1: "gzip", 2: "lzma", 3: "lzo", 4: "xz", 5: "lz4", 6: "zstd"}


def parse_squashfs_superblock(header: bytes) -> dict[str, int | str]:
    if len(header) < 96:
        raise RuntimeError("Header shorter than 96 bytes")
    if header[:4] != b"hsqs":
        raise RuntimeError(f"Not a little-endian SquashFS image (magic={header[:4]!r})")
    block_size = struct.unpack_from("<I", header, 12)[0]
    comp_id = struct.unpack_from("<H", header, 20)[0]
    bytes_used = struct.unpack_from("<Q", header, 40)[0]
    if bytes_used <= 96 or bytes_used > 0x02800000:
        raise RuntimeError(f"Unexpected SquashFS bytes_used={bytes_used}")
    return {
        "compression": COMPRESSION_NAMES.get(comp_id, f"unknown({comp_id})"),
        "block_size": block_size,
        "bytes_used": bytes_used,
    }


def main() -> int:
    os.umask(0o077)
    boot_meta = LOCAL / "boot-part.json"
    if not boot_meta.is_file():
        print("STOP: .local/boot-part.json missing; run backup_partitions.py first.")
        return 1

    active_boot = json.loads(boot_meta.read_text())["boot_part"]
    active_sys = SYSTEM_FOR_BOOT[active_boot]
    raw_sys = LOCAL / "backup" / f"{active_sys}.1.bin"
    if not raw_sys.is_file():
        print(f"STOP: missing {raw_sys}")
        return 1

    data = raw_sys.read_bytes()
    sb = parse_squashfs_superblock(data[:96])
    size = int(sb["bytes_used"])

    extracted_dir = LOCAL / "extracted"
    extracted_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(extracted_dir, 0o700)
    squashfs_path = extracted_dir / "root.squashfs"
    squashfs_path.write_bytes(data[:size])
    squashfs_path.chmod(0o600)

    info_path = extracted_dir / "squashfs-meta.json"
    info_path.write_text(
        json.dumps({"active_boot": active_boot, "active_system": active_sys, **sb}, indent=2),
        encoding="utf-8",
    )
    info_path.chmod(0o600)

    unsquashfs = shutil.which("unsquashfs")
    if not unsquashfs:
        print("STOP: `unsquashfs` not found. Install with: brew install squashfs")
        return 1

    tree = LOCAL / "rootfs-stock"
    if tree.exists():
        print(f"INFO: {tree} already exists; skipping unsquashfs.")
    else:
        exclude = LOCAL / "exclude.txt"
        exclude.write_text("dev/console\n", encoding="utf-8")
        exclude.chmod(0o600)
        res = subprocess.run(
            [unsquashfs, "-exclude-file", str(exclude), "-d", str(tree), str(squashfs_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            print(f"STOP: unsquashfs failed:\n{res.stdout}")
            return 1

    version_file = tree / "usr/share/mico/version"
    print(f"Active slot:  {active_boot} -> {active_sys}")
    print(f"SquashFS:     comp={sb['compression']} block_size={sb['block_size']} bytes_used={size}")
    if version_file.is_file():
        print("--- usr/share/mico/version ---")
        print(version_file.read_text(encoding="utf-8").strip())
    for rel in ("etc/config/dropbear", "etc/rc.local"):
        target = tree / rel
        if target.is_file():
            print(f"--- {rel} ---")
            print(target.read_text(encoding="utf-8", errors="replace").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())

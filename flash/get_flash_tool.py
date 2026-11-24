#!/usr/bin/env python3
"""Download and prepare the pinned Khadas Amlogic `update` binary for arm64 macOS."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".local"

TOOL_URL = (
    "https://raw.githubusercontent.com/khadas/utils/"
    "0dee958a4c3c53c0546ef98758ed721183f11a5f/aml-flash-tool/tools/macos/update"
)
TOOL_SHA256 = "19f37cc129d061512c24bffbbad3a51355d976c12d6091366fe7dfce3c949b30"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str | Path]) -> str:
    res = subprocess.run(
        [str(x) for x in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {cmd[0]}\n{res.stdout}")
    return res.stdout


def check_dylibs(binary: Path, libusb_path: Path) -> None:
    pending = [binary, libusb_path]
    seen: set[Path] = set()
    while pending:
        item = pending.pop()
        if not item.is_file():
            raise RuntimeError(f"Missing library: {item}")
        if item in seen:
            continue
        seen.add(item)
        own_ids = {s.strip() for s in run_cmd(["otool", "-D", item]).splitlines()[1:]}
        for line in run_cmd(["otool", "-L", item]).splitlines()[1:]:
            dep = line.strip().split(" (compatibility version", 1)[0]
            if dep in own_ids or dep.startswith(("/usr/lib/", "/System/")):
                continue
            if dep.startswith("@loader_path/"):
                dep = str(item.parent / dep[len("@loader_path/") :])
            target = Path(os.path.abspath(dep))
            if not target.is_file():
                raise RuntimeError(f"Missing dependency library: {target}")
            pending.append(target)
    ctypes.CDLL(str(libusb_path))


def main() -> int:
    os.umask(0o077)
    LOCAL.mkdir(mode=0o700, exist_ok=True)
    os.chmod(LOCAL, 0o700)

    brew = shutil.which("brew")
    if not brew:
        print("ERROR: Homebrew not found.")
        return 1

    prefix = run_cmd([brew, "--prefix", "libusb-compat"]).strip()
    libusb = Path(prefix) / "lib" / "libusb-0.1.4.dylib"
    if not libusb.is_file():
        print(f"ERROR: {libusb} not found. Run: brew install libusb-compat")
        return 1

    vendor = LOCAL / "update.vendor"
    if not vendor.exists():
        print("Downloading pinned Khadas macOS `update` binary...")
        run_cmd(["curl", "--fail", "--location", "--silent", "--show-error", TOOL_URL, "--output", vendor])
    vendor.chmod(0o600)

    digest = sha256_file(vendor)
    if digest != TOOL_SHA256:
        raise RuntimeError(f"Downloaded update SHA-256 mismatch: {digest}")

    out = LOCAL / "update"
    run_cmd(["lipo", "-thin", "arm64", vendor, "-output", out])
    run_cmd(
        [
            "install_name_tool",
            "-change",
            "/opt/homebrew/opt/libusb-compat/lib/libusb-0.1.4.dylib",
            str(libusb),
            str(out),
        ]
    )
    run_cmd(["codesign", "--force", "--sign", "-", str(out)])
    out.chmod(0o700)

    check_dylibs(out, libusb)
    meta = LOCAL / "tool.json"
    meta.write_text(json.dumps({"sha256": sha256_file(out)}, indent=2), encoding="utf-8")
    meta.chmod(0o600)
    print(f"OK: flash tool ready at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

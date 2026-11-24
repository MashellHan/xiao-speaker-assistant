#!/usr/bin/env python3
"""Read-only environment check for OH2P flashing and Mac host runtime."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def has_brew_formula(name: str) -> bool:
    brew = shutil.which("brew")
    if not brew:
        return False
    res = subprocess.run(
        [brew, "list", "--formula", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return res.returncode == 0


def main() -> int:
    arch = platform.machine()
    print(f"macOS arch: {arch}")
    if arch != "arm64":
        print("WARN: expected Apple Silicon (arm64); verify flash tool binary architecture before flashing.")

    missing_formulae: list[str] = []
    for formula, cmd in (("uv", "uv"), ("squashfs", "unsquashfs"), ("libusb-compat", None)):
        ok = bool(shutil.which(cmd)) if cmd else has_brew_formula(formula)
        status = "OK" if ok else "MISSING"
        print(f"  [{status}] {formula}")
        if not ok:
            missing_formulae.append(formula)

    ssh_dir = Path("~/.ssh").expanduser()
    print(f"  [INFO] ~/.ssh exists: {ssh_dir.exists()}")

    if missing_formulae:
        print("\nInstall missing dependencies with:")
        print(f"  brew install {' '.join(missing_formulae)}")
        return 1

    print("\nOK: Mac environment ready for next step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

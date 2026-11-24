#!/usr/bin/env python3
"""Read-only USB probe for Xiaomi Smart Speaker Pro (OH2P) via macOS ioreg."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

NORMAL_VID_PID = ("0x2717", "0xd002")
WORLDCUP_VID_PID = ("0x1b8e", "0xc003")


@dataclass(frozen=True)
class UsbDevice:
    name: str
    vendor_id: str
    product_id: str
    location_id: str

    @property
    def mode(self) -> str:
        pair = (self.vendor_id, self.product_id)
        if pair == NORMAL_VID_PID:
            return "normal-oh2p"
        if pair == WORLDCUP_VID_PID:
            return "amlogic-worldcup-flash"
        if self.vendor_id == "0x1b8e":
            return "amlogic-other"
        return "other"


def parse_ioreg_usb(output: str) -> list[UsbDevice]:
    devices: list[UsbDevice] = []
    blocks = re.split(r"\n(?=\s*\+-o )", output)
    for block in blocks:
        vid_m = re.search(r'"idVendor"\s*=\s*(\d+)', block)
        pid_m = re.search(r'"idProduct"\s*=\s*(\d+)', block)
        if not (vid_m and pid_m):
            continue
        loc_m = re.search(r'"locationID"\s*=\s*(\d+)', block)
        name_m = re.search(r'"USB Product Name"\s*=\s*"([^"]+)"', block)
        vid = f"0x{int(vid_m.group(1)):04x}"
        pid = f"0x{int(pid_m.group(1)):04x}"
        loc = f"0x{int(loc_m.group(1)):08x}" if loc_m else "unknown"
        name = name_m.group(1) if name_m else "USB Device"
        devices.append(UsbDevice(name=name, vendor_id=vid, product_id=pid, location_id=loc))
    return devices


def main() -> int:
    res = subprocess.run(
        ["/usr/sbin/ioreg", "-p", "IOUSB", "-w0", "-l", "-r", "-c", "IOUSBHostDevice"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if res.returncode != 0:
        print(f"ERROR: ioreg failed ({res.returncode}): {res.stderr.strip()}")
        return 1

    relevant = [d for d in parse_ioreg_usb(res.stdout) if d.mode != "other"]
    if not relevant:
        print("No Xiaomi OH2P / Amlogic USB device detected.")
        print("Checklist:")
        print("  1. Speaker is powered with its 12V adapter.")
        print("  2. Type-C cable supports data transfer (not charge-only).")
        print("  3. If macOS prompted 'Allow accessory to connect', click Allow.")
        return 0

    for d in relevant:
        print(f"[{d.mode}] {d.name} vid={d.vendor_id} pid={d.product_id} location={d.location_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

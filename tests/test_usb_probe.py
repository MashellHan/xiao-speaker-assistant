import importlib.util
from pathlib import Path

PROBE_PATH = Path(__file__).resolve().parent.parent / "flash" / "usb_probe.py"
spec = importlib.util.spec_from_file_location("usb_probe", PROBE_PATH)
assert spec and spec.loader
usb_probe = importlib.util.module_from_spec(spec)
import sys
sys.modules["usb_probe"] = usb_probe
spec.loader.exec_module(usb_probe)


SAMPLE_IOREG = """
+-o AppleUSBHostCompositeDevice@01100000
  | "idVendor" = 10007
  | "idProduct" = 53250
  | "locationID" = 17825792
  | "USB Product Name" = "OH2P"
+-o WorldCup Device@01100000
  | "idVendor" = 7054
  | "idProduct" = 49155
  | "locationID" = 17825792
  | "USB Product Name" = "GX-CHIP"
+-o Keyboard@02100000
  | "idVendor" = 1452
  | "idProduct" = 610
  | "locationID" = 34603008
  | "USB Product Name" = "Apple Keyboard"
"""


def test_parse_ioreg_usb_detects_normal_and_worldcup() -> None:
    devices = usb_probe.parse_ioreg_usb(SAMPLE_IOREG)
    modes = [d.mode for d in devices]
    assert modes == ["normal-oh2p", "amlogic-worldcup-flash", "other"]

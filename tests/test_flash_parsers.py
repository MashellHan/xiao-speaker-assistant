import importlib.util
import struct
import sys
from pathlib import Path

FLASH_DIR = Path(__file__).resolve().parent.parent / "flash"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, FLASH_DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_squashfs_superblock_reads_xz_bytes_used() -> None:
    if str(FLASH_DIR) not in sys.path:
        sys.path.insert(0, str(FLASH_DIR))
    mod = _load("extract_active_rootfs")
    buf = bytearray(96)
    buf[0:4] = b"hsqs"
    struct.pack_into("<I", buf, 12, 131072)
    struct.pack_into("<H", buf, 20, 4)  # xz
    struct.pack_into("<Q", buf, 40, 28_000_000)
    sb = mod.parse_squashfs_superblock(bytes(buf))
    assert sb["compression"] == "xz"
    assert sb["block_size"] == 131072
    assert sb["bytes_used"] == 28_000_000


def test_android_boot_size_matches_page_alignment() -> None:
    if str(FLASH_DIR) not in sys.path:
        sys.path.insert(0, str(FLASH_DIR))
    _load("usb_probe")
    _load("usb_connect")
    mod = _load("backup_partitions")
    buf = bytearray(4096)
    buf[0:8] = b"ANDROID!"
    # ks=4097 (3 pages), rs=2048 (1 page), ss=0, dt=2049 (2 pages), pg=2048
    struct.pack_into("<10I", buf, 8, 4097, 0, 2048, 0, 0, 0, 0, 2048, 2049, 0)
    size = mod.android_boot_size(bytes(buf))
    assert size == 2048 + (3 + 1 + 0 + 2) * 2048 + 512

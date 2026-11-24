#!/usr/bin/env python3
"""Offline minimal patch builder for OH2P stock rootfs (key-only SSH + /data boot hook)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".local"

RC_LOCAL = """# Start Dropbear with key-only auth after S95done runs rc.local.
umask 077
if [ -s /etc/dropbear/authorized_keys ]; then
    printf '1\\n' > /tmp/ssh_en
    /etc/init.d/dropbear start
fi

# Launch user agent client from persistent /data if installed.
if [ -d /data/xiaoai-agent ] && [ ! -L /data/xiaoai-agent ] && \\
   [ -x /data/xiaoai-agent/boot.sh ] && [ ! -L /data/xiaoai-agent/boot.sh ]; then
    /bin/sh /data/xiaoai-agent/boot.sh >/dev/null 2>&1 &
fi

exit 0
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_fixed(path: Path, data: bytes, mode: int, epoch: int) -> None:
    path.write_bytes(data)
    path.chmod(mode)
    os.utime(path, (epoch, epoch))


def ensure_ssh_key(key_path: Path) -> Path:
    key = key_path.expanduser()
    key.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not key.exists():
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "rsa", "-b", "4096", "-N", "", "-f", str(key)],
            check=True,
        )
    key.chmod(0o600)
    pub_orig = Path(str(key) + ".pub")
    if pub_orig.exists():
        pub_orig.chmod(0o600)
    pub_out = LOCAL / "patch.pub"
    pub_bytes = subprocess.check_output(["ssh-keygen", "-y", "-f", str(key)])
    pub_out.write_bytes(pub_bytes)
    pub_out.chmod(0o600)
    fp = subprocess.check_output(["ssh-keygen", "-lf", str(pub_out)], text=True)
    if not fp.startswith("4096 "):
        raise RuntimeError(f"Expected RSA-4096 key, got: {fp.strip()}")
    return pub_out


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key",
        type=Path,
        default=Path("~/.ssh/xiaoai_oh2p_rsa"),
        help="Private key path (generated if missing; only public key enters rootfs)",
    )
    parser.add_argument("--rebuild", action="store_true", help="Replace existing .local/patched-tree and image")
    args = parser.parse_args()

    stock = LOCAL / "rootfs-stock"
    meta_path = LOCAL / "extracted" / "squashfs-meta.json"
    if not stock.is_dir() or not meta_path.is_file():
        print("STOP: run `python3 flash/extract_active_rootfs.py` first.")
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    comp = str(meta["compression"])
    block_size = int(meta["block_size"])
    if comp not in {"xz", "gzip", "lzo", "lz4", "zstd"}:
        raise RuntimeError(f"Unsupported SquashFS compression: {comp}")

    version_txt = (stock / "usr/share/mico/version").read_text(encoding="utf-8")
    if "OH2P" not in version_txt:
        raise RuntimeError("Stock rootfs is not OH2P")

    mksquashfs = shutil.which("mksquashfs")
    if not mksquashfs:
        print("STOP: `mksquashfs` not found. Run: brew install squashfs")
        return 1

    pub_path = ensure_ssh_key(args.key)
    public_key = pub_path.read_bytes()
    if not public_key.startswith(b"ssh-rsa "):
        raise RuntimeError("Public key does not start with ssh-rsa")

    tree = LOCAL / "patched-tree"
    image = LOCAL / "root-patched.squashfs"
    if args.rebuild:
        if tree.exists():
            shutil.rmtree(tree)
        if image.exists():
            image.unlink()
    elif tree.exists() or image.exists():
        print("STOP: .local/patched-tree or .local/root-patched.squashfs already exists; pass --rebuild.")
        return 1

    shutil.copytree(stock, tree, symlinks=True, copy_function=shutil.copy2)

    epoch = 1770285320
    dropbear_cfg = tree / "etc/config/dropbear"
    cfg_text = dropbear_cfg.read_text(encoding="utf-8")
    for option in ("PasswordAuth", "RootPasswordAuth"):
        old = f"option {option} '1'"
        if cfg_text.count(old) != 1:
            raise RuntimeError(f"Expected `{old}` exactly once in etc/config/dropbear")
        cfg_text = cfg_text.replace(old, f"option {option} '0'")

    write_fixed(dropbear_cfg, cfg_text.encode("utf-8"), 0o644, epoch)
    write_fixed(tree / "etc/rc.local", RC_LOCAL.encode("utf-8"), 0o644, epoch)
    write_fixed(tree / "etc/dropbear/authorized_keys", public_key, 0o600, epoch)

    root_ssh = tree / "root/.ssh"
    root_ssh.mkdir(parents=True, exist_ok=True, mode=0o700)
    root_ssh.chmod(0o700)
    write_fixed(root_ssh / "authorized_keys", public_key, 0o600, epoch)
    os.utime(root_ssh, (epoch, epoch))
    for rel in ("etc", "etc/config", "etc/dropbear", "root"):
        os.utime(tree / rel, (epoch, epoch))

    cmd = [
        mksquashfs,
        str(tree),
        str(image),
        "-comp",
        comp,
        "-b",
        str(block_size),
        "-all-root",
        "-no-xattrs",
        "-repro-time",
        str(epoch),
        "-noappend",
        "-processors",
        "2",
        "-p",
        "dev/console c 600 0 0 5 1",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"mksquashfs failed ({res.returncode}):\n{res.stdout[-2000:]}")

    image.chmod(0o600)
    size = image.stat().st_size
    if not (0 < size < 0x02800000):
        raise RuntimeError(f"Patched image size {size} out of bounds (< 41943040)")

    digest = sha256_file(image)
    patch_meta = LOCAL / "patch.json"
    patch_meta.write_text(
        json.dumps(
            {
                "active_boot": meta["active_boot"],
                "target_system": meta["active_system"],
                "size": size,
                "sha256": digest,
                "ssh_key": str(args.key.expanduser()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    patch_meta.chmod(0o600)
    print(f"OK: built {image.name} ({size} bytes, {size / 1048576:.2f} MiB / 40.00 MiB max)")
    print(f"    target_system={meta['active_system']} sha256={digest[:16]}...")
    print(f"    ssh_key={args.key.expanduser()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Safe read-only macOS local file helpers scoped to user directories."""

from __future__ import annotations

import codecs
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from xiao_speaker_assistant.tools.browser import truncate_text

DEFAULT_ALLOWED_ROOTS: tuple[str, ...] = (
    "~/Desktop",
    "~/Downloads",
    "~/Documents",
)

_SENSITIVE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".key",
        ".pem",
        ".p12",
        ".pfx",
        ".ppk",
        ".kdbx",
        ".keychain",
        ".keychain-db",
        ".binarycookies",
    }
)

_SENSITIVE_NAME_TOKENS: tuple[str, ...] = (
    "token_cache",
    "keychain",
    "login.keychain",
    "id_rsa",
    "id_ed25519",
    "id_dsa",
    "id_ecdsa",
    "authorized_keys",
    "known_hosts",
    "client_secret",
    "cookies.binarycookies",
    "cookies.sqlite",
    "login data",
    "logins.json",
    "key4.db",
    "key3.db",
)

_SENSITIVE_PATH_SEGMENTS: tuple[str, ...] = (
    "/.ssh",
    "/.gnupg",
    "/.aws",
    "/.kube",
    "/library/keychains",
    "login.keychain-db",
    "login.keychain",
    "/microsoft edge/default/cookies",
    "/microsoft edge/default/login data",
    "/google/chrome/default/cookies",
    "/google/chrome/default/login data",
)


class MacLocalAccessError(PermissionError, ValueError):
    """Raised when a path escapes allowed roots or targets sensitive credentials/keychains."""


def resolve_allowed_roots(allowed_roots: Sequence[str | Path] | None = None) -> tuple[Path, ...]:
    raw_roots = DEFAULT_ALLOWED_ROOTS if allowed_roots is None else allowed_roots
    resolved = tuple(Path(root).expanduser().resolve(strict=False) for root in raw_roots)
    if not resolved:
        raise ValueError("At least one allowed root directory must be configured.")
    return resolved


def is_sensitive_path(path: Path, *, root: Path | None = None) -> bool:
    """Return True if path is hidden, a key/cert/token cache, browser cookie/login DB, or keychain."""
    name = path.name
    if name.startswith("."):
        return True

    if root is not None and path.is_relative_to(root):
        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts):
            return True
        rel_lower = "/".join(rel_parts).lower()
    else:
        rel_lower = name.lower()

    if any(suffix.lower() in _SENSITIVE_EXTENSIONS for suffix in path.suffixes):
        return True

    lower_name = name.lower()
    if any(token in lower_name or token in rel_lower for token in _SENSITIVE_NAME_TOKENS):
        return True

    posix_lower = path.as_posix().lower()
    if any(segment in posix_lower for segment in _SENSITIVE_PATH_SEGMENTS):
        return True

    return False


def _resolve_scoped_path(
    raw_path: str | Path,
    allowed_roots: Sequence[str | Path] | None = None,
) -> tuple[Path, Path, tuple[Path, ...]]:
    if not isinstance(raw_path, (str, Path)):
        raise MacLocalAccessError("Path must be a string or Path.")
    text = str(raw_path).strip()
    if not text or "\x00" in text:
        raise MacLocalAccessError("A valid non-empty path is required.")

    roots = resolve_allowed_roots(allowed_roots)
    expanded = Path(text).expanduser()
    resolved = expanded.resolve(strict=False)

    matching_root = next((root for root in roots if resolved.is_relative_to(root)), None)
    if matching_root is None:
        raise MacLocalAccessError(f"Path is outside allowed user directories: {raw_path}")

    if is_sensitive_path(resolved, root=matching_root):
        raise MacLocalAccessError(f"Access to sensitive or hidden path is blocked: {resolved.name}")

    # Also block symlinks whose own name/relative location inside an allowed root is sensitive.
    unfollowed_abs = Path(expanded if expanded.is_absolute() else Path.cwd() / expanded)
    for root in roots:
        if unfollowed_abs.is_relative_to(root) and is_sensitive_path(unfollowed_abs, root=root):
            raise MacLocalAccessError(
                f"Access to sensitive or hidden path is blocked: {unfollowed_abs.name}"
            )

    return resolved, matching_root, roots


def list_recent_files(
    directory: str = "~/Desktop",
    limit: int = 10,
    *,
    allowed_roots: Sequence[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """List recent non-sensitive regular files inside an allowed macOS user directory."""
    target_dir, matching_root, roots = _resolve_scoped_path(directory, allowed_roots)
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {target_dir}")

    clamped_limit = max(1, min(100, int(limit)))
    entries: list[dict[str, Any]] = []

    for entry in target_dir.iterdir():
        if is_sensitive_path(entry, root=matching_root):
            continue
        resolved_entry = entry.resolve(strict=False)
        entry_root = next((root for root in roots if resolved_entry.is_relative_to(root)), None)
        if entry_root is None:
            continue
        if is_sensitive_path(resolved_entry, root=entry_root):
            continue
        if not resolved_entry.is_file():
            continue
        stat = resolved_entry.stat()
        entries.append(
            {
                "name": entry.name,
                "path": str(resolved_entry),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )

    entries.sort(key=lambda item: (-item["mtime"], item["name"]))
    return entries[:clamped_limit]


def _decode_bounded_utf8(raw_bytes: bytes, *, hit_byte_cap: bool) -> str:
    if b"\x00" in raw_bytes:
        raise ValueError("Binary files cannot be read as text.")
    if not hit_byte_cap:
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File is not valid UTF-8 text: {exc}") from exc

    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        return decoder.decode(raw_bytes, final=False)
    except UnicodeDecodeError as exc:
        raise ValueError(f"File is not valid UTF-8 text: {exc}") from exc


def read_local_text_file(
    path: str,
    max_chars: int = 3000,
    *,
    allowed_roots: Sequence[str | Path] | None = None,
) -> str:
    """Read a UTF-8 text file inside allowed roots, blocking traversal and sensitive files."""
    target_file, _root, _roots = _resolve_scoped_path(path, allowed_roots)
    if not target_file.exists():
        raise FileNotFoundError(f"File does not exist: {target_file}")
    if not target_file.is_file():
        raise IsADirectoryError(f"Path is not a regular file: {target_file}")

    limit = max(0, int(max_chars))
    if limit == 0:
        return ""

    read_cap = max(65536, limit * 4 + 16)
    with target_file.open("rb") as fh:
        raw_bytes = fh.read(read_cap + 1)

    hit_byte_cap = len(raw_bytes) > read_cap
    if hit_byte_cap:
        raw_bytes = raw_bytes[:read_cap]

    text = _decode_bounded_utf8(raw_bytes, hit_byte_cap=hit_byte_cap)
    return truncate_text(text, limit)

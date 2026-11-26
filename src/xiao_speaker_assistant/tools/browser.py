"""Safe Microsoft Edge desktop launcher and HTTP webpage text extractor."""

from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
import subprocess
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx

EDGE_APP_NAME = "Microsoft Edge"

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_SHELL_META_CHARS = frozenset(";|`$<>(){}\\'\"\n\r\t ")
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9.\-:\[\]]+$")
_BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})

_NOISE_BLOCK_RE = re.compile(
    r"<(script|style|noscript|svg|title)\b[^>]*>[\s\S]*?</\1>|<!--[\s\S]*?-->",
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"<title\b[^>]*>([\s\S]*?)</title>", re.IGNORECASE)
_BLOCK_TAG_RE = re.compile(
    r"</?(?:p|div|br|li|ul|ol|h[1-6]|tr|td|th|section|article|header|footer|blockquote|pre)\b[^>]*>",
    re.IGNORECASE,
)
_ANY_TAG_RE = re.compile(r"<[^>]+>")


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def extract_html_summary(raw_html: str, max_chars: int = 2000) -> tuple[str | None, str]:
    """Strip scripts/styles/tags from HTML and return (title, clean_text_excerpt)."""
    title: str | None = None
    title_match = _TITLE_RE.search(raw_html)
    if title_match:
        raw_title = _ANY_TAG_RE.sub(" ", title_match.group(1))
        cleaned_title = re.sub(r"\s+", " ", html.unescape(raw_title)).strip()
        if cleaned_title:
            title = cleaned_title

    cleaned = _NOISE_BLOCK_RE.sub(" ", raw_html)
    cleaned = _BLOCK_TAG_RE.sub(" ", cleaned)
    cleaned = _ANY_TAG_RE.sub("", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return title, truncate_text(cleaned, max_chars)


def _is_non_public_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip_obj, ipaddress.IPv6Address):
        if ip_obj.ipv4_mapped is not None:
            return _is_non_public_ip(ip_obj.ipv4_mapped)
        if ip_obj.sixtofour is not None and _is_non_public_ip(ip_obj.sixtofour):
            return True
        if ip_obj.teredo is not None:
            server_ip, client_ip = ip_obj.teredo
            if _is_non_public_ip(server_ip) or _is_non_public_ip(client_ip):
                return True
    return (
        not ip_obj.is_global
        or ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    )


def _parse_ip_literal(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse canonical IPv4/IPv6 AND alternate IPv4 notations (octal/hex/decimal/short)."""
    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        pass
    # Catch non-canonical IPv4 literals (e.g. 2130706433, 0x7f000001, 127.1, 0177.0.0.1)
    # that Chromium/Edge/WHATWG URL parsers normalize to loopback/private IPs.
    try:
        packed = socket.inet_aton(hostname)
    except OSError:
        return None
    return ipaddress.IPv4Address(packed)


def _check_public_host(hostname: str, *, resolve_dns: bool = False) -> None:
    lower = hostname.strip().rstrip(".").lower()
    if lower in _BLOCKED_HOSTNAMES or lower.endswith((".localhost", ".local", ".internal")):
        raise ValueError(f"Access to local/internal host is blocked: {hostname}")
    ip_obj = _parse_ip_literal(lower)
    if ip_obj is not None:
        # Reject any non-public IP, or ambiguous non-canonical IPv4 literal (octal/hex/decimal/short).
        if _is_non_public_ip(ip_obj):
            raise ValueError(f"Access to loopback/private/link-local IP is blocked: {hostname}")
        if isinstance(ip_obj, ipaddress.IPv4Address) and str(ip_obj) != lower:
            raise ValueError(f"Non-canonical IPv4 literal is blocked: {hostname}")
        return
    if resolve_dns:
        try:
            infos = socket.getaddrinfo(lower, None, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError(f"Failed to resolve host {hostname}: {exc}") from exc
        if not infos:
            raise ValueError(f"Host {hostname} did not resolve to any IP address.")
        for _fam, _type, _proto, _canon, sockaddr in infos:
            addr = sockaddr[0]
            try:
                resolved_ip = ipaddress.ip_address(addr)
            except ValueError:
                continue
            if _is_non_public_ip(resolved_ip):
                raise ValueError(
                    f"Host {hostname} resolves to blocked loopback/private IP ({addr})."
                )


def validate_http_url(
    url: str,
    *,
    block_private_network: bool = False,
    resolve_dns: bool = False,
) -> str:
    """Validate that url is an http:// or https:// URL free of shell metacharacters."""
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("A non-empty http:// or https:// URL is required.")
    if any(ord(ch) < 32 or ord(ch) == 127 or ch in _SHELL_META_CHARS for ch in cleaned):
        raise ValueError("URL contains disallowed whitespace, control characters, or shell metacharacters.")
    if "&&" in cleaned or "||" in cleaned:
        raise ValueError("URL contains disallowed shell command chaining operators.")

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES or not cleaned.lower().startswith(f"{scheme}://"):
        raise ValueError("Only http:// and https:// URLs are allowed.")
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("URL must include a valid host.")
    if not _HOSTNAME_RE.match(parsed.hostname):
        raise ValueError("URL hostname contains disallowed characters.")
    # Accessing parsed.port validates that any port number is numeric and in 0..65535.
    _ = parsed.port
    if block_private_network:
        _check_public_host(parsed.hostname, resolve_dns=resolve_dns)
    return cleaned


def build_edge_open_command(url: str) -> list[str]:
    """Build the macOS `open -a \"Microsoft Edge\" <url>` argv list without shell expansion."""
    validated_url = validate_http_url(url)
    return ["open", "-a", EDGE_APP_NAME, validated_url]


def open_in_edge(
    url: str,
    *,
    dry_run: bool = False,
    runner: Callable[[list[str]], Any] | None = None,
) -> str:
    """Validate an http/https URL and open it in Microsoft Edge, returning status JSON."""
    cmd = build_edge_open_command(url)
    validated_url = cmd[-1]
    if dry_run:
        return json.dumps(
            {
                "status": "dry_run",
                "browser": EDGE_APP_NAME,
                "url": validated_url,
                "command": cmd,
            },
            ensure_ascii=False,
        )

    if runner is not None:
        runner(cmd)
    else:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    return json.dumps(
        {
            "status": "opened",
            "browser": EDGE_APP_NAME,
            "url": validated_url,
            "command": cmd,
        },
        ensure_ascii=False,
    )


def _format_response_text(resp: httpx.Response, max_chars: int) -> tuple[str | None, str]:
    content_type = resp.headers.get("content-type", "").lower()
    body = resp.text
    if "html" in content_type or (not content_type and "<html" in body.lower()):
        return extract_html_summary(body, max_chars=max_chars)
    return None, truncate_text(re.sub(r"\s+", " ", body).strip(), max_chars)


def fetch_webpage_text(
    url: str,
    max_chars: int = 2000,
    *,
    client: httpx.Client | None = None,
) -> str:
    """Validate URL, fetch via httpx, and return clean readable text."""
    validated_url = validate_http_url(url)
    limit = max(0, int(max_chars))
    owns_client = client is None
    http = client or httpx.Client(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "xiao-speaker-assistant/0.1"},
    )
    try:
        resp = http.get(validated_url)
        resp.raise_for_status()
    finally:
        if owns_client:
            http.close()

    title, excerpt = _format_response_text(resp, max_chars=limit)
    if title and excerpt and not excerpt.startswith(title):
        return truncate_text(f"{title}\n{excerpt}", limit)
    return excerpt or (truncate_text(title, limit) if title else "")


async def async_fetch_webpage_summary(
    url: str,
    max_chars: int = 2000,
    *,
    client: httpx.AsyncClient | None = None,
    block_private_network: bool = True,
) -> dict[str, Any]:
    """Async helper returning structured summary metadata for MCP server tool calls."""
    validated_url = validate_http_url(
        url,
        block_private_network=block_private_network,
        resolve_dns=(block_private_network and client is None),
    )
    limit = max(0, int(max_chars))
    owns_client = client is None
    http = client or httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=False,
        headers={"User-Agent": "xiao-speaker-assistant/0.1"},
    )
    try:
        current_url = validated_url
        resp: httpx.Response | None = None
        for _ in range(5):
            resp = await http.get(current_url, follow_redirects=False)
            if resp.is_redirect and resp.headers.get("location"):
                next_url = str(resp.url.join(resp.headers["location"]))
                current_url = validate_http_url(
                    next_url,
                    block_private_network=block_private_network,
                    resolve_dns=(block_private_network and client is None),
                )
                continue
            break
        assert resp is not None
        if resp.is_redirect:
            raise ValueError("Too many redirects.")
        resp.raise_for_status()
    finally:
        if owns_client:
            await http.aclose()

    title, excerpt = _format_response_text(resp, max_chars=limit)
    return {
        "source": "http_fetch",
        "url": str(resp.url),
        "status_code": resp.status_code,
        "title": title,
        "excerpt": excerpt,
    }

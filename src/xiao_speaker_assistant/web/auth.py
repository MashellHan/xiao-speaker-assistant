"""Authentication, CSRF Origin validation, and rate limiting for Web Console & local MCP HTTP."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from urllib.parse import urlparse

from aiohttp import web

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
LOOPBACK_REMOTES = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})
PROXY_HEADER_NAMES = (
    "CF-Connecting-IP",
    "CF-Ray",
    "X-Forwarded-For",
    "X-Forwarded-Host",
    "Forwarded",
)
CONSOLE_COOKIE_NAME = "xsa_console_session"


def extract_hostname(host_header: str) -> str:
    val = host_header.strip().lower()
    if val.startswith("["):
        end = val.find("]")
        return val[1:end] if end != -1 else val
    return val.split(":", 1)[0]


class ConsoleAuthGuard:
    """Encapsulates loopback bypass, HMAC session cookies, CSRF origin check, and login throttling."""

    def __init__(
        self,
        *,
        console_password: str | None = None,
        mcp_auth_token: str | None = None,
    ) -> None:
        self.console_password = console_password
        self.mcp_auth_token = mcp_auth_token
        self._session_secret = hashlib.sha256(
            (console_password or mcp_auth_token or secrets.token_hex(32)).encode("utf-8")
        ).digest()
        self._login_failures: dict[str, list[float]] = {}

    @staticmethod
    def is_direct_local_request(request: web.Request) -> bool:
        remote = (request.remote or "").strip()
        if remote not in LOOPBACK_REMOTES:
            return False
        if any(h in request.headers for h in PROXY_HEADER_NAMES):
            return False
        host_header = request.headers.get("Host", "")
        if host_header and extract_hostname(host_header) not in LOOPBACK_HOSTS:
            return False
        return True

    def make_session_cookie(self, ttl_sec: int = 7 * 86400) -> str:
        exp = str(int(time.time()) + ttl_sec)
        sig = hmac.new(self._session_secret, exp.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{exp}.{sig}"

    def verify_session_cookie(self, raw_cookie: str) -> bool:
        if "." not in raw_cookie:
            return False
        exp_str, sig = raw_cookie.split(".", 1)
        if not exp_str.isdigit():
            return False
        if int(exp_str) < int(time.time()):
            return False
        expected = hmac.new(self._session_secret, exp_str.encode("ascii"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)

    def has_valid_console_auth(self, request: web.Request) -> bool:
        if self.is_direct_local_request(request):
            return True
        cookies = getattr(request, "cookies", {}) or {}
        cookie_val = cookies.get(CONSOLE_COOKIE_NAME, "")
        if cookie_val and self.verify_session_cookie(cookie_val):
            return True
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            supplied = auth_header[7:].strip()
            if self.console_password and hmac.compare_digest(supplied, self.console_password):
                return True
            if self.mcp_auth_token and hmac.compare_digest(supplied, self.mcp_auth_token):
                return True
        return False

    def check_mcp_security(self, request: web.Request) -> web.Response | None:
        if not self.is_direct_local_request(request):
            return web.json_response(
                {"error": "forbidden", "message": "MCP endpoint is local-only"},
                status=403,
            )
        if request.headers.get("Origin"):
            return web.json_response(
                {"error": "forbidden", "message": "Cross-origin requests are not allowed"},
                status=403,
            )
        if self.mcp_auth_token:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "unauthorized", "message": "Missing Bearer token"},
                    status=401,
                )
            supplied = auth_header[7:].strip()
            if not hmac.compare_digest(supplied, self.mcp_auth_token):
                return web.json_response(
                    {"error": "unauthorized", "message": "Invalid Bearer token"},
                    status=401,
                )
        return None

    @staticmethod
    def check_same_origin(request: web.Request, *, message: str = "Cross-origin requests are not allowed") -> web.Response | None:
        origin = request.headers.get("Origin")
        if origin:
            origin_host = (urlparse(origin).hostname or "").lower()
            req_host = extract_hostname(request.headers.get("Host", ""))
            allowed_hosts = set(LOOPBACK_HOSTS)
            if req_host:
                allowed_hosts.add(req_host)
            if origin_host not in allowed_hosts:
                return web.json_response(
                    {"error": "forbidden", "message": message},
                    status=403,
                )
        return None

    def check_console_read_security(self, request: web.Request) -> web.Response | None:
        if not self.has_valid_console_auth(request):
            return web.json_response(
                {"error": "unauthorized", "message": "Console login required"},
                status=401,
            )
        return None

    def check_console_write_security(self, request: web.Request) -> web.Response | None:
        origin_err = self.check_same_origin(request)
        if origin_err is not None:
            return origin_err
        if not self.has_valid_console_auth(request):
            return web.json_response(
                {"error": "unauthorized", "message": "Console login required"},
                status=401,
            )
        return None

    def _resolve_login_client_key(self, request: web.Request) -> str:
        return (
            request.headers.get("CF-Connecting-IP")
            or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or (request.remote or "unknown")
        )

    def _prune_login_failures(self, now: float) -> None:
        if len(self._login_failures) < 4096:
            return
        stale_keys = [
            k
            for k, ts_list in self._login_failures.items()
            if not ts_list or (now - ts_list[-1]) >= 60.0
        ]
        for k in stale_keys:
            self._login_failures.pop(k, None)
        if len(self._login_failures) >= 4096:
            overflow = len(self._login_failures) - 4096 + 1
            for k, _ in sorted(
                self._login_failures.items(), key=lambda kv: kv[1][-1] if kv[1] else 0.0
            )[:overflow]:
                self._login_failures.pop(k, None)

    def check_login_preconditions(self, request: web.Request) -> web.Response | None:
        """Run same-origin, password-configured, and rate-limit checks BEFORE reading JSON body."""
        origin_err = self.check_same_origin(request, message="Cross-origin login is not allowed")
        if origin_err is not None:
            return origin_err
        if not self.console_password:
            return web.json_response(
                {"error": "auth_unavailable", "message": "Console remote password is not configured"},
                status=503,
            )
        client_key = self._resolve_login_client_key(request)
        now = time.monotonic()
        self._prune_login_failures(now)
        recent = [t for t in self._login_failures.get(client_key, []) if (now - t) < 60.0]
        if not recent:
            self._login_failures.pop(client_key, None)
        if len(recent) >= 5:
            self._login_failures[client_key] = recent
            return web.json_response(
                {"error": "rate_limited", "message": "尝试次数过多，请 1 分钟后再试"},
                status=429,
            )
        return None

    def verify_and_record_login(self, request: web.Request, supplied_password: str) -> tuple[bool, web.Response | None]:
        """Verify supplied password and update sliding-window failure tracker."""
        pre_err = self.check_login_preconditions(request)
        if pre_err is not None:
            return False, pre_err
        assert self.console_password is not None
        client_key = self._resolve_login_client_key(request)
        now = time.monotonic()
        recent = [t for t in self._login_failures.get(client_key, []) if (now - t) < 60.0]
        if not supplied_password or not hmac.compare_digest(supplied_password, self.console_password):
            recent.append(now)
            self._login_failures[client_key] = recent
            return False, web.json_response(
                {"error": "invalid_password", "message": "访问口令错误"},
                status=401,
            )
        self._login_failures.pop(client_key, None)
        return True, None

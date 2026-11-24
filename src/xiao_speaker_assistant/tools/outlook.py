"""Outlook / Microsoft Graph integration using file-backed MSAL cache (no macOS Keychain)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import msal

from xiao_speaker_assistant.config import OutlookToolConfig, read_env_secret

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPES = ["User.Read", "Mail.Read", "Mail.Send"]


@dataclass
class PendingEmailDraft:
    to: str
    subject: str
    body: str
    confirmed: bool = False


class FileTokenCache:
    """Strictly file-backed MSAL SerializableTokenCache with 0600 permissions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.cache = msal.SerializableTokenCache()
        if self.path.exists():
            self.cache.deserialize(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        if not self.cache.has_state_changed:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, self.cache.serialize().encode("utf-8"))
        finally:
            os.close(fd)
        os.chmod(self.path, 0o600)


class OutlookClient:
    def __init__(self, config: OutlookToolConfig) -> None:
        self._config = config
        self._cache = FileTokenCache(config.token_cache_file)

    def _build_app(self) -> msal.PublicClientApplication:
        client_id = read_env_secret(self._config.client_id_env)
        if not client_id:
            raise RuntimeError(f"Missing Outlook client ID in env {self._config.client_id_env}")
        authority = f"https://login.microsoftonline.com/{self._config.tenant_id}"
        return msal.PublicClientApplication(
            client_id=client_id,
            authority=authority,
            token_cache=self._cache.cache,
        )

    def login_device_flow(self) -> dict[str, Any]:
        app = self._build_app()
        flow = app.initiate_device_flow(scopes=GRAPH_SCOPES)
        if "user_code" not in flow:
            raise RuntimeError(f"Failed to start device flow: {flow}")
        print(flow["message"], flush=True)
        result = app.acquire_token_by_device_flow(flow)
        self._cache.save()
        if "access_token" not in result:
            raise RuntimeError(result.get("error_description") or "Outlook login failed")
        return result

    def acquire_access_token(self) -> str:
        app = self._build_app()
        accounts = app.get_accounts()
        if not accounts:
            raise RuntimeError("Outlook not logged in; run `xsa outlook-login` first.")
        result = app.acquire_token_silent(GRAPH_SCOPES, account=accounts[0])
        self._cache.save()
        if not result or "access_token" not in result:
            raise RuntimeError("Outlook cached token expired; run `xsa outlook-login`.")
        return result["access_token"]

    async def list_recent_emails(self, limit: int = 5, unread_only: bool = False) -> list[dict[str, Any]]:
        token = self.acquire_access_token()
        params: dict[str, str] = {
            "$top": str(limit),
            "$select": "subject,from,receivedDateTime,bodyPreview,isRead",
            "$orderby": "receivedDateTime desc",
        }
        if unread_only:
            params["$filter"] = "isRead eq false"
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{GRAPH_BASE}/me/messages",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
            resp.raise_for_status()
            return resp.json().get("value", [])

    async def send_email(self, draft: PendingEmailDraft) -> str:
        if self._config.require_voice_confirmation_for_send and not draft.confirmed:
            return json.dumps(
                {
                    "status": "confirmation_required",
                    "message": (
                        f"即将发送邮件给 {draft.to}，主题是 {draft.subject}。"
                        f"请先语音确认后再发送。"
                    ),
                },
                ensure_ascii=False,
            )
        token = self.acquire_access_token()
        payload = {
            "message": {
                "subject": draft.subject,
                "body": {"contentType": "Text", "content": draft.body},
                "toRecipients": [{"emailAddress": {"address": draft.to}}],
            },
            "saveToSentItems": True,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{GRAPH_BASE}/me/sendMail",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
        return json.dumps({"status": "sent", "to": draft.to, "subject": draft.subject}, ensure_ascii=False)

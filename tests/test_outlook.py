import asyncio
import json
import stat

from xiao_speaker_assistant.config import OutlookToolConfig
from xiao_speaker_assistant.tools.outlook import FileTokenCache, OutlookClient, PendingEmailDraft


def test_file_token_cache_uses_strict_permissions(tmp_path) -> None:
    cache_path = tmp_path / "private" / "outlook_cache.json"
    ftc = FileTokenCache(cache_path)
    ftc.cache.has_state_changed = True
    ftc.save()

    assert cache_path.exists()
    assert stat.S_IMODE(cache_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600


def test_outlook_send_requires_voice_confirmation(tmp_path) -> None:
    cfg = OutlookToolConfig(
        token_cache_file=str(tmp_path / "cache.json"),
        require_voice_confirmation_for_send=True,
    )
    client = OutlookClient(cfg)
    result = asyncio.run(
        client.send_email(
            PendingEmailDraft(to="a@example.com", subject="测试", body="你好", confirmed=False)
        )
    )
    data = json.loads(result)
    assert data["status"] == "confirmation_required"

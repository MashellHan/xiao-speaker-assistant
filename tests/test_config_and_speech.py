from xiao_speaker_assistant.config import AppConfig, load_config
from xiao_speaker_assistant.llm.router import format_for_speech


def test_default_config_targets_oh2p_opencode_and_outlook() -> None:
    cfg = AppConfig()
    assert cfg.speaker.model == "OH2P"
    assert cfg.llm.default_provider == "opencode"
    assert cfg.tools.outlook.require_voice_confirmation_for_send is True


def test_load_config_ignores_example_and_uses_code_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.example.yaml").write_text(
        "llm:\n  default_provider: codex\n", encoding="utf-8"
    )
    cfg = load_config()
    assert cfg.llm.default_provider == "opencode"

    custom = tmp_path / "custom.yaml"
    custom.write_text("llm:\n  default_provider: codex\n", encoding="utf-8")
    cfg2 = load_config(custom)
    assert cfg2.llm.default_provider == "codex"


def test_format_for_speech_strips_markdown_and_urls() -> None:
    raw = "## 今日邮件\n请查看 [详情](https://example.com) 和 `代码`:\n```python\nprint(1)\n```"
    spoken = format_for_speech(raw, max_chars=50)
    assert "http" not in spoken
    assert "#" not in spoken
    assert "`" not in spoken
    assert "详情" in spoken

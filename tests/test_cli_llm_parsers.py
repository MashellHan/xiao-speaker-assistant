import json

from xiao_speaker_assistant.llm.router import parse_opencode_jsonl


def test_parse_opencode_jsonl_collects_text_parts() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "step_start", "sessionID": "ses_1"}),
            json.dumps({"type": "text", "part": {"type": "text", "text": "你好，"}}),
            json.dumps({"type": "text", "part": {"type": "text", "text": "我是 Nomi。"}}),
            json.dumps({"type": "step_finish", "reason": "stop"}),
        ]
    )
    assert parse_opencode_jsonl(stdout) == "你好，\n我是 Nomi。"

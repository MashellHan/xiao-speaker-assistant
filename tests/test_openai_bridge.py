import asyncio
import json
from unittest.mock import patch

import httpx

from xiao_speaker_assistant.config import SpeakerConfig
from xiao_speaker_assistant.llm.openai_bridge import OpenAIBridgeServer, extract_last_user_prompt


def test_extract_last_user_prompt_strips_bridge_rule_prompt() -> None:
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "今天天气\n注意：将结果处理成纯文字版，不要返回任何 markdown"},
    ]
    assert extract_last_user_prompt(msgs) == "今天天气"


def test_openai_bridge_chat_and_speech_endpoints() -> None:
    seen_prompts: list[str] = []
    resets: list[bool] = []

    async def fake_ask(prompt: str) -> str:
        seen_prompts.append(prompt)
        return f"收到：{prompt}"

    async def fake_stream(prompt: str):
        seen_prompts.append(prompt)
        yield "第一句。"
        yield "第二句。"

    async def fake_synth(text: str, *, voice: str, response_format: str):
        return b"ID3-fake-mp3-" + text.encode("utf-8"), "audio/mpeg"

    bridge = OpenAIBridgeServer(
        ask_fn=fake_ask,
        speaker_config=SpeakerConfig(),
        stream_fn=fake_stream,
        on_reset=lambda: resets.append(True),
        auth_token="bridge-tok",
    )

    async def run() -> None:
        runner, port = await bridge.start("127.0.0.1", 0)
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                unauth = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
                assert unauth.status_code == 401

                csrf = await client.post(
                    "/v1/chat/completions",
                    headers={"Origin": "https://evil.example", "Authorization": "Bearer bridge-tok"},
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
                assert csrf.status_code == 403

                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer bridge-tok", "X-Hermes-Session-Key": "s1"},
                    json={"model": "nomi", "messages": [{"role": "user", "content": "查一下磁盘"}]},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["choices"][0]["message"]["content"] == "收到：查一下磁盘"

                # Streaming SSE mode
                stream_resp = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer bridge-tok", "X-Hermes-Session-Key": "s1"},
                    json={"model": "nomi", "stream": True, "messages": [{"role": "user", "content": "流式测试"}]},
                )
                assert stream_resp.status_code == 200
                assert "text/event-stream" in stream_resp.headers["content-type"]
                assert "第一句。" in stream_resp.text and "第二句。" in stream_resp.text
                assert "data: [DONE]" in stream_resp.text

                # Switching session key triggers reset callback
                resp2 = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer bridge-tok", "X-Hermes-Session-Key": "s2"},
                    json={"model": "nomi", "messages": [{"role": "user", "content": "新会话"}]},
                )
                assert resp2.status_code == 200
                assert resets == [True]

                bad_fmt = await client.post(
                    "/v1/audio/speech",
                    headers={"Authorization": "Bearer bridge-tok"},
                    json={"input": "你好", "response_format": "exe"},
                )
                assert bad_fmt.status_code == 400

                with patch("xiao_speaker_assistant.llm.openai_bridge.synthesize_edge_audio_bytes", fake_synth):
                    tts_resp = await client.post(
                        "/v1/audio/speech",
                        headers={"Authorization": "Bearer bridge-tok"},
                        json={"input": "你好", "voice": "alloy", "response_format": "mp3"},
                    )
                assert tts_resp.status_code == 200
                assert tts_resp.content.startswith(b"ID3-fake-mp3-")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen_prompts == ["查一下磁盘", "流式测试", "新会话"]

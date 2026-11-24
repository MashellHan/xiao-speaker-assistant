import asyncio
import json

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.llm.router import SentenceStreamBuffer
from xiao_speaker_assistant.voice.ws_server import SpeakerGateway


def test_sentence_stream_buffer_emits_sentences_incrementally() -> None:
    buf = SentenceStreamBuffer(min_chars=8, max_total_chars=100)
    s1 = buf.feed("太阳系一共有八大行星。")
    assert s1 == ["太阳系一共有八大行星。"]
    s2 = buf.feed("地球排在第三位！")
    assert s2 == ["地球排在第三位！"]
    assert buf.feed("最后一点") == []
    assert buf.flush() == ["最后一点"]


def test_gateway_streaming_speaks_sentences_sequentially_via_rpc() -> None:
    sent_payloads: list[dict] = []

    class FakeConn:
        remote_address = ("127.0.0.1", 12345)

        async def send(self, raw: str) -> None:
            msg = json.loads(raw)
            sent_payloads.append(msg)
            req_id = msg["Request"]["id"]
            # Simulate open-xiaoai-client completing tts_play.sh immediately
            asyncio.get_running_loop().call_soon(
                lambda: asyncio.create_task(
                    gw._handle_text_message(
                        self,
                        json.dumps({"Response": {"id": req_id, "code": 0}}),
                    )
                )
            )

    async def fake_stream(query: str):
        yield "第一句话。"
        yield "第二句话。"

    gw = SpeakerGateway(AppConfig(), on_stream_query=fake_stream)

    async def run() -> None:
        conn = FakeConn()
        await gw._run_streaming_reply(conn, "测试")

    asyncio.run(run())
    assert len(sent_payloads) == 2
    assert "killall miplayer" in sent_payloads[0]["Request"]["payload"]
    assert "第一句话。" in sent_payloads[0]["Request"]["payload"]
    # Second sentence must NOT killall the first sentence since it waited for completion!
    assert "killall miplayer" not in sent_payloads[1]["Request"]["payload"]
    assert "第二句话。" in sent_payloads[1]["Request"]["payload"]

import asyncio
import base64
import io
import json
import struct
import wave
from email.parser import BytesParser
from email.policy import default as email_policy

import httpx
import numpy as np
import pytest

from xiao_speaker_assistant.voice.cloud_asr import (
    DOUBAO_FLASH_ENDPOINT,
    CloudASRConfigError,
    CloudASRError,
    DoubaoFlashASR,
    OpenAIWhisperASR,
    pcm_samples_to_wav_bytes,
)


def _parse_multipart(request: httpx.Request) -> dict[str, tuple[str | None, bytes]]:
    content_type = request.headers["Content-Type"]
    raw_msg = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + request.content
    )
    parsed = BytesParser(policy=email_policy).parsebytes(raw_msg)
    parts: dict[str, tuple[str | None, bytes]] = {}
    for part in parsed.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if name:
            parts[name] = (filename, payload)
    return parts


def test_pcm_samples_to_wav_bytes_packs_float32_and_clips() -> None:
    samples = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    wav_bytes = pcm_samples_to_wav_bytes(samples, sample_rate=16000)

    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 5
        frames = wf.readframes(5)
    unpacked = struct.unpack("<5h", frames)
    assert unpacked[0] == -32767
    assert unpacked[2] == 0
    assert unpacked[4] == 32767
    assert unpacked[1] == -round(0.5 * 32767)
    assert unpacked[3] == round(0.5 * 32767)


def test_pcm_samples_to_wav_bytes_preserves_int16_and_handles_list_and_empty() -> None:
    pcm16 = np.array([-32768, -1234, 0, 1234, 32767], dtype=np.int16)
    wav16 = pcm_samples_to_wav_bytes(pcm16, sample_rate=16000)
    with wave.open(io.BytesIO(wav16), "rb") as wf:
        assert wf.getnframes() == 5
        assert wf.readframes(5) == pcm16.tobytes()

    wav_list = pcm_samples_to_wav_bytes([0.25, -0.25], sample_rate=8000)
    with wave.open(io.BytesIO(wav_list), "rb") as wf:
        assert wf.getframerate() == 8000
        assert wf.getnframes() == 2

    wav_empty = pcm_samples_to_wav_bytes(np.zeros(0, dtype=np.float32))
    assert len(wav_empty) == 44
    with wave.open(io.BytesIO(wav_empty), "rb") as wf:
        assert wf.getnframes() == 0


def test_pcm_samples_to_wav_bytes_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        pcm_samples_to_wav_bytes([0.1, 0.2], sample_rate=0)
    with pytest.raises(ValueError, match="1D mono"):
        pcm_samples_to_wav_bytes(np.zeros((4, 2), dtype=np.float32))


def test_doubao_flash_asr_success_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOUBAO_ASR_APP_KEY", "test-app-key")
    monkeypatch.setenv("DOUBAO_ASR_ACCESS_KEY", "test-access-key")

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            headers={
                "X-Api-Status-Code": "20000000",
                "X-Api-Message": "OK",
                "X-Api-Logid": "log-123",
            },
            json={"result": {"text": "  今天上海天气怎么样  "}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    asr = DoubaoFlashASR(client=client)
    samples = np.array([0.0, 0.25, -0.25], dtype=np.float32)
    text = asr.transcribe(samples)

    assert text == "今天上海天气怎么样"
    assert captured["url"] == DOUBAO_FLASH_ENDPOINT
    headers = {k.lower(): v for k, v in captured["headers"].items()}  # type: ignore[union-attr]
    assert headers["x-api-app-key"] == "test-app-key"
    assert headers["x-api-access-key"] == "test-access-key"
    assert headers["x-api-resource-id"] == "volc.bigasr.auc_turbo"
    assert headers["x-api-sequence"] == "-1"
    assert len(headers["x-api-request-id"]) == 36

    body = captured["body"]
    assert body["request"]["model_name"] == "bigmodel"  # type: ignore[index]
    decoded_wav = base64.b64decode(body["audio"]["data"])  # type: ignore[index]
    assert decoded_wav[:4] == b"RIFF"
    with wave.open(io.BytesIO(decoded_wav), "rb") as wf:
        assert wf.getnframes() == 3
        assert wf.getframerate() == 16000


def test_doubao_flash_asr_silence_returns_empty_string_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOUBAO_ASR_APP_KEY", "app-1")
    monkeypatch.setenv("DOUBAO_ASR_ACCESS_KEY", "acc-1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-Api-Status-Code": "20000003", "X-Api-Message": "silence"},
            json={},
        )

    async def run() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as async_client:
            asr = DoubaoFlashASR(async_client=async_client)
            return await asr.atranscribe(np.ones(160, dtype=np.int16))

    assert asyncio.run(run()) == ""


def test_doubao_flash_asr_errors_and_missing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOUBAO_ASR_APP_KEY", raising=False)
    monkeypatch.delenv("DOUBAO_ASR_ACCESS_KEY", raising=False)

    asr_missing = DoubaoFlashASR()
    with pytest.raises(CloudASRConfigError, match="DOUBAO_ASR_APP_KEY"):
        asr_missing.transcribe([0.1, 0.2])

    monkeypatch.setenv("DOUBAO_ASR_APP_KEY", "app-only")
    with pytest.raises(CloudASRConfigError, match="DOUBAO_ASR_ACCESS_KEY"):
        asr_missing.transcribe([0.1, 0.2])

    monkeypatch.setenv("DOUBAO_ASR_ACCESS_KEY", "acc-now")

    def err_status_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "X-Api-Status-Code": "45000001",
                "X-Api-Message": "invalid audio",
                "X-Api-Logid": "trace-9",
            },
            json={},
        )

    asr_err = DoubaoFlashASR(client=httpx.Client(transport=httpx.MockTransport(err_status_handler)))
    with pytest.raises(CloudASRError, match="45000001"):
        asr_err.transcribe([0.1, 0.2])

    def http_500_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="bad gateway")

    asr_500 = DoubaoFlashASR(client=httpx.Client(transport=httpx.MockTransport(http_500_handler)))
    with pytest.raises(CloudASRError, match="HTTP 502"):
        asr_500.transcribe([0.1, 0.2])

    assert asr_500.transcribe([]) == ""


def test_openai_whisper_asr_multipart_upload_and_custom_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret-123")

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["parts"] = _parse_multipart(request)
        return httpx.Response(200, json={"text": " Hi Nomi, check my emails. "})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    asr = OpenAIWhisperASR(
        base_url="https://api.groq.com/openai/v1/",
        api_key_env="GROQ_API_KEY",
        model="whisper-large-v3-turbo",
        language="en",
        prompt="Hi Nomi",
        temperature=0.0,
        client=client,
    )
    text = asr.transcribe(np.array([0.1, -0.1, 0.2], dtype=np.float32))

    assert text == "Hi Nomi, check my emails."
    assert captured["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert captured["auth"] == "Bearer groq-secret-123"
    parts = captured["parts"]
    assert parts["model"] == (None, b"whisper-large-v3-turbo")  # type: ignore[index]
    assert parts["response_format"] == (None, b"json")  # type: ignore[index]
    assert parts["language"] == (None, b"en")  # type: ignore[index]
    assert parts["prompt"] == (None, b"Hi Nomi")  # type: ignore[index]
    assert parts["temperature"] == (None, b"0.0")  # type: ignore[index]
    filename, wav_payload = parts["file"]  # type: ignore[index]
    assert filename == "audio.wav"
    assert wav_payload[:4] == b"RIFF"
    with wave.open(io.BytesIO(wav_payload), "rb") as wf:
        assert wf.getnframes() == 3


def test_openai_whisper_asr_local_keyless_async_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    asr_missing = OpenAIWhisperASR()
    with pytest.raises(CloudASRConfigError, match="OPENAI_API_KEY"):
        asr_missing.transcribe([0.1, 0.2])

    seen_auth: list[str | None] = []

    def local_handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("Authorization"))
        return httpx.Response(200, text="本地语音识别结果\n")

    async def run_local() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(local_handler)) as async_client:
            asr = OpenAIWhisperASR(
                base_url="http://127.0.0.1:8000/v1",
                api_key_env=None,
                response_format="text",
                async_client=async_client,
            )
            return await asr.atranscribe(np.array([100, -100], dtype=np.int16))

    assert asyncio.run(run_local()) == "本地语音识别结果"
    assert seen_auth == [None]

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "Incorrect API key provided"}},
        )

    asr_err = OpenAIWhisperASR(client=httpx.Client(transport=httpx.MockTransport(error_handler)))
    with pytest.raises(CloudASRError, match="Incorrect API key provided"):
        asr_err.transcribe([0.1, 0.2])

    def bad_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    asr_bad = OpenAIWhisperASR(client=httpx.Client(transport=httpx.MockTransport(bad_json_handler)))
    with pytest.raises(CloudASRError, match="invalid JSON"):
        asr_bad.transcribe([0.1, 0.2])

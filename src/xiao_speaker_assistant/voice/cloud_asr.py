"""Low-local-compute Cloud ASR adapters for post-VAD speech segments."""

from __future__ import annotations

import base64
import io
import json
import uuid
import wave
from collections.abc import Sequence
from typing import Any, Protocol

import httpx
import numpy as np

from xiao_speaker_assistant.config import read_env_secret

DEFAULT_SAMPLE_RATE = 16000
DOUBAO_FLASH_ENDPOINT = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
DOUBAO_DEFAULT_RESOURCE_ID = "volc.bigasr.auc_turbo"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "whisper-1"

AudioSamples = np.ndarray | Sequence[float] | Sequence[int]


class CloudASRError(RuntimeError):
    """Base error raised by Cloud ASR adapters."""


class CloudASRConfigError(CloudASRError, ValueError):
    """Raised when required Cloud ASR configuration or env secrets are missing."""


class CloudASRClient(Protocol):
    """Protocol implemented by synchronous and asynchronous Cloud ASR adapters."""

    def transcribe(self, samples: AudioSamples, *, sample_rate: int | None = None) -> str: ...

    async def atranscribe(self, samples: AudioSamples, *, sample_rate: int | None = None) -> str: ...


def pcm_samples_to_wav_bytes(samples: AudioSamples, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Pack 16kHz float32/int16 mono audio samples into standard 16-bit PCM WAV bytes."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    arr = np.asarray(samples)
    if arr.ndim > 1:
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        else:
            raise ValueError(f"Expected 1D mono audio samples, got shape {arr.shape}")

    if arr.size == 0:
        pcm16 = np.zeros(0, dtype="<i2")
    elif np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == np.dtype("<i2"):
            pcm16 = arr
        else:
            pcm16 = np.clip(arr, -32768, 32767).astype("<i2")
    elif np.issubdtype(arr.dtype, np.floating):
        clipped = np.clip(arr.astype(np.float64, copy=False), -1.0, 1.0)
        pcm16 = np.clip(np.round(clipped * 32767.0), -32768, 32767).astype("<i2")
    else:
        raise TypeError(f"Unsupported audio sample dtype: {arr.dtype}")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _has_audio_samples(samples: AudioSamples) -> bool:
    return np.asarray(samples).size > 0


class DoubaoFlashASR:
    """Volcano Engine Doubao BigModel Flash recorded-file ASR client.

    Reads ``app_key`` and ``access_key`` from explicit environment variables
    (defaulting to ``DOUBAO_ASR_APP_KEY`` and ``DOUBAO_ASR_ACCESS_KEY``).
    """

    def __init__(
        self,
        *,
        app_key_env: str = "DOUBAO_ASR_APP_KEY",
        access_key_env: str = "DOUBAO_ASR_ACCESS_KEY",
        endpoint: str = DOUBAO_FLASH_ENDPOINT,
        resource_id: str = DOUBAO_DEFAULT_RESOURCE_ID,
        uid: str = "xiao-speaker-assistant",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        enable_itn: bool = True,
        enable_punc: bool = True,
        timeout_sec: float = 15.0,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.app_key_env = app_key_env
        self.access_key_env = access_key_env
        self.endpoint = endpoint
        self.resource_id = resource_id
        self.uid = uid
        self.sample_rate = sample_rate
        self.enable_itn = enable_itn
        self.enable_punc = enable_punc
        self.timeout_sec = timeout_sec
        self._client = client
        self._async_client = async_client

    def validate_credentials(self) -> tuple[str, str]:
        app_key = read_env_secret(self.app_key_env)
        if not app_key:
            raise CloudASRConfigError(f"Missing Doubao ASR app key in env {self.app_key_env}")
        access_key = read_env_secret(self.access_key_env)
        if not access_key:
            raise CloudASRConfigError(f"Missing Doubao ASR access key in env {self.access_key_env}")
        return app_key, access_key

    def _build_request(
        self, samples: AudioSamples, *, sample_rate: int | None = None
    ) -> tuple[dict[str, str], dict[str, Any]]:
        app_key, access_key = self.validate_credentials()
        wav_bytes = pcm_samples_to_wav_bytes(samples, sample_rate=sample_rate or self.sample_rate)
        headers = {
            "Content-Type": "application/json",
            "X-Api-App-Key": app_key,
            "X-Api-Access-Key": access_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        }
        payload: dict[str, Any] = {
            "user": {"uid": self.uid},
            "audio": {"data": base64.b64encode(wav_bytes).decode("ascii")},
            "request": {
                "model_name": "bigmodel",
                "enable_itn": self.enable_itn,
                "enable_punc": self.enable_punc,
            },
        }
        return headers, payload

    @staticmethod
    def _parse_response(resp: httpx.Response) -> str:
        if resp.status_code >= 400:
            detail = resp.text.strip() or f"HTTP {resp.status_code}"
            raise CloudASRError(f"Doubao ASR HTTP {resp.status_code}: {detail}")

        api_status = resp.headers.get("X-Api-Status-Code", "").strip()
        api_message = resp.headers.get("X-Api-Message", "").strip()
        log_id = resp.headers.get("X-Api-Logid", "").strip()

        if api_status == "20000003":
            return ""
        if api_status != "20000000":
            raise CloudASRError(
                f"Doubao ASR failed (status={api_status or 'missing'}, "
                f"message={api_message or 'unknown'}, logid={log_id or 'none'})"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise CloudASRError("Doubao ASR returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise CloudASRError("Doubao ASR returned unexpected JSON payload")

        result = data.get("result")
        if isinstance(result, dict):
            text = result.get("text", "")
            return str(text).strip() if text is not None else ""
        if isinstance(result, list):
            parts = [
                str(item.get("text", "")).strip()
                for item in result
                if isinstance(item, dict) and item.get("text")
            ]
            return " ".join(parts).strip()
        text = data.get("text", "")
        return str(text).strip() if text is not None else ""

    def transcribe(self, samples: AudioSamples, *, sample_rate: int | None = None) -> str:
        if not _has_audio_samples(samples):
            return ""
        headers, payload = self._build_request(samples, sample_rate=sample_rate)
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=self.timeout_sec)
        try:
            resp = client.post(self.endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise CloudASRError(f"Doubao ASR request failed: {exc}") from exc
        finally:
            if owns_client:
                client.close()
        return self._parse_response(resp)

    async def atranscribe(self, samples: AudioSamples, *, sample_rate: int | None = None) -> str:
        if not _has_audio_samples(samples):
            return ""
        headers, payload = self._build_request(samples, sample_rate=sample_rate)
        owns_client = self._async_client is None
        client = self._async_client or httpx.AsyncClient(timeout=self.timeout_sec)
        try:
            resp = await client.post(self.endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise CloudASRError(f"Doubao ASR request failed: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()
        return self._parse_response(resp)


class OpenAIWhisperASR:
    """OpenAI-compatible ``/audio/transcriptions`` multipart upload ASR client.

    Works with OpenAI, Groq, or a self-hosted Whisper-compatible server by
    customizing ``base_url``, ``model``, and ``api_key_env``.
    """

    def __init__(
        self,
        *,
        base_url: str = OPENAI_DEFAULT_BASE_URL,
        api_key_env: str | None = "OPENAI_API_KEY",
        model: str = OPENAI_DEFAULT_MODEL,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        timeout_sec: float = 20.0,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.model = model
        self.language = language
        self.prompt = prompt
        self.response_format = response_format
        self.temperature = temperature
        self.sample_rate = sample_rate
        self.timeout_sec = timeout_sec
        self._client = client
        self._async_client = async_client

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/audio/transcriptions"

    def validate_credentials(self) -> str | None:
        if self.api_key_env is None:
            return None
        env_name = self.api_key_env.strip()
        if not env_name:
            raise CloudASRConfigError("api_key_env must be non-empty or None")
        api_key = read_env_secret(env_name)
        if not api_key:
            raise CloudASRConfigError(f"Missing Whisper ASR API key in env {env_name}")
        return api_key

    def _build_request(
        self, samples: AudioSamples, *, sample_rate: int | None = None
    ) -> tuple[dict[str, str], dict[str, tuple[str, bytes, str]], dict[str, str]]:
        api_key = self.validate_credentials()
        wav_bytes = pcm_samples_to_wav_bytes(samples, sample_rate=sample_rate or self.sample_rate)
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data: dict[str, str] = {
            "model": self.model,
            "response_format": self.response_format,
        }
        if self.language and self.language.strip().lower() != "auto":
            data["language"] = self.language.strip()
        if self.prompt:
            data["prompt"] = self.prompt
        if self.temperature is not None:
            data["temperature"] = str(self.temperature)
        return headers, files, data

    def _parse_response(self, resp: httpx.Response) -> str:
        if resp.status_code >= 400:
            detail = self._extract_error_detail(resp)
            raise CloudASRError(f"Whisper ASR HTTP {resp.status_code}: {detail}")

        if self.response_format == "text":
            return resp.text.strip()

        try:
            data = resp.json()
        except ValueError as exc:
            raise CloudASRError("Whisper ASR returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise CloudASRError("Whisper ASR returned unexpected JSON payload")
        text = data.get("text", "")
        return str(text).strip() if text is not None else ""

    @staticmethod
    def _extract_error_detail(resp: httpx.Response) -> str:
        try:
            payload = resp.json()
        except (ValueError, json.JSONDecodeError):
            return resp.text.strip() or f"HTTP {resp.status_code}"
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
            if isinstance(err, str) and err:
                return err
            if payload.get("message"):
                return str(payload["message"])
        return resp.text.strip() or f"HTTP {resp.status_code}"

    def transcribe(self, samples: AudioSamples, *, sample_rate: int | None = None) -> str:
        if not _has_audio_samples(samples):
            return ""
        headers, files, data = self._build_request(samples, sample_rate=sample_rate)
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=self.timeout_sec)
        try:
            resp = client.post(self.endpoint, headers=headers, files=files, data=data)
        except httpx.HTTPError as exc:
            raise CloudASRError(f"Whisper ASR request failed: {exc}") from exc
        finally:
            if owns_client:
                client.close()
        return self._parse_response(resp)

    async def atranscribe(self, samples: AudioSamples, *, sample_rate: int | None = None) -> str:
        if not _has_audio_samples(samples):
            return ""
        headers, files, data = self._build_request(samples, sample_rate=sample_rate)
        owns_client = self._async_client is None
        client = self._async_client or httpx.AsyncClient(timeout=self.timeout_sec)
        try:
            resp = await client.post(self.endpoint, headers=headers, files=files, data=data)
        except httpx.HTTPError as exc:
            raise CloudASRError(f"Whisper ASR request failed: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()
        return self._parse_response(resp)

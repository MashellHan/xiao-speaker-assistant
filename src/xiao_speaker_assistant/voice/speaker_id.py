"""Local speaker embedding extraction & 0600 extensible family voiceprint store."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_ROLE_HINTS: dict[str, str] = {
    "host": "称呼为大雄或主人，直接高效回答",
    "大雄": "称呼为大雄，直接高效回答",
    "spouse": "称呼亲切自然，温柔体贴回答",
    "爱人": "称呼亲切自然，温柔体贴回答",
    "child": "用简单生动、温和耐心的儿童口吻回答，避免复杂术语",
    "小孩": "用简单生动、温和耐心的儿童口吻回答，避免复杂术语",
    "kid": "用简单生动、温和耐心的儿童口吻回答，避免复杂术语",
    "elder": "用亲切大白话、条理清晰、慢节奏耐心地回答",
    "老人": "用亲切大白话、条理清晰、慢节奏耐心地回答",
}

_DEFAULT_ROLE_CAPABILITIES: dict[str, list[str]] = {
    "child": ["storytelling", "homework_help", "science_qa", "kid_safe_only"],
    "小孩": ["storytelling", "homework_help", "science_qa", "kid_safe_only"],
    "kid": ["storytelling", "homework_help", "science_qa", "kid_safe_only"],
    "elder": ["health_tips", "weather_briefing", "large_clear_summary"],
    "老人": ["health_tips", "weather_briefing", "large_clear_summary"],
}


def default_style_hint_for_role(name: str, role: str) -> str:
    for key in (role.strip().lower(), name.strip().lower(), role.strip(), name.strip()):
        if key in _DEFAULT_ROLE_HINTS:
            return _DEFAULT_ROLE_HINTS[key]
    return ""


def default_capabilities_for_role(name: str, role: str) -> list[str]:
    for key in (role.strip().lower(), name.strip().lower(), role.strip(), name.strip()):
        if key in _DEFAULT_ROLE_CAPABILITIES:
            return list(_DEFAULT_ROLE_CAPABILITIES[key])
    return []


@dataclass
class SpeakerProfile:
    id: str
    name: str
    role: str
    style_hint: str
    embedding: list[float]
    capabilities: list[str] = field(default_factory=list)
    notes: str = ""
    preferred_voice: str | None = None
    sample_count: int = 1
    updated_at: float = 0.0


@dataclass(frozen=True)
class SpeakerMatch:
    id: str
    name: str
    role: str
    style_hint: str
    score: float
    capabilities: tuple[str, ...] = ()
    notes: str = ""
    preferred_voice: str | None = None

    def format_context_prefix(self) -> str:
        parts = [f"当前说话人声纹识别：{self.name}（角色：{self.role}，相似度：{self.score:.2f}）"]
        if self.style_hint:
            parts.append(f"称呼与语气：{self.style_hint}")
        if self.capabilities:
            parts.append(f"专属能力与限制：{', '.join(self.capabilities)}")
        if self.notes:
            parts.append(f"成员备注：{self.notes}")
        return f"[{'；'.join(parts)}]"


@dataclass(frozen=True)
class RecentUtteranceRecord:
    text: str
    embedding: list[float] | None
    matched_name: str | None
    timestamp: float


class SpeakerProfileStore:
    """Thread-safe 0600 JSON store of extensible family member voiceprints."""

    def __init__(
        self,
        path: str | Path = "~/.local/share/xiao-speaker-assistant/speakers.json",
        *,
        recent_limit: int = 8,
    ) -> None:
        self.path = Path(path).expanduser()
        self._lock = threading.RLock()
        self._profiles: list[SpeakerProfile] = []
        self._recent_utterances: deque[RecentUtteranceRecord] = deque(maxlen=recent_limit)
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._profiles = []
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._profiles = []
                return
            if not isinstance(raw, list):
                self._profiles = []
                return
            loaded: list[SpeakerProfile] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                emb = item.get("embedding")
                if not name or not isinstance(emb, list) or not emb:
                    continue
                raw_caps = item.get("capabilities")
                caps = (
                    [str(c).strip() for c in raw_caps if str(c).strip()]
                    if isinstance(raw_caps, list)
                    else []
                )
                pv = item.get("preferred_voice")
                loaded.append(
                    SpeakerProfile(
                        id=str(item.get("id") or uuid.uuid4().hex[:8]),
                        name=name,
                        role=str(item.get("role") or "family"),
                        style_hint=str(item.get("style_hint") or ""),
                        embedding=[float(x) for x in emb],
                        capabilities=caps,
                        notes=str(item.get("notes") or ""),
                        preferred_voice=str(pv).strip() if isinstance(pv, str) and pv.strip() else None,
                        sample_count=max(1, int(item.get("sample_count") or 1)),
                        updated_at=float(item.get("updated_at") or 0.0),
                    )
                )
            self._profiles = loaded

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        tmp = self.path.with_suffix(".tmp")
        payload = json.dumps([asdict(p) for p in self._profiles], ensure_ascii=False, indent=2) + "\n"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)
        os.chmod(self.path, 0o600)

    @staticmethod
    def _normalize(vec: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-8:
            return arr
        return arr / norm

    def record_utterance(
        self,
        text: str,
        embedding: np.ndarray | list[float] | None,
        *,
        matched: SpeakerMatch | None = None,
        timestamp: float | None = None,
    ) -> None:
        emb_list: list[float] | None = None
        if embedding is not None:
            vec = self._normalize(embedding)
            if vec.size > 0 and float(np.linalg.norm(vec)) > 1e-6:
                emb_list = [round(float(x), 6) for x in vec.tolist()]
        rec = RecentUtteranceRecord(
            text=text.strip(),
            embedding=emb_list,
            matched_name=matched.name if matched is not None else None,
            timestamp=time.time() if timestamp is None else timestamp,
        )
        with self._lock:
            self._recent_utterances.append(rec)

    def recent_utterances(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "index": idx,
                    "text": rec.text,
                    "has_embedding": rec.embedding is not None,
                    "matched_name": rec.matched_name,
                    "timestamp": rec.timestamp,
                }
                for idx, rec in enumerate(self._recent_utterances)
            ]

    def _select_recent_embedding(
        self, *, target: str = "previous", keyword: str = ""
    ) -> RecentUtteranceRecord | None:
        with self._lock:
            records = list(self._recent_utterances)
        with_emb = [r for r in records if r.embedding is not None]
        if not with_emb:
            return None
        kw = keyword.strip().lower()
        if kw:
            for rec in reversed(with_emb):
                if kw in rec.text.lower():
                    return rec
        norm_target = target.strip().lower()
        if norm_target == "current":
            return with_emb[-1]
        # "previous": strictly require an earlier utterance before the enrollment command;
        # never fall back to with_emb[-1] (which is the caller's own enrollment utterance).
        if len(with_emb) >= 2:
            return with_emb[-2]
        return None

    def enroll(
        self,
        name: str,
        embedding: np.ndarray | list[float],
        *,
        role: str = "family",
        style_hint: str = "",
        capabilities: list[str] | None = None,
        notes: str = "",
        preferred_voice: str | None = None,
    ) -> SpeakerProfile:
        clean_name = re.sub(r"\s+", " ", name.strip())
        clean_role = re.sub(r"\s+", " ", role.strip()) or "family"
        clean_hint = re.sub(r"\s+", " ", style_hint.strip()) or default_style_hint_for_role(
            clean_name, clean_role
        )
        clean_notes = re.sub(r"\s+", " ", notes.strip())
        clean_caps = (
            list(dict.fromkeys(str(c).strip() for c in capabilities if str(c).strip()))
            if capabilities is not None
            else default_capabilities_for_role(clean_name, clean_role)
        )
        clean_voice = preferred_voice.strip() if isinstance(preferred_voice, str) and preferred_voice.strip() else None

        if not clean_name or len(clean_name) > 40:
            raise ValueError("Speaker 'name' must be 1..40 characters.")
        vec = self._normalize(embedding)
        if vec.size == 0 or float(np.linalg.norm(vec)) <= 1e-6:
            raise ValueError("Invalid empty voiceprint embedding.")

        with self._lock:
            now = time.time()
            for idx, existing in enumerate(self._profiles):
                if existing.name.lower() == clean_name.lower():
                    old_vec = self._normalize(existing.embedding)
                    if old_vec.shape == vec.shape:
                        merged = self._normalize(
                            old_vec * float(existing.sample_count) + vec
                        )
                    else:
                        merged = vec
                    merged_caps = (
                        list(dict.fromkeys([*existing.capabilities, *clean_caps]))
                        if capabilities is not None
                        else (existing.capabilities or clean_caps)
                    )
                    updated = SpeakerProfile(
                        id=existing.id,
                        name=clean_name,
                        role=clean_role if role != "family" else existing.role,
                        style_hint=clean_hint or existing.style_hint,
                        embedding=[round(float(x), 6) for x in merged.tolist()],
                        capabilities=merged_caps,
                        notes=clean_notes or existing.notes,
                        preferred_voice=clean_voice if clean_voice is not None else existing.preferred_voice,
                        sample_count=existing.sample_count + 1,
                        updated_at=now,
                    )
                    self._profiles[idx] = updated
                    self._persist()
                    return updated

            created = SpeakerProfile(
                id=uuid.uuid4().hex[:8],
                name=clean_name,
                role=clean_role,
                style_hint=clean_hint,
                embedding=[round(float(x), 6) for x in vec.tolist()],
                capabilities=clean_caps,
                notes=clean_notes,
                preferred_voice=clean_voice,
                sample_count=1,
                updated_at=now,
            )
            self._profiles.append(created)
            self._persist()
            return created

    def enroll_from_recent(
        self,
        name: str,
        *,
        role: str = "family",
        style_hint: str = "",
        capabilities: list[str] | None = None,
        notes: str = "",
        preferred_voice: str | None = None,
        target: str = "previous",
        keyword: str = "",
    ) -> tuple[SpeakerProfile, RecentUtteranceRecord]:
        rec = self._select_recent_embedding(target=target, keyword=keyword)
        if rec is None or rec.embedding is None:
            raise ValueError("No recent speaker audio embedding is available to enroll yet.")
        profile = self.enroll(
            name,
            rec.embedding,
            role=role,
            style_hint=style_hint,
            capabilities=capabilities,
            notes=notes,
            preferred_voice=preferred_voice,
        )
        return profile, rec

    def update_profile(
        self,
        name_or_id: str,
        *,
        role: str | None = None,
        style_hint: str | None = None,
        capabilities: list[str] | None = None,
        notes: str | None = None,
        preferred_voice: str | None = None,
    ) -> SpeakerProfile | None:
        target = name_or_id.strip().lower()
        if not target:
            return None
        with self._lock:
            for idx, existing in enumerate(self._profiles):
                if existing.id.lower() != target and existing.name.lower() != target:
                    continue
                new_role = re.sub(r"\s+", " ", role.strip()) if role and role.strip() else existing.role
                new_hint = (
                    re.sub(r"\s+", " ", style_hint.strip())
                    if style_hint is not None
                    else existing.style_hint
                )
                new_caps = (
                    list(dict.fromkeys(str(c).strip() for c in capabilities if str(c).strip()))
                    if capabilities is not None
                    else existing.capabilities
                )
                new_notes = (
                    re.sub(r"\s+", " ", notes.strip())
                    if notes is not None
                    else existing.notes
                )
                if preferred_voice is None:
                    new_voice = existing.preferred_voice
                else:
                    new_voice = preferred_voice.strip() or None
                updated = SpeakerProfile(
                    id=existing.id,
                    name=existing.name,
                    role=new_role,
                    style_hint=new_hint,
                    embedding=existing.embedding,
                    capabilities=new_caps,
                    notes=new_notes,
                    preferred_voice=new_voice,
                    sample_count=existing.sample_count,
                    updated_at=time.time(),
                )
                self._profiles[idx] = updated
                self._persist()
                return updated
        return None

    def identify(
        self,
        embedding: np.ndarray | list[float],
        *,
        threshold: float = 0.45,
    ) -> SpeakerMatch | None:
        vec = self._normalize(embedding)
        if vec.size == 0 or float(np.linalg.norm(vec)) <= 1e-6:
            return None

        best: SpeakerProfile | None = None
        best_score = -1.0
        with self._lock:
            for profile in self._profiles:
                p_vec = self._normalize(profile.embedding)
                if p_vec.shape != vec.shape:
                    continue
                score = float(np.dot(vec, p_vec))
                if score > best_score:
                    best_score = score
                    best = profile
        if best is not None and best_score >= threshold:
            return SpeakerMatch(
                id=best.id,
                name=best.name,
                role=best.role,
                style_hint=best.style_hint,
                score=round(best_score, 4),
                capabilities=tuple(best.capabilities),
                notes=best.notes,
                preferred_voice=best.preferred_voice,
            )
        return None

    def list_profiles(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "role": p.role,
                    "style_hint": p.style_hint,
                    "capabilities": list(p.capabilities),
                    "notes": p.notes,
                    "preferred_voice": p.preferred_voice,
                    "sample_count": p.sample_count,
                }
                for p in self._profiles
            ]

    def delete_profile(self, name_or_id: str) -> bool:
        target = name_or_id.strip().lower()
        if not target:
            return False
        with self._lock:
            before = len(self._profiles)
            self._profiles = [
                p
                for p in self._profiles
                if p.id.lower() != target and p.name.lower() != target
            ]
            removed = len(self._profiles) < before
            if removed:
                self._persist()
            return removed


class SpeakerIdentifier:
    """Extracts 192-dim speaker embeddings via `sherpa_onnx.SpeakerEmbeddingExtractor`."""

    def __init__(self, model_path: Path, *, sample_rate: int = 16000) -> None:
        import sherpa_onnx

        self._sample_rate = sample_rate
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path),
            num_threads=1,
            provider="cpu",
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.dim = self._extractor.dim

    def extract_embedding(self, samples: list[float] | np.ndarray) -> np.ndarray | None:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.size < int(self._sample_rate * 0.35):
            return None
        stream = self._extractor.create_stream()
        stream.accept_waveform(self._sample_rate, arr)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            return None
        emb = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-8:
            return None
        return emb / norm

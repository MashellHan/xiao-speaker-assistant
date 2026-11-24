"""Download and prepare VAD + KWS + SenseVoice models for local `Hi Nomi` wakeup."""

from __future__ import annotations

import hashlib
import re
import subprocess
import zipfile
from pathlib import Path

MODELS_URL = (
    "https://github.com/coderzc/open-xiaoai-bridge/releases/download/"
    "vad-kws-asr-models/models.zip"
)
MODELS_ZIP_SHA256 = "8e8a709d6ea011f644f3c5055f536d5e6eb363eeafba53cc3232398fd636d273"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def find_sense_voice_dir(models_dir: Path) -> Path:
    for entry in sorted(models_dir.iterdir()):
        if entry.is_dir() and "sense-voice" in entry.name:
            if (entry / "model.int8.onnx").is_file() and (entry / "tokens.txt").is_file():
                return entry
    raise FileNotFoundError(
        f"SenseVoice model directory not found under {models_dir}. Run `xsa download-models`."
    )


def find_paraformer_dir(models_dir: Path) -> Path:
    for entry in sorted(models_dir.iterdir()):
        if entry.is_dir() and "paraformer" in entry.name:
            if (entry / "model.int8.onnx").is_file() and (entry / "tokens.txt").is_file():
                return entry
    raise FileNotFoundError(
        f"Paraformer model directory not found under {models_dir}. Run `xsa download-models`."
    )


def generate_keywords_file(
    models_dir: Path, keywords: list[str], filename: str = "keywords.txt"
) -> Path:
    from sherpa_onnx import text2token

    tokens = models_dir / "tokens.txt"
    bpe_model = models_dir / "bpe.model"
    output = models_dir / filename
    if not tokens.is_file() or not bpe_model.is_file():
        raise FileNotFoundError(
            f"Missing tokens.txt or bpe.model in {models_dir}. Run `xsa download-models`."
        )

    texts = [kw.strip().upper() for kw in keywords if kw.strip()]
    if not texts:
        raise ValueError("At least one local wake keyword is required")

    encoded = text2token(
        texts,
        tokens=str(tokens),
        tokens_type="cjkchar+bpe",
        bpe_model=str(bpe_model),
    )
    lines: list[str] = []
    for raw_kw, token_list in zip(keywords, encoded):
        joined = "".join(token_list)
        tag = re.sub(r"\s+", "_", raw_kw.strip())
        if re.match(r"^[▁A-Z\s]+$", joined):
            lines.append(" ".join(token_list) + f" @{tag}")
        else:
            lines.append(" ".join(token_list) + f" @{re.sub(r'\s+', '_', joined)}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def download_models(models_dir: Path, keywords: list[str]) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    zip_path = models_dir / "models.zip"

    need_download = True
    if zip_path.is_file():
        if sha256_file(zip_path) == MODELS_ZIP_SHA256:
            need_download = False
        else:
            zip_path.unlink()

    if need_download:
        print(f"Downloading VAD+KWS+SenseVoice models (~470 MB) to {zip_path}...", flush=True)
        subprocess.run(
            ["curl", "--fail", "--location", "--progress-bar", MODELS_URL, "--output", str(zip_path)],
            check=True,
        )
        digest = sha256_file(zip_path)
        if digest != MODELS_ZIP_SHA256:
            raise RuntimeError(f"models.zip SHA-256 mismatch: {digest}")

    print(f"Extracting {zip_path.name} into {models_dir}...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(models_dir)

    # Flatten nested models/ if archive contained a top-level models/ folder.
    nested = models_dir / "models"
    if nested.is_dir():
        for item in nested.iterdir():
            dest = models_dir / item.name
            if not dest.exists():
                item.rename(dest)

    kw_path = generate_keywords_file(models_dir, keywords)
    print(f"OK: models ready in {models_dir}, keywords generated at {kw_path}")
    return kw_path

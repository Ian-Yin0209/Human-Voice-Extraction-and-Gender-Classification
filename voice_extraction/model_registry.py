"""An offline model catalog and an explicit, checksum-verified fetch command."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
import zipfile


MODEL_ID = "mimo-scnet"
WEIGHT_NAME = "mimo-scnet-small-v1.pth"
WEIGHT_BYTES = 42491051
WEIGHT_SHA256 = "1d19305440e8439faeb4163d809b2dcaaffacf67f6f926425be581290d6536ee"
ARCHIVE_BYTES = 39354515
ARCHIVE_SHA256 = "f0eaaab7e899ffc548b29d51d76b89727253fb0a529b787b258eabfcf47db6fb"
ARCHIVE_URL = (
    "https://github.com/SonyResearch/mimo-audio-separation/releases/download/"
    "v1.0.0/mimo-scnet_small.zip"
)
MEMBER_NAME = (
    "runs/train/iterative-separation-4src/scnet-4src_small/329029/"
    "ckpt/mimo-scnet_small/backbone_model.pth"
)
SOURCE_REVISION = "e172761a0798bef8a549d440a276218529b53edb"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected_bytes: int, expected_sha256: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular local model file: {path}")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
        raise ValueError(f"Model checksum/size mismatch: {path}. Obtain the documented release file.")
    return path


def verify_weights(weights: str | Path) -> Path:
    """Accept only the documented, fixed upstream state dict; never fetch implicitly."""
    path = Path(weights).expanduser().absolute()
    return _verify(path, WEIGHT_BYTES, WEIGHT_SHA256).resolve()


def model_catalog() -> dict:
    """No network calls and no ML imports."""
    return {
        "id": MODEL_ID,
        "name": "Sony MIMO-SCNet small (2026 research integration)",
        "stems": ["vocals", "bass", "drums", "other"],
        "sample_rate": 44100,
        "iterations": [1, 2],
        "paper": "https://arxiv.org/abs/2609.07226",
        "source_revision": SOURCE_REVISION,
        "code_license": "MIT (Sony and SCNet notices included)",
        "weight_license": "No separate explicit weight license found; review upstream terms.",
        "weights_bundled": False,
        "archive_url": ARCHIVE_URL,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "weight_filename": WEIGHT_NAME,
        "weight_bytes": WEIGHT_BYTES,
        "weight_sha256": WEIGHT_SHA256,
        "validation_scope": "See docs/RESEARCH.md and the fixed-checkpoint validation record.",
    }


def fetch_model(models_dir: str | Path, *, archive: str | Path | None = None) -> dict:
    """Download once on explicit request, or install from a verified local ZIP.

    Only one known member is read; ZIP paths are never extracted to disk. An
    existing valid checkpoint is reused. An invalid file is never overwritten.
    """
    directory = Path(models_dir).expanduser().resolve()
    destination = directory / WEIGHT_NAME
    if destination.exists() or destination.is_symlink():
        verify_weights(destination)
        return {"model_id": MODEL_ID, "weights": str(destination), "downloaded": False}
    local_archive = None if archive is None else Path(archive).expanduser().resolve()
    if local_archive is not None:
        _verify(local_archive, ARCHIVE_BYTES, ARCHIVE_SHA256)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".voice-model-", dir=directory) as temporary:
            temporary = Path(temporary)
            if local_archive is None:
                local_archive = temporary / "model.zip"
                request = Request(ARCHIVE_URL, headers={"User-Agent": "VoiceExtractionToolkit/0.3"})
                with urlopen(request, timeout=30) as response, local_archive.open("wb") as output:
                    count = 0
                    while block := response.read(1024 * 1024):
                        count += len(block)
                        if count > ARCHIVE_BYTES:
                            raise ValueError("Model download exceeded its pinned byte count.")
                        output.write(block)
                _verify(local_archive, ARCHIVE_BYTES, ARCHIVE_SHA256)
            with zipfile.ZipFile(local_archive) as bundle:
                entry = bundle.getinfo(MEMBER_NAME)
                if entry.file_size != WEIGHT_BYTES:
                    raise ValueError("Unexpected model archive member size.")
                candidate = temporary / WEIGHT_NAME
                with bundle.open(entry) as source, candidate.open("wb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
            verify_weights(candidate)
            # Same filesystem: create-if-absent, including in a concurrent fetch.
            os.link(candidate, destination)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("The model archive is incomplete or has an unexpected layout.") from exc
    return {"model_id": MODEL_ID, "weights": str(destination), "downloaded": archive is None,
            "sha256": WEIGHT_SHA256}

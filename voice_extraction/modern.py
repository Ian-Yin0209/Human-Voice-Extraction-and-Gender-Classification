"""Local four-stem inference with the fixed Sony MIMO-SCNet small checkpoint."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from . import __version__
from .audio import load_audio, resample_audio, write_wav
from .model_registry import MODEL_ID, SOURCE_REVISION, WEIGHT_SHA256, verify_weights


STEMS = ("vocals", "bass", "drums", "other")


def _load_model(weights: Path, device: str):
    try:
        import torch
        from ._vendor.sony_scnet.scnet import MultiSourceSCNet
    except ImportError as exc:
        raise ValueError("MIMO separation requires PyTorch: pip install '.[modern]'.") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a compatible PyTorch build or use --device cpu.")
    model = MultiSourceSCNet(
        n_src=4, audio_channels=2, multi_src_input=True, use_time_embed=True,
        dim_time_emb=256, dims=[16, 32, 64, 128], gradient_checkpointing=False,
    )
    # Only a pinned tensor-only checkpoint is accepted; no YAML/Hydra targets run.
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    times = torch.linspace(0, 1, steps=2, device=device)

    def infer(chunk: np.ndarray, iterations: int) -> np.ndarray:
        length = len(chunk)
        # STFT reflect padding needs a minimum clip length. Crop exactly after inference.
        if length < 4096:
            chunk = np.pad(chunk, ((0, 4096 - length), (0, 0)))
        waveform = torch.from_numpy(np.ascontiguousarray(chunk.T)).to(device).unsqueeze(0)
        with torch.inference_mode():
            peak = waveform.abs().amax(dim=(-1, -2), keepdim=True)
            divisor = torch.where(peak < 1e-8, torch.ones_like(peak), peak + 1e-8)
            normalized = waveform / divisor
            estimate = (normalized / 4).unsqueeze(1).repeat(1, 4, 1, 1)
            for step in range(iterations):
                predicted = model(estimate, times[step].expand(1))
                # Direct prediction with the upstream equal-share consistency projection.
                estimate = predicted + (
                    estimate.sum(dim=1, keepdim=True) - predicted.sum(dim=1, keepdim=True)
                ) / 4
            estimate = estimate * divisor.unsqueeze(1)
        return estimate[0, :, :, :length].permute(0, 2, 1).cpu().numpy()

    return infer


def _runtime_versions() -> dict:
    result = {}
    for name in ("numpy", "scipy", "soundfile", "torch"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _publish(stage: Path, destination: Path) -> None:
    """Reserve a new directory exclusively; publish the completion report last."""
    destination.mkdir()  # Never replace any pre-existing directory or file.
    moved = []
    try:
        for name in [*(f"{stem}.wav" for stem in STEMS), "accompaniment.wav", "report.json"]:
            target = destination / name
            (stage / name).rename(target)
            moved.append(target)
    except BaseException:
        for target in reversed(moved):
            target.unlink(missing_ok=True)
        destination.rmdir()
        raise


def separate_mimo_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    weights: str | Path,
    device: str = "cpu",
    chunk_seconds: float = 6.0,
    overlap: float = 0.25,
    iterations: int = 2,
) -> dict:
    """Separate to a new directory; inference never downloads models or uploads audio."""
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda.")
    if not math.isfinite(chunk_seconds) or not 0.1 <= chunk_seconds <= 30:
        raise ValueError("Chunk seconds must be finite and between 0.1 and 30.")
    if not math.isfinite(overlap) or not 0 <= overlap <= 0.75:
        raise ValueError("Overlap must be a finite fraction between 0 and 0.75.")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations not in (1, 2):
        raise ValueError("Iterations must be 1 or 2 for this checkpoint.")
    chunk_seconds, overlap, iterations = float(chunk_seconds), float(overlap), int(iterations)
    destination = Path(output_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Output directory must be new: {destination}")
    checkpoint = verify_weights(weights)
    samples, source_rate = load_audio(input_path)
    samples = resample_audio(samples, source_rate, 44100)
    if samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    start_time = perf_counter()
    infer = _load_model(checkpoint, device)
    count = len(samples)
    chunk_frames = round(chunk_seconds * 44100)
    hop = max(1, round(chunk_frames * (1 - overlap)))
    combined = np.zeros((4, count, 2), dtype=np.float32)
    normalization = np.zeros(count, dtype=np.float32)
    chunks = 0
    for start in range(0, count, hop):
        end = min(count, start + chunk_frames)
        part = samples[start:end]
        stems = np.asarray(infer(part, iterations), dtype=np.float32)
        if stems.shape != (4, len(part), 2) or not np.isfinite(stems).all():
            raise ValueError("MIMO separator returned invalid stems or changed the audio length.")
        # Positive Hann weights avoid silent endpoint samples, including short clips.
        window = np.maximum(np.hanning(len(part) + 2)[1:-1], 1e-3).astype(np.float32)
        combined[:, start:end] += stems * window[None, :, None]
        normalization[start:end] += window
        chunks += 1
        if end == count:
            break
    combined /= normalization[None, :, None]
    if not np.isfinite(combined).all():
        raise ValueError("Non-finite output after overlap-add.")
    accompaniment = combined[1:].sum(axis=0)
    mixture_error = float(np.max(np.abs(combined.sum(axis=0) - samples)))
    report = {
        "model_id": MODEL_ID,
        "toolkit_version": __version__,
        "model_sha256": WEIGHT_SHA256,
        "source_revision": SOURCE_REVISION,
        "sample_rate": 44100,
        "source_sample_rate": source_rate,
        "channels": 2,
        "frames": count,
        "iterations": iterations,
        "device": device,
        "chunk_seconds": chunk_seconds,
        "overlap": overlap,
        "chunks": chunks,
        "seconds": perf_counter() - start_time,
        "timing_scope": "model_loading_inference_overlap_add; excludes audio I/O and weight verification",
        "mixture_max_abs_error": mixture_error,
        "stems": {stem: f"{stem}.wav" for stem in STEMS},
        "accompaniment": "accompaniment.wav",
        "runtime": _runtime_versions(),
        "validation_scope": "Inference/reconstruction record; no reference quality score.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".voice-stems-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        for stem, audio in zip(STEMS, combined):
            write_wav(stage / f"{stem}.wav", audio, 44100)
        write_wav(stage / "accompaniment.wav", accompaniment, 44100)
        (stage / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        _publish(stage, destination)
    return report

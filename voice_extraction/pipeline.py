"""Optional model-backed separation with explicit local checkpoint paths."""

from __future__ import annotations

from glob import escape
from pathlib import Path
import re

import numpy as np

from .audio import check_output, load_audio, resample_audio, write_wav


def validate_checkpoint(prefix: str | Path) -> Path:
    prefix = Path(prefix).expanduser().resolve()
    shards = []
    pattern = re.compile(re.escape(prefix.name) + r"\.data-(\d{5})-of-(\d{5})$")
    for candidate in prefix.parent.glob(escape(prefix.name) + ".data-*-of-*"):
        match = pattern.fullmatch(candidate.name)
        if candidate.is_file() and match:
            shards.append((int(match[1]), int(match[2])))
    counts = {count for _, count in shards}
    complete = (
        len(counts) == 1 and next(iter(counts)) > 0
        and {index for index, _ in shards} == set(range(next(iter(counts))))
    )
    if not Path(str(prefix) + ".index").is_file() or not complete:
        raise ValueError(
            f"TensorFlow checkpoint is incomplete: {prefix}. "
            "Pass the common prefix of the .index and .data-* files."
        )
    return prefix


def separate_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint: str | Path,
    device: str = "cpu",
    force: bool = False,
) -> dict:
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda.")
    directory = Path(output_dir)
    outputs = [directory / "vocals.wav", directory / "accompaniment.wav"]
    for path in outputs:
        check_output(input_path, path, force=force)
    checkpoint_path = validate_checkpoint(checkpoint)
    samples, source_rate = load_audio(input_path)
    samples = resample_audio(samples, source_rate, 44100)
    if samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    try:
        import torch
        from spleeter_style.estimator import Estimator
    except ImportError as exc:
        raise ValueError("Separation requires model dependencies: pip install '.[models]'.") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu.")
    model = Estimator(2, str(checkpoint_path)).to(device).eval()
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T)).to(device)
    with torch.inference_mode():
        stems = model.separate(waveform)
    if len(stems) != 2:
        raise ValueError("Separator must produce exactly two stems.")
    arrays = [stem.detach().cpu().numpy().T for stem in stems]
    for array in arrays:
        if array.shape != samples.shape or not np.isfinite(array).all():
            raise ValueError("Separator returned an invalid stem or changed the audio length.")
    for path, array in zip(outputs, arrays):
        write_wav(path, array, 44100, force=force)
    return {
        "sample_rate": 44100,
        "channels": 2,
        "frames": len(samples),
        "vocals": str(outputs[0]),
        "accompaniment": str(outputs[1]),
    }

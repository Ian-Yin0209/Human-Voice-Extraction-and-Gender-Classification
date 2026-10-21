"""Validated audio I/O with a consistent (frames, channels) array layout."""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


def _validate_samples(samples: np.ndarray) -> None:
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] not in (1, 2):
        raise ValueError("Audio must contain at least one frame and one or two channels.")
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples (NaN or infinity).")


def _validate_rate(sample_rate: int) -> None:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise ValueError("Sample rate must be a positive integer.")
    if sample_rate <= 0 or sample_rate > 384000:
        raise ValueError("Sample rate must be between 1 and 384000 Hz.")


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read finite mono/stereo samples as float32 without normalization."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Audio file does not exist: {source}")
    try:
        samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, OSError, ValueError) as exc:
        raise ValueError(f"Cannot read audio file {source}: {exc}") from exc
    _validate_rate(sample_rate)
    _validate_samples(samples)
    return samples, sample_rate


def select_segment(
    samples: np.ndarray, sample_rate: int, start: float = 0.0, duration: float | None = None
) -> np.ndarray:
    """Select seconds, clipping the requested end to EOF but never returning empty audio."""
    _validate_samples(samples)
    _validate_rate(sample_rate)
    if not math.isfinite(start) or start < 0:
        raise ValueError("Start must be a finite, non-negative number of seconds.")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Duration must be a finite, positive number of seconds.")
    total_seconds = len(samples) / sample_rate
    if start >= total_seconds:
        raise ValueError("Start is at or beyond the end of the audio.")
    first = round(start * sample_rate)
    if first >= len(samples):
        raise ValueError("Start is at or beyond the end of the audio.")
    bounded_duration = None if duration is None else min(duration, total_seconds - start)
    last = (
        len(samples) if bounded_duration is None
        else min(len(samples), first + round(bounded_duration * sample_rate))
    )
    if last <= first:
        raise ValueError("Selected duration is shorter than one audio frame.")
    return samples[first:last].copy()


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    _validate_samples(samples)
    _validate_rate(source_rate)
    _validate_rate(target_rate)
    if source_rate == target_rate:
        return samples.copy()
    divisor = math.gcd(source_rate, target_rate)
    result = resample_poly(samples, target_rate // divisor, source_rate // divisor, axis=0)
    result = np.asarray(result, dtype=np.float32)
    _validate_samples(result)
    return result


def same_path(first: str | Path, second: str | Path) -> bool:
    first, second = Path(first), Path(second)
    if first.exists() and second.exists():
        return os.path.samefile(first, second)
    return os.path.normcase(str(first.resolve())) == os.path.normcase(str(second.resolve()))


def check_output(source: str | Path, destination: str | Path, *, force: bool = False) -> None:
    if same_path(source, destination):
        raise ValueError("Input and output must be different files, including with --force.")
    destination = Path(destination)
    if destination.exists() and (not force or not destination.is_file()):
        raise ValueError(f"Output already exists: {destination}. Use --force to replace a file.")
    if destination.is_symlink():
        raise ValueError(f"Refusing to replace a symbolic link: {destination}")


def write_wav(
    path: str | Path, samples: np.ndarray, sample_rate: int, *, force: bool = False
) -> None:
    """Write a float WAV via a temporary file; never clip or normalize implicitly."""
    _validate_samples(samples)
    _validate_rate(sample_rate)
    destination = Path(path)
    if destination.suffix.lower() != ".wav":
        raise ValueError("Output filename must end in .wav.")
    if destination.exists() and not force:
        raise ValueError(f"Output already exists: {destination}. Use --force to replace it.")
    if destination.is_symlink():
        raise ValueError(f"Refusing to replace a symbolic link: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".voice-", suffix=".wav", dir=destination.parent)
    os.close(descriptor)
    try:
        sf.write(temporary, samples, sample_rate, format="WAV", subtype="FLOAT")
        if force:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def audio_metadata(samples: np.ndarray, sample_rate: int) -> dict:
    return {
        "sample_rate": int(sample_rate),
        "channels": int(samples.shape[1]),
        "frames": int(len(samples)),
        "duration_seconds": len(samples) / sample_rate,
        "peak": float(np.max(np.abs(samples))),
    }


def inspect_file(path: str | Path) -> dict:
    samples, sample_rate = load_audio(path)
    return {"path": str(Path(path)), **audio_metadata(samples, sample_rate)}


def prepare_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start: float = 0.0,
    duration: float | None = None,
    sample_rate: int = 44100,
    mono: bool = False,
    normalize: bool = False,
    force: bool = False,
) -> dict:
    check_output(input_path, output_path, force=force)
    samples, source_rate = load_audio(input_path)
    samples = select_segment(samples, source_rate, start, duration)
    if mono:
        samples = samples.mean(axis=1, keepdims=True)
    samples = resample_audio(samples, source_rate, sample_rate)
    if normalize:
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples = samples * (0.99 / peak)
    write_wav(output_path, samples, sample_rate, force=force)
    return {"path": str(Path(output_path)), **audio_metadata(samples, sample_rate)}

"""Check local pretrained-weight compatibility on an original stereo tone.

This is an integration smoke test, not a speech or separation-quality benchmark.
No weights or audio are downloaded. The output directory must not exist.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np

from voice_extraction.audio import load_audio, write_wav
from voice_extraction.classification import classify_file
from voice_extraction.pipeline import separate_file, validate_checkpoint


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"filename": Path(path).name, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = validate_checkpoint(args.checkpoint)
    if not args.weights.is_file():
        parser.error("--weights must name an existing local HDF5 file")
    if args.output_dir.exists():
        parser.error("--output-dir must not exist")
    args.output_dir.mkdir(parents=True)

    sample_rate = 44100
    positions = np.arange(2 * sample_rate) / sample_rate
    mixture = np.column_stack((
        0.25 * np.sin(2 * np.pi * 440 * positions),
        0.125 * np.sin(2 * np.pi * 660 * positions),
    )).astype(np.float32)
    source = args.output_dir / "synthetic-stereo.wav"
    write_wav(source, mixture, sample_rate)
    started = time.perf_counter()
    result = separate_file(source, args.output_dir / "stems", checkpoint=checkpoint)
    vocals, rate = load_audio(result["vocals"])
    accompaniment, second_rate = load_audio(result["accompaniment"])
    if rate != sample_rate or second_rate != sample_rate:
        raise RuntimeError("Separator changed the expected sample rate.")
    if vocals.shape != mixture.shape or accompaniment.shape != mixture.shape:
        raise RuntimeError("Separator changed the expected channels or length.")
    error = float(np.max(np.abs(vocals + accompaniment - mixture)))
    if error >= 1e-5:
        raise RuntimeError(f"Stems do not reconstruct the mixture: max error {error}.")
    separation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    classification = classify_file(source, weights=args.weights, start=0.25, duration=1.0)
    if not all(np.isfinite(score) for score in classification["scores"].values()):
        raise RuntimeError("Classifier returned non-finite scores.")
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "soundfile", "torch", "tensorflow", "keras", "librosa", "h5py")},
        "weights": [checksum(args.weights), checksum(str(checkpoint) + ".index")]
        + [checksum(path) for path in checkpoint.parent.iterdir()
           if path.name.startswith(checkpoint.name + ".data-") and path.is_file()],
        "separation": {"passed": True, "frames": len(vocals), "channels": 2,
                       "sample_rate": rate, "mixture_max_absolute_error": error,
                       "seconds": separation_seconds},
        "classification": {"passed": True, "finite_scores": True,
                           "seconds": time.perf_counter() - started},
        "limitation": "Synthetic tone compatibility check only; no model-quality claim.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

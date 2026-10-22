"""Exercise the fixed MIMO checkpoint on original generated audio; no downloads."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from voice_extraction.audio import write_wav
from voice_extraction.modern import separate_mimo_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True, help="A new directory.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    rate = 16000
    time = np.arange(2 * rate, dtype=np.float64) / rate
    samples = np.column_stack([
        0.2 * np.sin(2 * np.pi * 440 * time),
        0.15 * np.sin(2 * np.pi * 660 * time),
    ]).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="voice-mimo-smoke-") as temporary:
        mixture = Path(temporary) / "original-tone.wav"
        write_wav(mixture, samples, rate)
        report = separate_mimo_file(
            mixture, args.output_dir, weights=args.weights, device=args.device,
            chunk_seconds=0.5, overlap=0.25,
        )
    report["fixture"] = {
        "kind": "original generated stereo 440/660 Hz tones",
        "seconds": 2,
        "source_sample_rate": rate,
        "quality_benchmark": False,
    }
    if report["frames"] != 88200 or report["channels"] != 2:
        raise RuntimeError("Unexpected resampled length/channel count.")
    if report["mixture_max_abs_error"] > 1e-5:
        raise RuntimeError("Mixture reconstruction check failed.")
    record = Path(args.output_dir) / "compatibility.json"
    record.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()

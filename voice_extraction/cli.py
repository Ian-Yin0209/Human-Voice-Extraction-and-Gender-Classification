"""Command-line interface. Basic commands do not import model runtimes."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare audio and run optional local voice models.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect a mono/stereo audio file.")
    inspect.add_argument("input")
    inspect.add_argument("--json", action="store_true", help="Emit machine-readable metadata.")
    prepare = commands.add_parser("prepare", help="Crop, resample, and optionally normalize to WAV.")
    prepare.add_argument("input")
    prepare.add_argument("output")
    prepare.add_argument("--start", type=float, default=0.0, help="Start in seconds (default: 0).")
    prepare.add_argument("--duration", type=float, help="Duration in seconds, clipped to EOF.")
    prepare.add_argument("--sample-rate", type=int, default=44100)
    prepare.add_argument("--mono", action="store_true")
    prepare.add_argument("--normalize", action="store_true", help="Normalize non-silent peak to 0.99.")
    prepare.add_argument("--force", action="store_true")
    separate = commands.add_parser("separate", help="Separate vocals/accompaniment with local weights.")
    separate.add_argument("input")
    separate.add_argument("output_dir")
    separate.add_argument("--checkpoint", required=True, help="TensorFlow checkpoint prefix.")
    separate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    separate.add_argument("--force", action="store_true")
    classify = commands.add_parser("classify", help="Run the optional legacy binary voice-label model.")
    classify.add_argument("input")
    classify.add_argument("--weights", required=True, help="Path to upstream-compatible HDF5 weights.")
    classify.add_argument("--start", type=float, default=0.0)
    classify.add_argument("--duration", type=float)
    classify.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            from .audio import inspect_file
            result = inspect_file(args.input)
        elif args.command == "prepare":
            from .audio import prepare_file
            result = prepare_file(
                args.input, args.output, start=args.start, duration=args.duration,
                sample_rate=args.sample_rate, mono=args.mono, normalize=args.normalize, force=args.force,
            )
        elif args.command == "separate":
            from .pipeline import separate_file
            result = separate_file(
                args.input, args.output_dir, checkpoint=args.checkpoint,
                device=args.device, force=args.force,
            )
        else:
            from .classification import classify_file
            result = classify_file(
                args.input, weights=args.weights, start=args.start, duration=args.duration,
            )
        if getattr(args, "json", False):
            print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        else:
            for key, value in result.items():
                print(f"{key}: {value}")
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

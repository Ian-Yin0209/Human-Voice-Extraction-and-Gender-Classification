"""Command-line interface. Basic commands do not import model runtimes."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _emit(text: str, stream) -> None:
    """Keep diagnostics printable when a redirected Windows stream uses ANSI."""
    encoding = getattr(stream, "encoding", None)
    if encoding:
        text = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(text, file=stream)


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
    models = commands.add_parser("models", help="Show the pinned model catalog without a network call.")
    models.add_argument("--json", action="store_true")
    fetch = commands.add_parser("fetch-model", help="Explicitly fetch/check the fixed upstream model.")
    fetch.add_argument("model", choices=("mimo-scnet",))
    fetch.add_argument("--models-dir", default="models")
    fetch.add_argument("--archive", help="Install from the documented local ZIP without a download.")
    fetch.add_argument("--json", action="store_true")
    separate = commands.add_parser("separate", help="Separate local audio into stems.")
    separate.add_argument("input")
    separate.add_argument("output_dir")
    separate.add_argument("--backend", choices=("spleeter", "mimo"), default="spleeter")
    separate.add_argument("--checkpoint", help="Legacy Spleeter TensorFlow checkpoint prefix.")
    separate.add_argument("--weights", help="Fixed MIMO-SCNet .pth file, obtained separately.")
    separate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    separate.add_argument("--force", action="store_true")
    separate.add_argument("--chunk-seconds", type=float, default=6.0, help="MIMO: 0.1-30 seconds.")
    separate.add_argument("--overlap", type=float, default=0.25, help="MIMO: fraction 0-0.75.")
    separate.add_argument("--iterations", type=int, choices=(1, 2), default=2, help="MIMO refinement steps.")
    separate.add_argument("--json", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Compare aligned reference/estimate with SI-SDR.")
    evaluate.add_argument("reference")
    evaluate.add_argument("estimate")
    evaluate.add_argument("--json", action="store_true")
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
        elif args.command == "models":
            from .model_registry import model_catalog
            result = model_catalog()
        elif args.command == "fetch-model":
            from .model_registry import fetch_model
            result = fetch_model(args.models_dir, archive=args.archive)
        elif args.command == "evaluate":
            from .evaluation import evaluate_files
            result = evaluate_files(args.reference, args.estimate)
        elif args.command == "separate":
            if args.backend == "mimo":
                if not args.weights or args.checkpoint or args.force:
                    raise ValueError(
                        "MIMO needs --weights and a new output directory; --checkpoint/--force "
                        "are only for the legacy Spleeter backend. Run fetch-model mimo-scnet first."
                    )
                from .modern import separate_mimo_file
                result = separate_mimo_file(
                    args.input, args.output_dir, weights=args.weights, device=args.device,
                    chunk_seconds=args.chunk_seconds, overlap=args.overlap, iterations=args.iterations,
                )
            else:
                if not args.checkpoint or args.weights:
                    raise ValueError(
                        "Spleeter needs --checkpoint PREFIX. For MIMO use --backend mimo --weights FILE."
                    )
                if args.chunk_seconds != 6.0 or args.overlap != 0.25 or args.iterations != 2:
                    raise ValueError("Chunk/overlap/iteration settings apply only to --backend mimo.")
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
            _emit(json.dumps(result, ensure_ascii=True, allow_nan=False), sys.stdout)
        else:
            for key, value in result.items():
                _emit(f"{key}: {value}", sys.stdout)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        _emit(f"error: {exc}", sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

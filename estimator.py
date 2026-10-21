"""Compatibility wrapper for callers of the original estimate(input, output)."""

from pathlib import Path


def estimate(input, output, checkpoint="./checkpoints/2stems/model"):
    from voice_extraction.pipeline import separate_file

    result = separate_file(input, output, checkpoint=checkpoint)
    return str(Path(result["vocals"]))

"""Compatibility entry point; prefer `voice-extract` or `python -m voice_extraction`."""

from voice_extraction.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

"""Generate a small original stereo test signal, not a speech/quality benchmark."""

import argparse
from pathlib import Path
import struct
import wave
import math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sample_rate = 16000
    frames = bytearray()
    for index in range(2 * sample_rate):
        left = int(8000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        right = int(4000 * math.sin(2 * math.pi * 660 * index / sample_rate))
        frames.extend(struct.pack("<hh", left, right))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as destination:
        with wave.open(destination, "wb") as audio:
            audio.setparams((2, 2, sample_rate, 0, "NONE", "not compressed"))
            audio.writeframes(frames)


if __name__ == "__main__":
    main()

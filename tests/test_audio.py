"""Audio operations exercised with small, generated WAV files."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from voice_extraction.audio import (
    inspect_file,
    load_audio,
    prepare_file,
    resample_audio,
    select_segment,
)


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "中文 音频"
        self.directory.mkdir()
        self.input = self.directory / "原始 双声道.wav"
        self.rate = 16000
        time = np.arange(self.rate, dtype=np.float64) / self.rate
        samples = np.column_stack(
            (0.4 * np.sin(2 * np.pi * 440 * time),
             0.2 * np.sin(2 * np.pi * 660 * time))
        )
        sf.write(self.input, samples, self.rate, subtype="PCM_16")

    def test_load_preserves_stereo_and_returns_float32(self):
        samples, sample_rate = load_audio(self.input)
        self.assertEqual(sample_rate, self.rate)
        self.assertEqual(samples.shape, (self.rate, 2))
        self.assertEqual(samples.dtype, np.float32)
        self.assertTrue(np.isfinite(samples).all())
        self.assertFalse(np.allclose(samples[:, 0], samples[:, 1]))

    def test_inspect_reports_audio_properties(self):
        metadata = inspect_file(self.input)
        self.assertEqual(metadata["sample_rate"], self.rate)
        self.assertEqual(metadata["channels"], 2)
        self.assertEqual(metadata["frames"], self.rate)
        self.assertAlmostEqual(metadata["duration_seconds"], 1.0)
        self.assertAlmostEqual(metadata["peak"], 0.4, delta=0.0001)

    def test_segment_uses_seconds_and_preserves_both_channels(self):
        samples, _ = load_audio(self.input)
        segment = select_segment(samples, self.rate, start=0.2, duration=0.3)
        self.assertEqual(segment.shape, (4800, 2))
        np.testing.assert_array_equal(segment, samples[3200:8000])

    def test_segment_with_no_duration_keeps_remaining_audio(self):
        samples, _ = load_audio(self.input)
        segment = select_segment(samples, self.rate, start=0.75)
        np.testing.assert_array_equal(segment, samples[12000:])

    def test_huge_finite_duration_is_clipped_to_the_audio_end(self):
        samples, _ = load_audio(self.input)
        segment = select_segment(samples, self.rate, start=0.25, duration=1e308)
        np.testing.assert_array_equal(segment, samples[4000:])

    def test_resampling_changes_frame_count_and_preserves_channels(self):
        samples, _ = load_audio(self.input)
        resampled = resample_audio(samples, self.rate, 22050)
        self.assertEqual(resampled.shape, (22050, 2))
        self.assertTrue(np.isfinite(resampled).all())

    def test_prepare_crops_and_resamples_non_44100_input(self):
        output = self.directory / "裁剪 重采样.wav"
        metadata = prepare_file(
            self.input, output, start=0.2, duration=0.3, sample_rate=22050
        )
        self.assertIsInstance(metadata, dict)
        audio, sample_rate = sf.read(output, always_2d=True)
        self.assertEqual(sample_rate, 22050)
        self.assertEqual(audio.shape, (6615, 2))
        self.assertEqual(sf.info(output).format, "WAV")

    def test_prepare_defaults_to_44100_and_keeps_stereo(self):
        output = self.directory / "默认 输出.wav"
        prepare_file(self.input, output)
        info = sf.info(output)
        self.assertEqual(info.samplerate, 44100)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.frames, 44100)

    def test_mono_preparation_downmixes_channels(self):
        output = self.directory / "单声道.wav"
        prepare_file(self.input, output, sample_rate=self.rate, mono=True)
        original, _ = sf.read(self.input, always_2d=True)
        mono, sample_rate = sf.read(output, always_2d=True)
        self.assertEqual(sample_rate, self.rate)
        self.assertEqual(mono.shape, (self.rate, 1))
        np.testing.assert_allclose(mono[:, 0], original.mean(axis=1), atol=0.0001)

    def test_normalizing_silence_produces_finite_silent_audio(self):
        silence = self.directory / "静音.wav"
        output = self.directory / "归一化 静音.wav"
        sf.write(silence, np.zeros((8000, 2), dtype=np.float32), 8000)
        prepare_file(silence, output, normalize=True, sample_rate=8000)
        audio, _ = sf.read(output, always_2d=True)
        self.assertEqual(audio.shape, (8000, 2))
        self.assertTrue(np.isfinite(audio).all())
        np.testing.assert_array_equal(audio, 0)

    def test_existing_output_is_preserved_without_force(self):
        output = self.directory / "已存在.wav"
        output.write_bytes(b"existing file must survive")
        with self.assertRaises((FileExistsError, ValueError)):
            prepare_file(self.input, output)
        self.assertEqual(output.read_bytes(), b"existing file must survive")

    def test_force_allows_replacing_a_different_output(self):
        output = self.directory / "可替换.wav"
        output.write_bytes(b"old content")
        prepare_file(self.input, output, duration=0.25, sample_rate=8000, force=True)
        self.assertEqual(sf.info(output).frames, 2000)

    def test_input_cannot_be_overwritten_even_with_force(self):
        original = self.input.read_bytes()
        with self.assertRaises(ValueError):
            prepare_file(self.input, self.input, force=True)
        self.assertEqual(self.input.read_bytes(), original)

    def test_invalid_segment_times_are_rejected(self):
        samples, _ = load_audio(self.input)
        cases = [
            {"start": -0.1},
            {"start": float("nan")},
            {"start": float("inf")},
            {"start": 1.0},
            {"start": 2.0},
            {"start": 1e308},
            {"duration": -0.1},
            {"duration": 0.0},
            {"duration": float("nan")},
            {"duration": float("inf")},
        ]
        for kwargs in cases:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    select_segment(samples, self.rate, **kwargs)

    def test_empty_audio_is_rejected(self):
        empty = self.directory / "空音频.wav"
        sf.write(empty, np.empty((0, 2), dtype=np.float32), self.rate)
        with self.assertRaises(ValueError):
            load_audio(empty)


if __name__ == "__main__":
    unittest.main()

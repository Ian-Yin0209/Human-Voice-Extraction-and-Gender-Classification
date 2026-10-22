"""Analytic SI-SDR checks using generated reference/estimate audio files."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

from voice_extraction.evaluation import evaluate_files


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "参考 音频"
        self.directory.mkdir()
        self.reference = self.directory / "原始 人声.wav"
        self.estimate = self.directory / "估计 人声.wav"
        # Zero-mean, equal-energy vectors with exactly zero cross-correlation.
        self.signal = np.tile([1.0, -1.0, 1.0, -1.0], 256).astype(np.float32)
        self.interference = np.tile([1.0, 1.0, -1.0, -1.0], 256).astype(np.float32)

    def write_pair(self, reference, estimate, reference_rate=16000, estimate_rate=16000):
        sf.write(self.reference, reference, reference_rate, subtype="FLOAT")
        sf.write(self.estimate, estimate, estimate_rate, subtype="FLOAT")

    def test_mono_orthogonal_interference_has_analytic_score(self):
        self.write_pair(self.signal, self.signal + 0.5 * self.interference)
        result = evaluate_files(self.reference, self.estimate)
        self.assertAlmostEqual(result["si_sdr_db"], 10 * math.log10(4), places=10)
        self.assertIsNone(result["reason"])
        self.assertEqual(result["channels"], 1)
        self.assertEqual(result["frames"], len(self.signal))
        self.assertTrue(result["zero_mean"])
        self.assertEqual(result["alignment"], "none")

    def test_score_is_invariant_to_nonzero_estimate_gain(self):
        expected = 10 * math.log10(4)
        for gain in [0.25, 2.0, -4.0]:
            with self.subTest(gain=gain):
                self.write_pair(self.signal, gain * (self.signal + 0.5 * self.interference))
                result = evaluate_files(self.reference, self.estimate)
                self.assertAlmostEqual(result["si_sdr_db"], expected, places=10)

    def test_metric_removes_channel_dc_offsets(self):
        self.write_pair(self.signal + 4.0, self.signal + 0.5 * self.interference - 8.0)
        result = evaluate_files(self.reference, self.estimate)
        self.assertAlmostEqual(result["si_sdr_db"], 10 * math.log10(4), places=10)

    def test_stereo_overall_is_mean_of_independent_channel_db_scores(self):
        reference = np.column_stack([self.signal, self.signal])
        estimate = np.column_stack([
            self.signal + 0.5 * self.interference,
            2 * self.signal + 2 * self.interference,
        ])
        self.write_pair(reference, estimate)
        result = evaluate_files(self.reference, self.estimate)
        scores = [channel["si_sdr_db"] for channel in result["per_channel"]]
        self.assertEqual([channel["channel"] for channel in result["per_channel"]], [1, 2])
        self.assertAlmostEqual(scores[0], 10 * math.log10(4), places=10)
        self.assertAlmostEqual(scores[1], 0.0, places=10)
        self.assertAlmostEqual(result["si_sdr_db"], sum(scores) / 2, places=10)
        self.assertEqual(result["aggregation"], "arithmetic_mean_of_channel_db")

    def test_perfect_estimate_returns_null_and_reason_in_strict_json(self):
        for gain in [1.0, 2.0, -0.5]:
            with self.subTest(gain=gain):
                self.write_pair(self.signal, gain * self.signal)
                result = evaluate_files(self.reference, self.estimate)
                self.assertIsNone(result["si_sdr_db"])
                self.assertEqual(result["reason"], "undefined_channel")
                self.assertEqual(result["per_channel"][0]["reason"], "perfect_reconstruction")
                self.assertIsNone(result["per_channel"][0]["si_sdr_db"])
                json.dumps(result, allow_nan=False)

    def test_silence_and_constant_dc_have_explicit_undefined_reasons(self):
        cases = [
            (np.zeros_like(self.signal), self.signal, "reference_zero_variance"),
            (np.ones_like(self.signal), self.signal, "reference_zero_variance"),
            (self.signal, np.zeros_like(self.signal), "estimate_zero_variance"),
            (self.signal, np.ones_like(self.signal), "estimate_zero_variance"),
            (np.zeros_like(self.signal), np.zeros_like(self.signal), "reference_zero_variance"),
        ]
        for reference, estimate, reason in cases:
            with self.subTest(reason=reason, reference_peak=float(np.max(reference))):
                self.write_pair(reference, estimate)
                result = evaluate_files(self.reference, self.estimate)
                self.assertIsNone(result["si_sdr_db"])
                self.assertEqual(result["per_channel"][0]["reason"], reason)
                json.dumps(result, allow_nan=False)

    def test_exactly_orthogonal_estimate_has_null_not_negative_infinity(self):
        self.write_pair(self.signal, self.interference)
        result = evaluate_files(self.reference, self.estimate)
        self.assertIsNone(result["si_sdr_db"])
        self.assertEqual(result["per_channel"][0]["reason"], "orthogonal_estimate")
        json.dumps(result, allow_nan=False)

    def test_one_undefined_stereo_channel_does_not_get_silently_discarded(self):
        reference = np.column_stack([self.signal, np.zeros_like(self.signal)])
        estimate = np.column_stack([self.signal + self.interference, np.zeros_like(self.signal)])
        self.write_pair(reference, estimate)
        result = evaluate_files(self.reference, self.estimate)
        self.assertAlmostEqual(result["per_channel"][0]["si_sdr_db"], 0.0)
        self.assertEqual(result["per_channel"][1]["reason"], "reference_zero_variance")
        self.assertIsNone(result["si_sdr_db"])
        self.assertEqual(result["reason"], "undefined_channel")

    def test_mismatched_sample_rates_are_rejected_without_resampling(self):
        self.write_pair(self.signal, self.signal, reference_rate=16000, estimate_rate=8000)
        with self.assertRaisesRegex(ValueError, "same sample rate"):
            evaluate_files(self.reference, self.estimate)

    def test_mismatched_frame_counts_are_rejected_without_trimming(self):
        self.write_pair(self.signal, self.signal[:-1])
        with self.assertRaisesRegex(ValueError, "same frame count and channel count"):
            evaluate_files(self.reference, self.estimate)

    def test_mismatched_channels_are_rejected_without_downmixing(self):
        self.write_pair(self.signal, np.column_stack([self.signal, self.signal]))
        with self.assertRaisesRegex(ValueError, "same frame count and channel count"):
            evaluate_files(self.reference, self.estimate)

    def test_delay_is_not_automatically_aligned(self):
        # A delay turns this reference into an orthogonal estimate. Auto-alignment
        # would instead turn it into a perfect match and hide the timing error.
        reference = np.tile([1.0, 0.0, -1.0, 0.0], 256).astype(np.float32)
        self.write_pair(reference, np.roll(reference, 1))
        result = evaluate_files(self.reference, self.estimate)
        self.assertEqual(result["per_channel"][0]["reason"], "orthogonal_estimate")

    def test_nonfinite_reference_and_estimate_samples_are_rejected(self):
        for side in ["reference", "estimate"]:
            for nonfinite in [float("nan"), float("inf"), -float("inf")]:
                with self.subTest(side=side, value=nonfinite):
                    reference = self.signal.copy()
                    estimate = self.signal.copy()
                    if side == "reference":
                        reference[0] = nonfinite
                    else:
                        estimate[0] = nonfinite
                    self.write_pair(reference, estimate)
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        evaluate_files(self.reference, self.estimate)

    def test_evaluation_does_not_import_optional_model_dependencies(self):
        self.write_pair(self.signal, self.signal + self.interference)
        script = r"""
import importlib.abc
import sys

class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0].lower() in {'torch', 'tensorflow', 'librosa', 'pyaudio'}:
            raise AssertionError('Evaluation imported optional dependency: ' + fullname)
        return None

sys.meta_path.insert(0, NoModels())
from voice_extraction.evaluation import evaluate_files
result = evaluate_files(sys.argv[1], sys.argv[2])
assert abs(result['si_sdr_db']) < 1e-10
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.reference), str(self.estimate)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

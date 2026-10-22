"""Public CLI behavior, including operation without optional model imports."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Run real commands in a fresh interpreter and fail if optional ML/audio stacks
# are imported. This checks the lightweight commands independently of packages
# that happen to be installed on the test machine.
NO_MODEL_IMPORTS = r"""
import importlib.abc
import runpy
import sys

blocked = {'torch', 'tensorflow', 'librosa', 'pyaudio', 'keras'}

class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0].lower() in blocked:
            raise AssertionError('Optional model dependency imported: ' + fullname)
        return None

sys.meta_path.insert(0, NoModels())
sys.argv = ['voice_extraction'] + sys.argv[1:]
try:
    runpy.run_module('voice_extraction', run_name='__main__')
except SystemExit as result:
    if result.code not in (None, 0):
        raise
loaded = {name.split('.')[0].lower() for name in sys.modules}
if blocked.intersection(loaded):
    raise AssertionError('Optional model dependency was loaded')
"""


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "中文 路径"
        self.directory.mkdir()
        self.input = self.directory / "双声道 输入.wav"
        rate = 16000
        time = np.arange(rate, dtype=np.float64) / rate
        samples = np.column_stack(
            (0.4 * np.sin(2 * np.pi * 440 * time),
             0.2 * np.sin(2 * np.pi * 660 * time))
        )
        sf.write(self.input, samples, rate, subtype="PCM_16")

    def run_cli(self, *arguments, block_models=False):
        invocation = [sys.executable]
        if block_models:
            invocation += ["-c", NO_MODEL_IMPORTS]
        else:
            invocation += ["-m", "voice_extraction"]
        return subprocess.run(
            invocation + [str(argument) for argument in arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )

    def assert_cli_error(self, result):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertTrue(result.stderr.strip(), "An error needs a readable message")
        self.assertNotIn("traceback", (result.stdout + result.stderr).lower())

    def test_all_help_commands_work_without_model_dependencies(self):
        for arguments in [
            ("--help",),
            ("inspect", "--help"),
            ("prepare", "--help"),
            ("separate", "--help"),
            ("classify", "--help"),
            ("models", "--help"),
            ("fetch-model", "--help"),
            ("evaluate", "--help"),
        ]:
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments, block_models=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_model_catalog_is_offline_and_lightweight(self):
        result = self.run_cli("models", "--json", block_models=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        model = json.loads(result.stdout)
        self.assertEqual(model["id"], "mimo-scnet")
        self.assertEqual(model["stems"], ["vocals", "bass", "drums", "other"])
        self.assertFalse(model["weights_bundled"])

    def test_evaluate_json_works_without_model_dependencies(self):
        result = self.run_cli("evaluate", self.input, self.input, "--json", block_models=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        scores = json.loads(result.stdout)
        self.assertIsNone(scores["si_sdr_db"])
        self.assertEqual(scores["per_channel"][0]["reason"], "perfect_reconstruction")

    def test_mimo_requires_explicit_weights_and_new_output_directory(self):
        result = self.run_cli("separate", self.input, self.directory / "stems", "--backend", "mimo")
        self.assert_cli_error(result)
        self.assertIn("--weights", result.stderr)

    def test_inspect_json_works_without_model_dependencies(self):
        result = self.run_cli("inspect", self.input, "--json", block_models=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metadata = json.loads(result.stdout)
        self.assertEqual(metadata["sample_rate"], 16000)
        self.assertEqual(metadata["channels"], 2)
        self.assertEqual(metadata["frames"], 16000)
        self.assertAlmostEqual(metadata["duration_seconds"], 1.0)
        self.assertAlmostEqual(metadata["peak"], 0.4, delta=0.0001)

    def test_prepare_crops_and_resamples_without_model_dependencies(self):
        output = self.directory / "准备 输出.wav"
        result = self.run_cli(
            "prepare", self.input, output,
            "--start", "0.2", "--duration", "0.3", "--sample-rate", "22050",
            block_models=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        info = sf.info(output)
        self.assertEqual(info.samplerate, 22050)
        self.assertEqual(info.frames, 6615)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.format, "WAV")

    def test_prepare_mono_and_normalize_are_available(self):
        output = self.directory / "单声道 归一化.wav"
        result = self.run_cli(
            "prepare", self.input, output, "--mono", "--normalize"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        audio, sample_rate = sf.read(output, always_2d=True)
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(audio.shape, (44100, 1))
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(np.max(np.abs(audio)), 0.95)
        self.assertLessEqual(np.max(np.abs(audio)), 1.0001)

    def test_unicode_paths_work_with_ascii_console_encoding(self):
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "ascii"

        def run_ascii_cli(*arguments):
            return subprocess.run(
                [sys.executable, "-m", "voice_extraction",
                 *[str(argument) for argument in arguments]],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="ascii",
                timeout=45,
            )

        with self.subTest(operation="prepare"):
            output = self.directory / "受限编码 输出.wav"
            result = run_ascii_cli(
                "prepare", self.input, output,
                "--duration", "0.25", "--sample-rate", "16000",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(result.stdout.strip())
            actual, sample_rate = sf.read(output, always_2d=True)
            original, _ = sf.read(self.input, always_2d=True)
            self.assertEqual(sample_rate, 16000)
            self.assertEqual(actual.shape, (4000, 2))
            np.testing.assert_allclose(actual, original[:4000], atol=1e-6)

        with self.subTest(operation="missing input"):
            output = self.directory / "不应生成 输出.wav"
            result = run_ascii_cli("prepare", self.directory / "不存在 输入.wav", output)
            self.assert_cli_error(result)
            self.assertFalse(output.exists())

    def test_missing_input_reports_error_without_traceback(self):
        result = self.run_cli("inspect", self.directory / "不存在.wav", "--json")
        self.assert_cli_error(result)

    def test_invalid_time_arguments_report_error_without_traceback(self):
        cases = [
            ("--start", "-1"),
            ("--start", "nan"),
            ("--start", "inf"),
            ("--start", "1e308"),
            ("--duration", "0"),
            ("--duration", "-1"),
            ("--duration", "nan"),
        ]
        for option, value in cases:
            with self.subTest(option=option, value=value):
                output = self.directory / "无效输出.wav"
                result = self.run_cli("prepare", self.input, output, option, value)
                self.assert_cli_error(result)
                self.assertFalse(output.exists())

    def test_huge_finite_duration_is_clipped_without_traceback(self):
        output = self.directory / "超长 时长.wav"
        result = self.run_cli(
            "prepare", self.input, output, "--start", "0.25",
            "--duration", "1e308", "--sample-rate", "16000",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("traceback", (result.stdout + result.stderr).lower())
        self.assertEqual(sf.info(output).frames, 12000)

    def test_empty_wav_reports_error_without_traceback(self):
        empty = self.directory / "空文件.wav"
        sf.write(empty, np.empty((0, 1), dtype=np.float32), 16000)
        result = self.run_cli("inspect", empty, "--json")
        self.assert_cli_error(result)

    def test_existing_output_requires_force(self):
        output = self.directory / "已有 输出.wav"
        output.write_bytes(b"existing content")
        result = self.run_cli("prepare", self.input, output)
        self.assert_cli_error(result)
        self.assertEqual(output.read_bytes(), b"existing content")
        forced = self.run_cli("prepare", self.input, output, "--force")
        self.assertEqual(forced.returncode, 0, forced.stdout + forced.stderr)
        self.assertEqual(sf.info(output).frames, 44100)

    def test_force_does_not_allow_overwriting_the_input(self):
        original = self.input.read_bytes()
        result = self.run_cli("prepare", self.input, self.input, "--force")
        self.assert_cli_error(result)
        self.assertEqual(self.input.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

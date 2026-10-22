"""Modern separator contracts tested with generated audio and NumPy-only inference.

These tests exercise chunk assembly, model boundaries, and publication safety;
they do not download weights or make claims about pretrained separation quality.
"""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voice_extraction import modern


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SIZE = 42_491_051
MODEL_SHA256 = "1d19305440e8439faeb4163d809b2dcaaffacf67f6f926425be581290d6536ee"
STEM_FILES = {
    "vocals": "vocals.wav",
    "bass": "bass.wav",
    "drums": "drums.wav",
    "other": "other.wav",
    "accompaniment": "accompaniment.wav",
}
FRACTIONS = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)


class FractionalInference:
    """Return distinguishable, mixture-consistent stems and record each call."""

    def __init__(self):
        self.calls = []

    def __call__(self, chunk, iterations):
        if not isinstance(chunk, np.ndarray) or chunk.ndim != 2 or chunk.shape[1] != 2:
            raise AssertionError("Inference must receive a frames x stereo NumPy array.")
        self.calls.append((chunk.shape, iterations))
        return FRACTIONS[:, None, None] * chunk[None, :, :]


NO_MODEL_IMPORTS = r"""
import importlib.abc
import json
from pathlib import Path
import sys

blocked = {'torch', 'tensorflow', 'keras', 'torchaudio', 'librosa', 'jax'}

class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0].lower() in blocked:
            raise AssertionError('Optional model runtime imported: ' + fullname)
        return None

sys.meta_path.insert(0, NoModels())
import voice_extraction
import voice_extraction.cli
from voice_extraction import modern

operation, arguments = sys.argv[1], json.loads(sys.argv[2])
if operation == 'import':
    print(json.dumps({'imported': True}))
else:
    try:
        if operation == 'verify':
            modern.verify_weights(Path(arguments['weights']))
        elif operation == 'separate':
            modern.separate_mimo_file(**arguments)
        else:
            raise AssertionError('Unknown operation: ' + operation)
    except (ValueError, OSError) as error:
        print(json.dumps({'message': str(error), 'type': type(error).__name__}))
    else:
        raise AssertionError('Invalid weights unexpectedly reached inference.')

loaded = {name.split('.')[0].lower() for name in sys.modules}
if blocked.intersection(loaded):
    raise AssertionError('An optional model runtime was loaded.')
"""


class ModernSeparatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="voice-modern-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "中文 音频"
        self.directory.mkdir()
        self.input = self.directory / "输入 双声道.wav"
        self.weights = self.directory / "weights.ckpt"
        self.weights.write_bytes(b"deliberately not pretrained weights")
        self.output = self.directory / "分离 输出"
        self.samples = self.make_samples(12_457)
        self.write_input(self.samples)

    @staticmethod
    def make_samples(frames):
        # Nonzero boundaries detect faded/missing first and final overlap regions.
        time = np.arange(frames, dtype=np.float64) / 44100
        return np.column_stack((
            0.2 + 0.17 * np.sin(2 * np.pi * 137 * time),
            -0.3 + 0.11 * np.cos(2 * np.pi * 293 * time),
        )).astype(np.float32)

    def write_input(self, samples, sample_rate=44100):
        sf.write(self.input, samples, sample_rate, subtype="FLOAT")

    def separate(self, inference=None, **arguments):
        inference = FractionalInference() if inference is None else inference
        with patch.object(modern, "verify_weights", return_value=self.weights.resolve()), \
                patch.object(modern, "_load_model", return_value=inference) as loader:
            report = modern.separate_mimo_file(
                self.input, self.output, weights=self.weights, **arguments
            )
        self.assertEqual(loader.call_count, 1)
        return report, inference

    def read_stems(self):
        result = {}
        for name, filename in STEM_FILES.items():
            samples, sample_rate = sf.read(self.output / filename, dtype="float32", always_2d=True)
            self.assertEqual(sample_rate, 44100)
            self.assertTrue(np.isfinite(samples).all(), name)
            result[name] = samples
        return result

    def assert_no_publication_or_staging(self, before):
        self.assertFalse(self.output.exists(), "Failed operations must not publish the final directory.")
        self.assertEqual(set(self.directory.iterdir()), before, "Temporary stage directories must be removed.")

    def test_chunked_overlap_preserves_stem_order_mixture_and_boundaries(self):
        for overlap in (0.0, 0.25, 0.75):
            with self.subTest(overlap=overlap):
                self.output = self.directory / f"overlap-{overlap}"
                report, inference = self.separate(chunk_seconds=0.1, overlap=overlap)
                self.assertGreater(len(inference.calls), 1)
                self.assertTrue(all(0 < shape[0] <= 4410 for shape, _ in inference.calls))
                stems = self.read_stems()
                for index, name in enumerate(("vocals", "bass", "drums", "other")):
                    self.assertEqual(stems[name].shape, self.samples.shape)
                    np.testing.assert_allclose(
                        stems[name], self.samples * FRACTIONS[index], atol=2e-6, rtol=2e-5,
                        err_msg=f"{name}: chunk boundaries must not add gaps or duplicate gain",
                    )
                np.testing.assert_allclose(
                    sum(stems[name] for name in ("vocals", "bass", "drums", "other")),
                    self.samples, atol=2e-6, rtol=2e-5,
                )
                np.testing.assert_allclose(
                    stems["accompaniment"], stems["bass"] + stems["drums"] + stems["other"],
                    atol=2e-6, rtol=2e-5,
                )
                self.assertLessEqual(report["mixture_max_abs_error"], 2e-6)

    def test_short_and_single_frame_clips_keep_their_exact_length(self):
        for frames in (1, 17, 4410, 4411):
            with self.subTest(frames=frames):
                self.output = self.directory / f"short-{frames}"
                samples = self.make_samples(frames)
                self.write_input(samples)
                report, _ = self.separate(chunk_seconds=0.1)
                self.assertEqual(report["frames"], frames)
                stems = self.read_stems()
                np.testing.assert_allclose(stems["vocals"], samples * 0.1, atol=2e-6, rtol=2e-5)
                self.assertTrue(all(stem.shape == (frames, 2) for stem in stems.values()))

    def test_mono_is_resampled_to_44100_and_duplicated_to_stereo(self):
        self.write_input(np.full((1600, 1), 0.3, dtype=np.float32), sample_rate=16000)
        report, inference = self.separate(chunk_seconds=0.1)
        self.assertEqual(report["sample_rate"], 44100)
        self.assertEqual(report["frames"], 4410)
        self.assertEqual(report["channels"], 2)
        stems = self.read_stems()
        for stem in stems.values():
            self.assertEqual(stem.shape, (4410, 2))
            np.testing.assert_array_equal(stem[:, 0], stem[:, 1])
        reconstructed = sum(stems[name] for name in ("vocals", "bass", "drums", "other"))
        np.testing.assert_allclose(reconstructed[200:-200], 0.3, atol=0.001)
        self.assertTrue(all(shape[1] == 2 for shape, _ in inference.calls))

    def test_silent_audio_produces_finite_silent_stems(self):
        self.write_input(np.zeros((29, 2), dtype=np.float32))
        self.separate(chunk_seconds=0.1)
        for stem in self.read_stems().values():
            np.testing.assert_array_equal(stem, 0)

    def test_iterations_are_forwarded_and_report_matches_saved_json(self):
        for iterations in (1, 2):
            with self.subTest(iterations=iterations):
                self.output = self.directory / f"iterations-{iterations}"
                before = set(self.directory.iterdir())
                report, inference = self.separate(chunk_seconds=0.1, iterations=iterations)
                self.assertTrue(all(count == iterations for _, count in inference.calls))
                self.assertEqual(report["iterations"], iterations)
                self.assertEqual(report["model_sha256"], MODEL_SHA256)
                self.assertIsInstance(report["model_id"], str)
                self.assertTrue(report["model_id"].strip())
                self.assertEqual(report["frames"], len(self.samples))
                self.assertEqual(report["channels"], 2)
                self.assertEqual(report["sample_rate"], 44100)
                self.assertTrue(math.isfinite(report["seconds"]))
                self.assertGreaterEqual(report["seconds"], 0)
                self.assertEqual(report["stems"], {
                    name: filename for name, filename in STEM_FILES.items()
                    if name != "accompaniment"
                })
                self.assertEqual(report["accompaniment"], STEM_FILES["accompaniment"])
                on_disk = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(on_disk, report)
                self.assertEqual(set(path.name for path in self.output.iterdir()),
                                 set(STEM_FILES.values()) | {"report.json"})
                self.assertEqual(set(self.directory.iterdir()), before | {self.output})

    def test_requested_device_is_forwarded_to_the_optional_model_loader(self):
        inference = FractionalInference()
        checkpoint = self.weights.resolve()
        with patch.object(modern, "verify_weights", return_value=checkpoint), \
                patch.object(modern, "_load_model", return_value=inference) as loader:
            modern.separate_mimo_file(
                self.input, self.output, weights=self.weights, device="cuda", chunk_seconds=0.1
            )
        loader.assert_called_once_with(checkpoint, "cuda")
        self.assertTrue((self.output / "report.json").is_file())

    def test_numpy_scalar_parameters_produce_json_serializable_python_numbers(self):
        cases = [
            {"iterations": np.int64(1)},
            {"chunk_seconds": np.float32(0.1)},
            {"overlap": np.float32(0.5)},
            {
                "iterations": np.int64(1),
                "chunk_seconds": np.float32(0.1),
                "overlap": np.float32(0.5),
            },
        ]
        for index, arguments in enumerate(cases):
            with self.subTest(arguments=arguments):
                self.output = self.directory / f"numpy-scalars-{index}"
                report, inference = self.separate(**arguments)
                saved = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, report)
                for value in (report, saved):
                    self.assertIs(type(value["iterations"]), int)
                    self.assertIs(type(value["chunk_seconds"]), float)
                    self.assertIs(type(value["overlap"]), float)
                self.assertEqual(report["iterations"], int(arguments.get("iterations", 2)))
                self.assertEqual(report["chunk_seconds"], float(arguments.get("chunk_seconds", 6.0)))
                self.assertEqual(report["overlap"], float(arguments.get("overlap", 0.25)))
                self.assertTrue(all(type(count) is int for _, count in inference.calls))

    def test_existing_output_directory_is_preserved_even_if_empty(self):
        self.output.mkdir()
        with patch.object(modern, "_load_model") as loader:
            with self.assertRaises((ValueError, OSError)):
                modern.separate_mimo_file(self.input, self.output, weights=self.weights)
        loader.assert_not_called()
        self.assertTrue(self.output.is_dir())
        self.assertEqual(list(self.output.iterdir()), [])

    def test_existing_files_are_never_overwritten(self):
        original = self.input.read_bytes()
        self.output.mkdir()
        sentinel = self.output / "vocals.wav"
        sentinel.write_bytes(b"previous vocal must survive")
        with patch.object(modern, "_load_model") as loader:
            with self.assertRaises((ValueError, OSError)):
                modern.separate_mimo_file(self.input, self.output, weights=self.weights)
        loader.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"previous vocal must survive")
        self.assertEqual(self.input.read_bytes(), original)
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_output_cannot_replace_the_input_file(self):
        original = self.input.read_bytes()
        with patch.object(modern, "_load_model") as loader:
            with self.assertRaises((ValueError, OSError)):
                modern.separate_mimo_file(self.input, self.input, weights=self.weights)
        loader.assert_not_called()
        self.assertEqual(self.input.read_bytes(), original)

    def test_output_symlink_does_not_write_into_its_target(self):
        target = self.directory / "existing-target"
        target.mkdir()
        try:
            self.output.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Directory symlinks are unavailable to this test user.")
        with patch.object(modern, "_load_model") as loader:
            with self.assertRaises((ValueError, OSError)):
                modern.separate_mimo_file(self.input, self.output, weights=self.weights)
        loader.assert_not_called()
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(list(target.iterdir()), [])

    def test_dangling_output_symlink_is_preserved(self):
        target = self.directory / "missing-symlink-target"
        try:
            self.output.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Directory symlinks are unavailable to this test user.")
        with patch.object(modern, "_load_model") as loader:
            with self.assertRaises((ValueError, OSError)):
                modern.separate_mimo_file(self.input, self.output, weights=self.weights)
        loader.assert_not_called()
        self.assertTrue(self.output.is_symlink())
        self.assertFalse(target.exists())

    def test_concurrently_created_output_is_not_replaced_or_deleted(self):
        before = set(self.directory.iterdir())
        inference = FractionalInference()
        sentinel = self.output / "other-process.txt"

        def competing_inference(chunk, iterations):
            if not self.output.exists():
                self.output.mkdir()
                sentinel.write_bytes(b"created by a competing process")
            return inference(chunk, iterations)

        with self.assertRaises((ValueError, OSError)):
            self.separate(inference=competing_inference, chunk_seconds=0.1)
        self.assertTrue(self.output.is_dir())
        self.assertEqual(sentinel.read_bytes(), b"created by a competing process")
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.assertEqual(set(self.directory.iterdir()), before | {self.output})

    def test_invalid_parameters_fail_before_model_loading_without_publication(self):
        cases = [
            {"chunk_seconds": 0}, {"chunk_seconds": 0.09}, {"chunk_seconds": 30.1},
            {"chunk_seconds": float("nan")}, {"chunk_seconds": float("inf")},
            {"chunk_seconds": 1e308},
            {"overlap": -0.01}, {"overlap": 0.751},
            {"overlap": float("nan")}, {"overlap": float("inf")},
            {"iterations": 0}, {"iterations": 3}, {"iterations": 1.5}, {"iterations": True},
            {"device": "mps"}, {"device": "cuda:0"},
        ]
        before = set(self.directory.iterdir())
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with patch.object(modern, "verify_weights", return_value=self.weights), \
                        patch.object(modern, "_load_model") as loader:
                    with self.assertRaises((ValueError, OSError)):
                        modern.separate_mimo_file(
                            self.input, self.output, weights=self.weights, **arguments
                        )
                loader.assert_not_called()
                self.assert_no_publication_or_staging(before)

    def test_invalid_audio_fails_before_model_loading(self):
        invalid = (
            np.empty((0, 2), dtype=np.float32),
            np.full((29, 2), float("nan"), dtype=np.float32),
            np.full((29, 2), float("inf"), dtype=np.float32),
            np.zeros((29, 3), dtype=np.float32),
        )
        for samples in invalid:
            with self.subTest(shape=samples.shape):
                self.write_input(samples)
                before = set(self.directory.iterdir())
                with patch.object(modern, "verify_weights", return_value=self.weights), \
                        patch.object(modern, "_load_model") as loader:
                    with self.assertRaises((ValueError, OSError)):
                        modern.separate_mimo_file(self.input, self.output, weights=self.weights)
                loader.assert_not_called()
                self.assert_no_publication_or_staging(before)

    def test_model_load_failure_does_not_publish_or_leave_staging(self):
        before = set(self.directory.iterdir())
        with patch.object(modern, "verify_weights", return_value=self.weights), \
                patch.object(modern, "_load_model", side_effect=RuntimeError("cannot load model")):
            with self.assertRaisesRegex(RuntimeError, "cannot load"):
                modern.separate_mimo_file(self.input, self.output, weights=self.weights)
        self.assert_no_publication_or_staging(before)

    def test_mid_inference_failure_does_not_publish_or_leave_staging(self):
        base = FractionalInference()

        def failing_inference(chunk, iterations):
            if base.calls:
                raise RuntimeError("second chunk failed")
            return base(chunk, iterations)

        before = set(self.directory.iterdir())
        with self.assertRaisesRegex(RuntimeError, "second chunk"):
            self.separate(inference=failing_inference, chunk_seconds=0.1)
        self.assert_no_publication_or_staging(before)

    def test_invalid_inference_shapes_or_nonfinite_stems_are_rejected(self):
        cases = {
            "three stems": lambda n: np.zeros((3, n, 2), dtype=np.float32),
            "wrong length": lambda n: np.zeros((4, n + 1, 2), dtype=np.float32),
            "mono stems": lambda n: np.zeros((4, n, 1), dtype=np.float32),
            "NaN": lambda n: np.full((4, n, 2), float("nan"), dtype=np.float32),
            "infinity": lambda n: np.full((4, n, 2), float("inf"), dtype=np.float32),
        }
        before = set(self.directory.iterdir())
        for name, build in cases.items():
            with self.subTest(output=name):
                def bad_inference(chunk, iterations, build=build):
                    return build(len(chunk))

                with self.assertRaises((ValueError, RuntimeError)):
                    self.separate(inference=bad_inference, chunk_seconds=0.1)
                self.assert_no_publication_or_staging(before)

    def test_write_failure_does_not_publish_or_leave_staging(self):
        real_write = modern.write_wav
        writes = []

        def failing_write(*arguments, **keywords):
            writes.append(Path(arguments[0]))
            if len(writes) == 2:
                raise OSError("simulated disk write failure")
            return real_write(*arguments, **keywords)

        before = set(self.directory.iterdir())
        with patch.object(modern, "write_wav", side_effect=failing_write):
            with self.assertRaisesRegex(OSError, "disk write failure"):
                self.separate(chunk_seconds=0.1)
        self.assertEqual(len(writes), 2)
        self.assertTrue(all(path.parent != self.output for path in writes))
        self.assert_no_publication_or_staging(before)

    def test_partial_publication_failure_rolls_back_the_new_output_directory(self):
        original = self.input.read_bytes()
        before = set(self.directory.iterdir())
        rename = Path.rename
        attempted = []

        def interrupted_rename(source, target):
            target = Path(target)
            if target.parent == self.output:
                attempted.append(target.name)
                if target.name == "drums.wav":
                    raise OSError("simulated publication failure")
            return rename(source, target)

        with patch.object(Path, "rename", new=interrupted_rename):
            with self.assertRaisesRegex(OSError, "publication failure"):
                self.separate(chunk_seconds=0.1)
        self.assertEqual(attempted, ["vocals.wav", "bass.wav", "drums.wav"])
        self.assertNotIn("report.json", attempted)
        self.assertEqual(self.input.read_bytes(), original)
        self.assert_no_publication_or_staging(before)


class ModernWeightBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="modern-weight-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.bad_weights = self.directory / "corrupt.ckpt"
        self.bad_weights.write_bytes(b"not a model")

    def test_real_weight_verification_rejects_missing_or_wrong_sized_files(self):
        for path in (self.directory / "missing.ckpt", self.bad_weights, self.directory):
            with self.subTest(path=path):
                with self.assertRaises((ValueError, OSError)) as error:
                    modern.verify_weights(path)
                self.assertTrue(str(error.exception).strip())

    def test_real_hash_verification_rejects_a_correctly_sized_corrupt_file(self):
        # Sparse zeros cover actual streaming SHA verification without downloading
        # or allocating a real 42 MB model array in the test process.
        with self.bad_weights.open("wb") as stream:
            stream.seek(MODEL_SIZE - 1)
            stream.write(b"\0")
        self.assertEqual(self.bad_weights.stat().st_size, MODEL_SIZE)
        with self.assertRaises((ValueError, OSError)) as error:
            modern.verify_weights(self.bad_weights)
        self.assertTrue(str(error.exception).strip())

    def guarded_call(self, operation, **arguments):
        result = subprocess.run(
            [sys.executable, "-c", NO_MODEL_IMPORTS, operation, json.dumps(arguments)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_base_cli_and_modern_module_import_without_model_runtimes(self):
        self.assertTrue(self.guarded_call("import")["imported"])

    def test_corrupt_weights_are_rejected_without_importing_model_runtimes(self):
        self.assertTrue(self.guarded_call("verify", weights=str(self.bad_weights))["message"])
        source = self.directory / "input.wav"
        sf.write(source, np.zeros((17, 2), dtype=np.float32), 44100)
        output = self.directory / "output"
        error = self.guarded_call(
            "separate", input_path=str(source), output_dir=str(output), weights=str(self.bad_weights)
        )
        self.assertTrue(error["message"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

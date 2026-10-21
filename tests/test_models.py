"""Optional-model validation must fail clearly before loading a model stack."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from voice_extraction.classification import scores_from_probability
from voice_extraction.pipeline import validate_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# These are real public API calls in an isolated process. An import guard proves
# that input validation does not depend on any installed ML framework.
VALIDATE_WITHOUT_MODELS = r"""
import importlib.abc
import json
import sys

class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0].lower() in {
            'torch', 'tensorflow', 'keras', 'librosa', 'pyaudio'
        }:
            raise AssertionError('Model import before input validation: ' + fullname)
        return None

sys.meta_path.insert(0, NoModels())
from voice_extraction.classification import classify_file
from voice_extraction.pipeline import separate_file

operation, arguments = sys.argv[1], json.loads(sys.argv[2])
try:
    if operation == 'classify':
        classify_file(**arguments)
    elif operation == 'separate':
        separate_file(**arguments)
    else:
        raise AssertionError('Unknown test operation: ' + operation)
except (ValueError, FileExistsError) as error:
    print(json.dumps({'message': str(error), 'type': type(error).__name__}))
else:
    raise AssertionError('Invalid input unexpectedly reached successful inference')
"""


class ModelBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "模型 边界"
        self.directory.mkdir()
        self.input = self.directory / "输入 音频.wav"
        sf.write(self.input, np.zeros((8000, 2), dtype=np.float32), 8000)
        self.weights = self.directory / "分类 权重.h5"
        self.weights.touch()
        self.checkpoint = self.directory / "separator"
        Path(str(self.checkpoint) + ".index").touch()
        Path(str(self.checkpoint) + ".data-00000-of-00001").touch()

    def validation_error_without_models(self, operation, **arguments):
        result = subprocess.run(
            [sys.executable, "-c", VALIDATE_WITHOUT_MODELS, operation,
             json.dumps(arguments)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        error = json.loads(result.stdout)
        self.assertTrue(error["message"].strip())
        return error["message"]

    def test_probability_maps_original_binary_labels_and_scores(self):
        for probability, label in [
            (0.0, "female"), (0.25, "female"), (0.5, "female"),
            (0.75, "male"), (1.0, "male"),
        ]:
            with self.subTest(probability=probability):
                result = scores_from_probability(probability)
                self.assertEqual(result["label"], label)
                self.assertAlmostEqual(result["scores"]["male"], probability)
                self.assertAlmostEqual(result["scores"]["female"], 1 - probability)
                self.assertAlmostEqual(sum(result["scores"].values()), 1.0)

    def test_invalid_probabilities_are_rejected(self):
        for probability in [-0.01, 1.01, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(ValueError, "invalid score"):
                    scores_from_probability(probability)

    def test_missing_classifier_weights_fail_before_model_imports(self):
        missing = self.directory / "缺少 权重.h5"
        message = self.validation_error_without_models(
            "classify", input_path=str(self.input), weights=str(missing)
        )
        self.assertIn("weights", message.lower())
        self.assertIn(str(missing), message)

    def test_wrong_classifier_weight_suffix_fails_before_model_imports(self):
        unsupported = self.directory / "weights.bin"
        unsupported.touch()
        message = self.validation_error_without_models(
            "classify", input_path=str(self.input), weights=str(unsupported)
        )
        self.assertIn(".h5", message)

    def test_invalid_classification_start_fails_before_model_imports(self):
        for start in [-0.1, float("nan"), float("inf"), 1.0, 2.0, 1e308]:
            with self.subTest(start=start):
                message = self.validation_error_without_models(
                    "classify", input_path=str(self.input), weights=str(self.weights),
                    start=start,
                )
                self.assertIn("start", message.lower())

    def test_missing_separator_checkpoint_fails_before_model_imports(self):
        output = self.directory / "输出"
        message = self.validation_error_without_models(
            "separate", input_path=str(self.input), output_dir=str(output),
            checkpoint=str(self.directory / "missing-checkpoint"),
        )
        self.assertIn("checkpoint", message.lower())
        self.assertFalse(output.exists())

    def test_existing_separator_outputs_are_preserved_before_model_imports(self):
        for filename in ["vocals.wav", "accompaniment.wav"]:
            with self.subTest(filename=filename):
                output = self.directory / (filename + " output")
                output.mkdir()
                existing = output / filename
                existing.write_bytes(b"previous output must survive")
                message = self.validation_error_without_models(
                    "separate", input_path=str(self.input), output_dir=str(output),
                    checkpoint=str(self.checkpoint),
                )
                self.assertIn("exists", message.lower())
                self.assertEqual(existing.read_bytes(), b"previous output must survive")
                self.assertEqual(list(output.iterdir()), [existing])

    def test_separation_cannot_replace_its_input_even_with_force(self):
        for filename in ["vocals.wav", "accompaniment.wav"]:
            with self.subTest(filename=filename):
                output = self.directory / (filename + " same input")
                output.mkdir()
                source = output / filename
                sf.write(source, np.zeros((8000, 2), dtype=np.float32), 8000)
                original = source.read_bytes()
                message = self.validation_error_without_models(
                    "separate", input_path=str(source), output_dir=str(output),
                    checkpoint=str(self.checkpoint), force=True,
                )
                self.assertIn("different", message.lower())
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(list(output.iterdir()), [source])

    def test_checkpoint_requires_both_index_and_data(self):
        for index_exists, data_exists in [(False, False), (True, False), (False, True)]:
            with self.subTest(index=index_exists, data=data_exists):
                prefix = self.directory / f"partial-{index_exists}-{data_exists}"
                if index_exists:
                    Path(str(prefix) + ".index").touch()
                if data_exists:
                    Path(str(prefix) + ".data-00000-of-00001").touch()
                with self.assertRaisesRegex(ValueError, "checkpoint.*incomplete"):
                    validate_checkpoint(prefix)

    def test_checkpoint_requires_all_declared_data_shards(self):
        prefix = self.directory / "missing-second-shard"
        Path(str(prefix) + ".index").touch()
        Path(str(prefix) + ".data-00000-of-00002").touch()
        with self.assertRaisesRegex(ValueError, "checkpoint.*incomplete"):
            validate_checkpoint(prefix)

    def test_checkpoint_prefix_can_contain_literal_glob_characters(self):
        prefix = self.directory / "model[1]"
        Path(str(prefix) + ".index").touch()
        Path(str(prefix) + ".data-00000-of-00001").touch()
        self.assertEqual(validate_checkpoint(prefix), prefix.resolve())


if __name__ == "__main__":
    unittest.main()

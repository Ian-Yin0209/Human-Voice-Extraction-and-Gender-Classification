"""Real archive/hash boundaries with small fixtures and a mocked HTTP transport."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
from http.client import HTTPResponse, IncompleteRead
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError
import zipfile

from voice_extraction import cli, model_registry as registry


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "模型 缓存测试"
        self.directory.mkdir()
        # Keep even a hypothetical ../ traversal within this test's temp root.
        self.models = self.directory / "隔离" / "models"
        self.member_name = "release/checkpoint/backbone.pth"
        self.weight_name = "fixture-small.pth"
        self.payload = b"verified synthetic checkpoint bytes\x00\x01\x02" * 8
        self.archive_bytes = self.make_zip({self.member_name: self.payload})
        constants = patch.multiple(
            registry,
            WEIGHT_NAME=self.weight_name,
            WEIGHT_BYTES=len(self.payload),
            WEIGHT_SHA256=hashlib.sha256(self.payload).hexdigest(),
            ARCHIVE_BYTES=len(self.archive_bytes),
            ARCHIVE_SHA256=hashlib.sha256(self.archive_bytes).hexdigest(),
            ARCHIVE_URL="https://example.invalid/verified-fixture.zip",
            MEMBER_NAME=self.member_name,
        )
        constants.start()
        self.addCleanup(constants.stop)

    @staticmethod
    def make_zip(entries):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, payload in entries.items():
                bundle.writestr(name, payload)
        return stream.getvalue()

    @staticmethod
    def truncated_chunked_response():
        class SocketFixture:
            def makefile(self, *args):
                return io.BytesIO(
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                    b"8\r\nabcdefgh\r\n8\r\nabc"
                )

        # Use the real HTTP parser: the second chunk ends before its declared
        # length, so response.read() raises IncompleteRead rather than OSError.
        response = HTTPResponse(SocketFixture())
        response.begin()
        return response

    def pin_archive(self, data):
        """Authenticate this fixture ZIP; keep expected weight bytes unchanged."""
        return patch.multiple(
            registry,
            ARCHIVE_BYTES=len(data),
            ARCHIVE_SHA256=hashlib.sha256(data).hexdigest(),
        )

    def local_archive(self, data):
        path = self.directory / "本地 模型.zip"
        path.write_bytes(data)
        return path

    def assert_no_published_or_staged_files(self):
        self.assertFalse((self.models / self.weight_name).exists())
        if self.models.exists():
            self.assertEqual(list(self.models.iterdir()), [], "A failed fetch left staged files")

    def test_catalog_is_available_without_network_or_optional_model_imports(self):
        script = r"""
import importlib.abc
import json
import sys
from unittest.mock import patch

class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0].lower() in {'torch', 'tensorflow', 'librosa', 'requests'}:
            raise AssertionError('Catalog imported optional dependency: ' + fullname)
        return None

sys.meta_path.insert(0, NoModels())
with patch('urllib.request.urlopen', side_effect=AssertionError('Catalog used the network')):
    from voice_extraction.model_registry import model_catalog
    print(json.dumps(model_catalog(), allow_nan=False))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        catalog = json.loads(result.stdout)
        self.assertEqual(catalog["id"], "mimo-scnet")
        self.assertIn("vocals", catalog["stems"])
        self.assertFalse(catalog["weights_bundled"])
        self.assertEqual(len(catalog["archive_sha256"]), 64)
        self.assertEqual(len(catalog["weight_sha256"]), 64)

    def test_local_verified_archive_installs_only_the_known_member(self):
        entries = {
            self.member_name: self.payload,
            "../../escaped.pth": b"must not escape the model directory",
            "../extra.pth": b"must not be extracted",
            "release/unused.txt": b"must not be extracted either",
        }
        data = self.make_zip(entries)
        archive = self.local_archive(data)
        with self.pin_archive(data), patch.object(
            registry, "urlopen", side_effect=AssertionError("Local archive used network")
        ) as transport:
            result = registry.fetch_model(self.models, archive=archive)
        transport.assert_not_called()
        destination = self.models / self.weight_name
        self.assertEqual(destination.read_bytes(), self.payload)
        self.assertEqual(list(self.models.iterdir()), [destination])
        self.assertFalse((self.directory / "escaped.pth").exists())
        self.assertFalse((self.models.parent / "extra.pth").exists())
        self.assertEqual(archive.read_bytes(), data)
        self.assertFalse(result["downloaded"])
        self.assertEqual(Path(result["weights"]).resolve(), destination.resolve())

    def test_downloaded_verified_archive_publishes_valid_weight_and_cleans_stage(self):
        with patch.object(registry, "urlopen", return_value=io.BytesIO(self.archive_bytes)) as transport:
            result = registry.fetch_model(self.models)
        transport.assert_called_once()
        destination = self.models / self.weight_name
        self.assertEqual(destination.read_bytes(), self.payload)
        self.assertEqual(list(self.models.iterdir()), [destination])
        self.assertTrue(result["downloaded"])
        self.assertEqual(result["sha256"], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(registry.verify_weights(destination), destination.resolve())

    def test_existing_valid_weight_is_reused_without_downloading(self):
        self.models.mkdir(parents=True)
        destination = self.models / self.weight_name
        destination.write_bytes(self.payload)
        with patch.object(registry, "urlopen", side_effect=AssertionError("Valid cache used network")) as transport:
            result = registry.fetch_model(self.models)
        transport.assert_not_called()
        self.assertFalse(result["downloaded"])
        self.assertEqual(destination.read_bytes(), self.payload)
        self.assertEqual(list(self.models.iterdir()), [destination])

    def test_existing_invalid_weight_is_never_overwritten_or_downloaded(self):
        self.models.mkdir(parents=True)
        destination = self.models / self.weight_name
        invalid = b"x" * len(self.payload)
        destination.write_bytes(invalid)
        with patch.object(registry, "urlopen", side_effect=AssertionError("Invalid cache used network")) as transport:
            with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
                registry.fetch_model(self.models)
        transport.assert_not_called()
        self.assertEqual(destination.read_bytes(), invalid)
        self.assertEqual(list(self.models.iterdir()), [destination])

    def test_weight_verification_checks_size_and_full_hash(self):
        path = self.directory / self.weight_name
        for content in [self.payload[:-1], b"x" * len(self.payload)]:
            with self.subTest(bytes=len(content)):
                path.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
                    registry.verify_weights(path)

    def test_local_archive_checks_size_and_hash_before_installing(self):
        corrupted = bytearray(self.archive_bytes)
        corrupted[10] ^= 1
        for data in [self.archive_bytes[:-1], bytes(corrupted)]:
            with self.subTest(bytes=len(data)):
                archive = self.local_archive(data)
                with patch.object(registry, "urlopen", side_effect=AssertionError("Local archive used network")):
                    with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
                        registry.fetch_model(self.models, archive=archive)
                self.assert_no_published_or_staged_files()
                self.assertEqual(archive.read_bytes(), data)

    def test_verified_archive_with_wrong_weight_hash_does_not_publish(self):
        wrong_weight = b"x" * len(self.payload)
        data = self.make_zip({self.member_name: wrong_weight})
        archive = self.local_archive(data)
        with self.pin_archive(data):
            with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
                registry.fetch_model(self.models, archive=archive)
        self.assert_no_published_or_staged_files()

    def test_verified_archive_with_wrong_member_size_does_not_publish(self):
        data = self.make_zip({self.member_name: self.payload[:-1]})
        archive = self.local_archive(data)
        with self.pin_archive(data):
            with self.assertRaisesRegex(ValueError, "member size"):
                registry.fetch_model(self.models, archive=archive)
        self.assert_no_published_or_staged_files()

    def test_authenticated_nonzip_or_missing_member_is_rejected_and_cleaned(self):
        for data in [b"authenticated bytes that are not a ZIP", self.make_zip({"unrelated.pth": self.payload})]:
            with self.subTest(bytes=len(data)):
                archive = self.local_archive(data)
                with self.pin_archive(data):
                    with self.assertRaisesRegex(ValueError, "archive.*unexpected layout"):
                        registry.fetch_model(self.models, archive=archive)
                self.assert_no_published_or_staged_files()

    def test_oversized_incomplete_and_corrupt_downloads_leave_no_weight_or_stage(self):
        corrupted = bytearray(self.archive_bytes)
        corrupted[10] ^= 1
        cases = {
            "oversized": self.archive_bytes + b"unexpected trailing bytes",
            "incomplete": self.archive_bytes[:-4],
            "corrupt": bytes(corrupted),
        }
        for label, data in cases.items():
            with self.subTest(download=label):
                with patch.object(registry, "urlopen", return_value=io.BytesIO(data)):
                    with self.assertRaises(ValueError):
                        registry.fetch_model(self.models)
                self.assert_no_published_or_staged_files()

    def test_interrupted_download_also_cleans_the_stage(self):
        class InterruptedResponse(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise URLError("connection interrupted")
                return super().read(8)

        with patch.object(registry, "urlopen", return_value=InterruptedResponse(self.archive_bytes)):
            with self.assertRaises(URLError):
                registry.fetch_model(self.models)
        self.assert_no_published_or_staged_files()

    def test_truncated_chunked_download_suggests_local_archive_and_cleans_stage(self):
        response = self.truncated_chunked_response()
        with patch.object(registry, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "--archive") as caught:
                registry.fetch_model(self.models)
        self.assertIsInstance(caught.exception.__cause__, IncompleteRead)
        self.assertTrue(response.isclosed())
        self.assert_no_published_or_staged_files()

    def test_truncated_chunked_download_cli_has_readable_error_and_no_partial_json(self):
        for output_options in ([], ["--json"]):
            with self.subTest(output_options=output_options):
                response = self.truncated_chunked_response()
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(registry, "urlopen", return_value=response), \
                        redirect_stdout(stdout), redirect_stderr(stderr):
                    result = cli.main([
                        "fetch-model", "mimo-scnet", "--models-dir", str(self.models),
                        *output_options,
                    ])
                self.assertEqual(result, 2)
                self.assertEqual(stdout.getvalue(), "")
                diagnostic = stderr.getvalue()
                self.assertTrue(diagnostic.startswith("error:"), diagnostic)
                self.assertIn("interrupted", diagnostic.lower())
                self.assertIn("--archive", diagnostic)
                self.assertNotIn("traceback", diagnostic.lower())
                self.assertTrue(response.isclosed())
                self.assert_no_published_or_staged_files()


if __name__ == "__main__":
    unittest.main()

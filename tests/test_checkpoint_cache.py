import _requires_python  # noqa: F401  (must run first)
import ast
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import unittest
from unittest.mock import Mock
from urllib.parse import urlparse


SOURCE = Path(__file__).resolve().parents[1] / "src" / "handler.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
FUNCTIONS = [
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name in {
        "sanitize_filename", "checkpoint_leaf_from_url", "leaf_has_checkpoint_extension",
        "sniff_checkpoint_extension", "checkpoint_matches_path",
        "_volume_checkpoint_try_legacy_migrate", "resolve_volume_checkpoint_path",
        "finalize_volume_checkpoint_dest", "cache_downloaded_checkpoint",
    }
]


def load_cache_functions(**overrides):
    namespace = {
        "json": json, "os": os, "re": re, "shutil": shutil, "struct": struct,
        "tempfile": tempfile, "urlparse": urlparse,
        "_MIN_VOLUME_CHECKPOINT_BYTES": 16,
        "_MAX_SAFE_TENSORS_HEADER": 100 * 1024 * 1024,
        "_CHECKPOINT_SUFFIXES": (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin"),
        "log_worker_failure": Mock(), "print": Mock(),
    }
    exec(compile(ast.Module(body=FUNCTIONS, type_ignores=[]), str(SOURCE), "exec"), namespace)
    namespace.update(overrides)
    return namespace


def safetensors_bytes():
    header = json.dumps({"tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00" * 4


class CheckpointCacheTests(unittest.TestCase):
    def migrate(self, directory, copy=shutil.copy2):
        namespace = load_cache_functions(
            shutil=Mock(copy2=copy),
            sniff_checkpoint_extension=lambda path: ".safetensors",
        )
        return namespace["_volume_checkpoint_try_legacy_migrate"](
            directory, "123", os.path.join(directory, "123")
        )

    def test_complete_copy_gets_loader_suffix_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory, "123")
            original.write_bytes(b"synthetic checkpoint")
            result = self.migrate(directory)
            self.assertEqual(result, str(original) + ".safetensors")
            self.assertEqual(Path(result).read_bytes(), original.read_bytes())
            self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_failed_copy_never_returns_bare_path_or_publishes_partial_cache(self):
        def fail_copy(source, destination):
            Path(destination).write_bytes(b"partial")
            raise OSError("synthetic volume failure")

        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory, "123")
            original.write_bytes(b"synthetic checkpoint")
            self.assertIsNone(self.migrate(directory, fail_copy))
            self.assertEqual(list(Path(directory).iterdir()), [original])
            self.assertEqual(original.read_bytes(), b"synthetic checkpoint")

    def test_existing_suffixed_cache_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "123").write_bytes(safetensors_bytes())
            fixed = Path(directory, "123.safetensors")
            fixed.write_bytes(safetensors_bytes())
            copy = Mock(side_effect=AssertionError("must not copy"))
            namespace = load_cache_functions(shutil=Mock(copy2=copy))
            self.assertEqual(
                namespace["_volume_checkpoint_try_legacy_migrate"](directory, "123", str(Path(directory, "123"))),
                str(fixed),
            )
            self.assertEqual(fixed.read_bytes(), safetensors_bytes())

    def test_valid_legacy_cache_replaces_invalid_suffixed_cache(self):
        namespace = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory, "123")
            legacy.write_bytes(safetensors_bytes())
            fixed = Path(directory, "123.safetensors")
            fixed.write_bytes(b"\x08" + b"invalid cache" * 4)
            self.assertEqual(
                namespace["resolve_volume_checkpoint_path"](directory, "https://example.test/models/123"),
                str(fixed),
            )
            self.assertEqual(fixed.read_bytes(), legacy.read_bytes())

    def test_short_legacy_cache_becomes_a_cache_miss(self):
        namespace = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "123").write_bytes(b"bad")
            self.assertIsNone(
                namespace["resolve_volume_checkpoint_path"](directory, "https://example.test/models/123")
            )

    def test_cache_reuse_checks_content_and_loader_suffix(self):
        namespace = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory, "123.safetensors")
            good.write_bytes(safetensors_bytes())
            url = "https://example.test/models/123"
            self.assertEqual(namespace["resolve_volume_checkpoint_path"](directory, url), str(good))

            good.write_bytes(b"\x08" + b"not a checkpoint" * 4)
            self.assertIsNone(namespace["resolve_volume_checkpoint_path"](directory, url))
            self.assertTrue(good.exists())

            good.write_bytes(b"\x80\x02" + b"pickle bytes" * 4)
            self.assertIsNone(namespace["resolve_volume_checkpoint_path"](directory, url))
            ckpt = Path(directory, "123.ckpt")
            ckpt.write_bytes(good.read_bytes())
            self.assertEqual(namespace["resolve_volume_checkpoint_path"](directory, url), str(ckpt))

    def test_download_validation_rejects_bad_bytes_and_wrong_suffix(self):
        namespace = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            download = Path(directory, "download")
            download.write_bytes(b"\x08" + b"not a checkpoint" * 4)
            with self.assertRaises(ValueError):
                namespace["finalize_volume_checkpoint_dest"](directory, "https://example.test/models/123", str(download))

            download.write_bytes(safetensors_bytes())
            with self.assertRaises(ValueError):
                namespace["finalize_volume_checkpoint_dest"](directory, "https://example.test/model.ckpt", str(download))
            self.assertEqual(
                namespace["finalize_volume_checkpoint_dest"](directory, "https://example.test/models/123", str(download)),
                str(Path(directory, "123.safetensors")),
            )
            self.assertEqual(
                namespace["finalize_volume_checkpoint_dest"](directory, "https://example.test/model.safetensors", str(download)),
                str(Path(directory, "model.safetensors")),
            )

            for checkpoint in (b"\x80\x02" + b"pickle bytes" * 4, b"PK\x03\x04" + b"zip bytes" * 4):
                download.write_bytes(checkpoint)
                self.assertEqual(
                    namespace["finalize_volume_checkpoint_dest"](directory, "https://example.test/model.ckpt", str(download)),
                    str(Path(directory, "model.ckpt")),
                )

    def test_cache_publish_is_atomic_and_preserves_existing_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "download")
            source.write_bytes(safetensors_bytes())
            destination = Path(directory, "123.safetensors")
            destination.write_bytes(b"existing")

            def fail_copy(_, staged):
                Path(staged).write_bytes(b"partial")
                raise OSError("synthetic volume failure")

            namespace = load_cache_functions(shutil=Mock(copy2=fail_copy))
            with self.assertRaises(OSError):
                namespace["cache_downloaded_checkpoint"](str(source), str(destination), directory)
            self.assertEqual(destination.read_bytes(), b"existing")
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["123.safetensors", "download"])

            def corrupt_copy(_, staged):
                Path(staged).write_bytes(b"x" * source.stat().st_size)

            namespace = load_cache_functions(shutil=Mock(copy2=corrupt_copy))
            with self.assertRaises(ValueError):
                namespace["cache_downloaded_checkpoint"](str(source), str(destination), directory)
            self.assertEqual(destination.read_bytes(), b"existing")

            load_cache_functions()["cache_downloaded_checkpoint"](str(source), str(destination), directory)
            self.assertEqual(destination.read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main()

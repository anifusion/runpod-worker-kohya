import _requires_python  # noqa: F401  (must run first)
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlparse


SOURCE = Path(__file__).resolve().parents[1] / "src" / "handler.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
FUNCTIONS = [
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name in {
        "sanitize_filename", "checkpoint_leaf_from_url", "leaf_has_checkpoint_extension",
        "sniff_checkpoint_extension", "checkpoint_matches_path",
        "checkpoint_cache_path", "normalize_downloaded_checkpoint", "resolve_volume_checkpoint_path",
        "finalize_volume_checkpoint_dest", "cache_downloaded_checkpoint", "download_checkpoint",
    }
]


def load_cache_functions(**overrides):
    namespace = {
        "hashlib": hashlib, "json": json, "os": os, "re": re, "shutil": shutil, "struct": struct,
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
    def test_legacy_caches_are_preserved_but_not_reused(self):
        ns = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            for name in ("123", "123.safetensors", "123.ckpt"):
                Path(directory, name).write_bytes(safetensors_bytes())
            self.assertIsNone(ns["resolve_volume_checkpoint_path"](directory, "https://example.test/models/123"))
            self.assertEqual(len(list(Path(directory).iterdir())), 3)

    def test_source_identity_includes_host_path_and_query(self):
        ns = load_cache_functions()
        urls = ["https://one.example.test/model", "https://two.example.test/model",
                "https://one.example.test/other/model", "https://one.example.test/model?version=2"]
        paths = [ns["checkpoint_cache_path"]("/volume", url) for url in urls]
        self.assertEqual(len(set(paths)), len(urls))
        self.assertEqual(paths[0], ns["checkpoint_cache_path"]("/volume", urls[0]))
        for path in paths:
            self.assertNotIn("example.test", path)
            self.assertTrue(path.endswith(".safetensors"))

    def test_cache_reuse_and_symlink_rejection(self):
        ns = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            url = "https://example.test/model"
            cache = Path(ns["checkpoint_cache_path"](directory, url))
            cache.write_bytes(safetensors_bytes())
            self.assertEqual(ns["resolve_volume_checkpoint_path"](directory, url), str(cache))
            for data in (b"short", b"\x80\x02" + b"pickle" * 20, b"PK\x03\x04" + b"zip" * 20):
                cache.write_bytes(data)
                self.assertIsNone(ns["resolve_volume_checkpoint_path"](directory, url))
            cache.unlink()
            other = Path(directory, "other.safetensors")
            other.write_bytes(safetensors_bytes())
            cache.symlink_to(other)
            self.assertIsNone(ns["resolve_volume_checkpoint_path"](directory, url))

    def test_download_rejects_pickle_zip_and_misleading_suffixes(self):
        ns = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "download.safetensors")
            for data in (b"short", b"\x80\x02" + b"pickle" * 20, b"PK\x03\x04" + b"zip" * 20):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    ns["finalize_volume_checkpoint_dest"](directory, "https://example.test/model.safetensors", str(path))
                with self.assertRaises(ValueError):
                    ns["normalize_downloaded_checkpoint"](str(path))
            path.write_bytes(safetensors_bytes())
            for suffix in (".ckpt", ".pt", ".pth", ".bin"):
                with self.assertRaises(ValueError):
                    ns["finalize_volume_checkpoint_dest"](directory, "https://example.test/model" + suffix, str(path))

    def test_downloads_always_select_safetensors_loader(self):
        ns = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            for name in ("123", "download.sft", "download.bin", "model.SAFETENSORS", "ready.safetensors"):
                path = Path(directory, name)
                path.write_bytes(safetensors_bytes())
                result = ns["normalize_downloaded_checkpoint"](str(path))
                self.assertTrue(result.endswith(".safetensors"))
                self.assertTrue(ns["checkpoint_matches_path"](result))
                self.assertEqual(Path(result).read_bytes(), safetensors_bytes())
                if name == "ready.safetensors":
                    self.assertEqual(result, str(path))
                else:
                    self.assertFalse(path.exists())
                with self.assertRaises(OSError):
                    ns["cache_downloaded_checkpoint"](result, str(Path(directory, "missing", "cache.safetensors")), str(Path(directory, "missing")))
                self.assertTrue(ns["checkpoint_matches_path"](result))

    def test_header_length_with_pickle_prefix_is_accepted(self):
        ns = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            raw = safetensors_bytes()
            size = struct.unpack("<Q", raw[:8])[0]
            header = raw[8:8 + size].ljust(128, b" ")
            path = Path(directory, "model.safetensors")
            path.write_bytes(struct.pack("<Q", 128) + header + raw[8 + size:])
            self.assertTrue(ns["checkpoint_matches_path"](str(path)))

    def test_handler_preparation_keeps_safe_suffix_without_working_volume(self):
        handler = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "handler")
        start = next(i for i, node in enumerate(handler.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "VOLUME_DIR" for target in node.targets))
        end = next(i for i, node in enumerate(handler.body) if i > start and isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call) and node.value.args
                   and isinstance(node.value.args[0], ast.Constant)
                   and node.value.args[0].value == "runpod-worker-kohya: downloading training archive")
        for volume in (False, True):
            with self.subTest(volume=volume), tempfile.TemporaryDirectory() as directory:
                download = Path(directory, "download")
                download.write_bytes(safetensors_bytes())
                ns = load_cache_functions()
                ns.update({"job_input": {"model_url": "https://example.test/models/123"},
                           "model_url": "https://example.test/models/123", "job": {"id": "synthetic-job"},
                           "download_checkpoint": Mock(return_value={"file_path": str(download)}),
                           "resolve_volume_checkpoint_path": Mock(return_value=None),
                           "cache_downloaded_checkpoint": Mock(side_effect=OSError("synthetic full volume"))})
                # Exercise the handler's actual download/cache branch, changing only
                # the fixture volume directory and returning before dataset work.
                body = [ast.Assign(targets=[ast.Name(id="VOLUME_DIR", ctx=ast.Store())],
                                   value=ast.Constant(directory if volume else str(Path(directory, "absent"))))]
                body += handler.body[start + 1:end]
                body += [ast.Return(value=ast.Name(id="downloaded_model", ctx=ast.Load()))]
                function = ast.FunctionDef(name="prepare", args=ast.arguments(posonlyargs=[], args=[],
                    kwonlyargs=[], kw_defaults=[], defaults=[]), body=body, decorator_list=[])
                module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
                exec(compile(module, str(SOURCE), "exec"), ns)
                result = ns["prepare"]()
                self.assertTrue(ns["checkpoint_matches_path"](result["file_path"]))
                self.assertTrue(result["file_path"].endswith(".safetensors"))
                self.assertEqual(ns["cache_downloaded_checkpoint"].call_count, int(volume))
                # A misleading extension cannot make pickle bytes reach the loader.
                download.write_bytes(b"PK\x03\x04" + b"synthetic" * 20)
                ns["download_checkpoint"].return_value = {"file_path": str(download)}
                result = ns["prepare"]()
                self.assertEqual(result["failure_stage"], "model_checkpoint")

    def test_deep_json_and_uppercase_cache_fail_closed(self):
        ns = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "model.SAFETENSORS")
            path.write_bytes(safetensors_bytes())
            self.assertFalse(ns["checkpoint_matches_path"](str(path)))
            raw = b'{"tensor":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}"
            path.write_bytes(struct.pack("<Q", len(raw)) + raw)
            with self.assertRaises(ValueError):
                ns["sniff_checkpoint_extension"](str(path))

    def test_checkpoint_download_never_extracts_and_cleans_failed_stream(self):
        ns = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            session = Mock()
            response = Mock()
            session.get.return_value.__enter__ = Mock(return_value=response)
            session.get.return_value.__exit__ = Mock(return_value=False)
            manager = Mock()
            manager.__enter__ = Mock(return_value=session)
            manager.__exit__ = Mock(return_value=False)
            ns["requests"] = Mock(Session=Mock(return_value=manager))
            real_mkstemp = tempfile.mkstemp
            def local_temp(**kwargs):
                return real_mkstemp(dir=directory, suffix=kwargs["suffix"])
            with patch.object(os, "makedirs"), patch.object(tempfile, "mkstemp", side_effect=local_temp):
                response.iter_content.return_value = [b"PK\x03\x04", b"synthetic zip bytes"]
                result = ns["download_checkpoint"]("https://example.test/model.zip")
                self.assertEqual(Path(result["file_path"]).read_bytes(), b"PK\x03\x04synthetic zip bytes")
                self.assertEqual(len(list(Path(directory).iterdir())), 1)
                self.assertFalse(session.trust_env)
                session.get.assert_called_with("https://example.test/model.zip", stream=True, timeout=(10, 60))
                Path(result["file_path"]).unlink()
                response.iter_content.side_effect = OSError("synthetic interrupted stream")
                with self.assertRaises(OSError):
                    ns["download_checkpoint"]("https://example.test/model")
                self.assertEqual(list(Path(directory).iterdir()), [])
                response.raise_for_status.side_effect = ValueError("synthetic HTTP failure")
                response.iter_content.reset_mock()
                with self.assertRaises(ValueError):
                    ns["download_checkpoint"]("https://example.test/missing")
                response.iter_content.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failed_normalization_preserves_download_and_cleans_placeholder(self):
        ns = load_cache_functions()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "download")
            source.write_bytes(safetensors_bytes())
            with patch.object(os, "replace", side_effect=OSError("synthetic rename failure")):
                with self.assertRaises(OSError):
                    ns["normalize_downloaded_checkpoint"](str(source))
            self.assertEqual(source.read_bytes(), safetensors_bytes())
            self.assertEqual(list(Path(directory).iterdir()), [source])

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

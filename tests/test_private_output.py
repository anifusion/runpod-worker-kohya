import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock


class PrivateOutputTests(unittest.TestCase):
    def test_destination_preserves_legacy_and_fails_closed(self):
        source = Path(__file__).resolve().parents[1] / "src" / "handler.py"
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "lora_output_destination"]
        scope = {"os": os}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
        destination = scope["lora_output_destination"]
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(destination("fixture", False), ("lora", "fixture.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "private_model_storage_unavailable"):
                destination("fixture", True)
        with patch.dict(os.environ, {"R2_PRIVATE_BUCKET": "private-fixture"}, clear=True):
            self.assertEqual(destination("fixture", True),
                             ("private-fixture", "private-v1/lora/fixture.safetensors"))

    def test_private_upload_uses_selected_bucket_and_never_local_fallback(self):
        source = Path(__file__).resolve().parents[1] / "src" / "handler.py"
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "upload_lora_to_bucket"]
        client = MagicMock()
        session = MagicMock()
        session.Session.return_value.client.return_value = client
        scope = {"os": MagicMock(), "TransferConfig": MagicMock(), "BotoConfig": MagicMock(),
                 "boto_session": session, "extract_region_from_url": lambda _: "auto",
                 "tqdm": MagicMock(), "LORA_UPLOAD_MAX_CONCURRENCY": 2,
                 "LORA_UPLOAD_MAX_POOL_CONNECTIONS": 4}
        scope["os"].environ.get.return_value = None
        scope["os"].path.getsize.return_value = 4
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
        upload = scope["upload_lora_to_bucket"]
        with self.assertRaisesRegex(RuntimeError, "private_model_credentials_unavailable"):
            upload("private-v1/lora/fixture.safetensors", "fixture", require_remote=True)
        client.upload_file.assert_not_called()
        upload("private-v1/lora/fixture.safetensors", "fixture", bucket_name="private-fixture",
               bucket_creds={"endpointUrl": "https://storage.example.test", "accessId": "fixture", "accessSecret": "fixture"},
               require_remote=True)
        args = client.upload_file.call_args.kwargs
        self.assertEqual(args["Bucket"], "private-fixture")
        self.assertEqual(args["Key"], "private-v1/lora/fixture.safetensors")
        self.assertEqual(args["ExtraArgs"], {"CacheControl": "private, no-store"})

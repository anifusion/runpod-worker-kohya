import _requires_python  # noqa: F401  (must run first)
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
from urllib.parse import urlparse

SOURCE = Path(__file__).resolve().parents[1] / "src" / "handler.py"
TREE = ast.parse(SOURCE.read_text())


class FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def load(*names, **scope):
    nodes = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope


class StorageRoutingTests(unittest.TestCase):
    def test_endpoint_bucket_path_is_split_from_the_account_url(self):
        split = load("split_storage_endpoint", urlparse=urlparse)["split_storage_endpoint"]
        self.assertEqual(split("https://account.storage.example.test/example-public"),
                         ("https://account.storage.example.test", "example-public"))
        self.assertEqual(split("https://account.storage.example.test/example-public/"),
                         ("https://account.storage.example.test", "example-public"))
        self.assertEqual(split("https://nyc3.storage.example.test"),
                         ("https://nyc3.storage.example.test", None))
        for invalid in ("https://host.example.test/a/b", "ftp://host.example.test", "",
                        "https://host.example.test/a?x=1", "not a url"):
            with self.subTest(endpoint=invalid), self.assertRaises(ValueError):
                split(invalid)

    def test_destination_matches_the_app_read_path(self):
        destination = load("lora_output_destination", os=os)["lora_output_destination"]
        with patch.dict(os.environ, {}, clear=True):
            # Legacy layout without an endpoint bucket.
            self.assertEqual(destination("fixture", False), ("lora", "fixture.safetensors"))
            # The app reads public weights at lora/<id>.safetensors in its public bucket.
            self.assertEqual(destination("fixture", False, "example-public"),
                             ("example-public", "lora/fixture.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "private_model_storage_unavailable"):
                destination("fixture", True, "example-public")
        with patch.dict(os.environ, {"R2_PRIVATE_BUCKET": "example-private"}, clear=True):
            # Private weights never follow the endpoint's public bucket.
            self.assertEqual(destination("fixture", True, "example-public"),
                             ("example-private", "private-v1/lora/fixture.safetensors"))


class UploadTests(unittest.TestCase):
    CREDS = {"endpointUrl": "https://account.storage.example.test/example-public",
             "accessId": "fixture", "accessSecret": "fixture"}

    def make_upload(self, environ=None, head=None):
        client = MagicMock()
        if isinstance(head, Exception):
            client.head_object.side_effect = head
        else:
            client.head_object.return_value = head if head is not None else {"ContentLength": 4}
        session = MagicMock()
        session.Session.return_value.client.return_value = client
        fake_os = MagicMock()
        fake_os.environ.get.side_effect = lambda key, default=None: (environ or {}).get(key, default)
        fake_os.path.getsize.return_value = 4
        scope = load("upload_lora_to_bucket", "storage_credentials", "split_storage_endpoint",
                     os=fake_os, urlparse=urlparse, TransferConfig=MagicMock(), BotoConfig=MagicMock(),
                     boto_session=session, extract_region_from_url=lambda _: "auto", tqdm=MagicMock(),
                     shutil=MagicMock(), ClientError=FakeClientError,
                     LORA_UPLOAD_MAX_CONCURRENCY=2, LORA_UPLOAD_MAX_POOL_CONNECTIONS=4)
        return scope, client, session

    def test_private_upload_uses_account_url_and_private_bucket(self):
        scope, client, session = self.make_upload()
        scope["upload_lora_to_bucket"]("private-v1/lora/fixture.safetensors", "fixture",
                                       bucket_name="example-private", bucket_creds=self.CREDS,
                                       require_remote=True)
        client_kwargs = session.Session.return_value.client.call_args.kwargs
        # The endpoint's bucket path is stripped, so the requested bucket is not a folder in it.
        self.assertEqual(client_kwargs["endpoint_url"], "https://account.storage.example.test")
        botocfg = scope["BotoConfig"].call_args.kwargs
        self.assertEqual(botocfg["s3"], {"addressing_style": "path"})
        args = client.upload_file.call_args.kwargs
        self.assertEqual((args["Bucket"], args["Key"]),
                         ("example-private", "private-v1/lora/fixture.safetensors"))
        self.assertEqual(args["ExtraArgs"], {"CacheControl": "private, no-store"})
        client.head_object.assert_called_once_with(Bucket="example-private",
                                                   Key="private-v1/lora/fixture.safetensors")

    def test_missing_or_wrong_size_upload_fails_the_job(self):
        for head in ({"ContentLength": 3}, FakeClientError("404"), FakeClientError("NoSuchKey")):
            with self.subTest(head=head):
                scope, _, _ = self.make_upload(head=head)
                with self.assertRaisesRegex(RuntimeError, "uploaded_model_verification_failed"):
                    scope["upload_lora_to_bucket"]("lora/fixture.safetensors", "fixture",
                                                   bucket_name="example-public", bucket_creds=self.CREDS)

    def test_read_denied_verification_keeps_the_upload(self):
        scope, client, _ = self.make_upload(head=FakeClientError("AccessDenied"))
        scope["upload_lora_to_bucket"]("lora/fixture.safetensors", "fixture",
                                       bucket_name="example-public", bucket_creds=self.CREDS)
        client.generate_presigned_url.assert_called_once()

    def test_missing_credentials_never_report_a_local_copy_as_uploaded(self):
        scope, client, _ = self.make_upload()
        upload = scope["upload_lora_to_bucket"]
        with self.assertRaisesRegex(RuntimeError, "private_model_credentials_unavailable"):
            upload("private-v1/lora/fixture.safetensors", "fixture", require_remote=True)
        with self.assertRaisesRegex(RuntimeError, "model_storage_credentials_unavailable"):
            upload("fixture.safetensors", "fixture")
        client.upload_file.assert_not_called()
        scope, _, _ = self.make_upload(environ={"ALLOW_LOCAL_LORA_UPLOAD": "1"})
        self.assertEqual(scope["upload_lora_to_bucket"]("fixture.safetensors", "fixture"),
                         "local_upload/fixture.safetensors")


if __name__ == "__main__":
    unittest.main()

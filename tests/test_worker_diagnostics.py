import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
import types
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "src" / "handler.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
FUNCTIONS = [
    node for node in TREE.body
    if (isinstance(node, ast.FunctionDef) and node.name in {"log_worker_failure", "_run_training_subprocess", "safe_training_failure_label", "handler"})
    or (isinstance(node, ast.ClassDef) and node.name == "TrainingSubprocessTimeout")
]


def load_functions(**overrides):
    output = []
    namespace = {
        "json": json,
        "re": re,
        "subprocess": subprocess,
        "sys": sys,
        "time": __import__("time"),
        "traceback": traceback,
        "os": __import__("os"),
        "select": __import__("select"),
        "TRAINING_LOSS_NAN_PATTERN": re.compile(r"avr_loss=nan\b", re.IGNORECASE),
        "_TRAINING_DIAGNOSTIC_TAIL_CHARS": 131072,
        "print": lambda *args: output.append(" ".join(map(str, args))),
    }
    namespace.update(overrides)
    exec(compile(ast.Module(body=FUNCTIONS, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace, output


class WorkerDiagnosticsTests(unittest.TestCase):
    def test_credential_failure_keeps_stage_without_exception_text(self):
        namespace, output = load_functions()
        error = RuntimeError("signed-url-and-private-content")
        diagnostic = namespace["log_worker_failure"]("model_download", error, credential_boundary=True, job_id="job_123")
        self.assertIn('"stage": "model_download"', output[0])
        self.assertIn('"job_id": "job_123"', output[0])
        self.assertNotIn("signed-url-and-private-content", output[0])
        self.assertNotIn("signed-url-and-private-content", json.dumps(diagnostic))
        self.assertEqual(diagnostic["type"], "RuntimeError")
        self.assertEqual(
            json.loads(output[1].split("safe_event ", 1)[1]),
            {"event": "worker_failure", "stage": "model_download", "job_id": "job_123"},
        )

    def test_subprocess_launch_error_keeps_safe_errno(self):
        namespace, _ = load_functions()
        diagnostic = namespace["log_worker_failure"](
            "training_subprocess", FileNotFoundError(2, "private path"),
            credential_boundary=True, job_id="job_123"
        )
        self.assertEqual(diagnostic["errno"], 2)
        self.assertNotIn("private path", json.dumps(diagnostic))

    def test_provider_code_is_preserved_without_provider_payload(self):
        namespace, _ = load_functions()
        error = RuntimeError("private provider text")
        error.response = {"Error": {"Code": "ExpiredToken", "Message": "private payload"},
                          "ResponseMetadata": {"HTTPStatusCode": 403}}
        diagnostic = namespace["log_worker_failure"]("model_download", error, credential_boundary=True)
        self.assertEqual(diagnostic["code"], "ExpiredToken")
        self.assertEqual(diagnostic["status"], 403)
        self.assertNotIn("private payload", json.dumps(diagnostic))

    def test_validation_failure_keeps_safe_code_and_parameter(self):
        namespace, _ = load_functions()
        diagnostic = namespace["log_worker_failure"](
            "input_validation", job_id="job_123", code="InvalidNumericParameter", parameter="train_batch_size"
        )
        self.assertEqual(diagnostic["code"], "InvalidNumericParameter")
        self.assertEqual(diagnostic["parameter"], "train_batch_size")

    def test_handler_returns_validation_diagnostics(self):
        namespace, _ = load_functions(
            validate=lambda value, schema: {"validated_input": value},
            INPUT_SCHEMA={},
            validate_url=lambda value: value == "good",
            validate_numeric_param=lambda *args: False,
        )
        invalid_url = namespace["handler"]({"id": "job_123", "input": {"model_url": "bad", "zip_url": "good"}})
        self.assertEqual(invalid_url["diagnostic"]["code"], "InvalidModelUrl")
        invalid_number = namespace["handler"]({
            "id": "job_123", "input": {"model_url": "good", "zip_url": "good", "train_batch_size": 0}
        })
        self.assertEqual(invalid_number["diagnostic"]["code"], "InvalidNumericParameter")
        self.assertEqual(invalid_number["diagnostic"]["parameter"], "train_batch_size")

    def test_safe_event_omits_private_exception_and_bad_identifier(self):
        namespace, output = load_functions()
        namespace["log_worker_failure"](
            "training_subprocess", RuntimeError("secret\nprivate prompt"), job_id="private@example.test"
        )
        self.assertEqual(
            json.loads(output[1].split("safe_event ", 1)[1]),
            {"event": "worker_failure", "stage": "training_subprocess"},
        )
        self.assertEqual(output[1].count("\n"), 0)

    def test_internal_failure_keeps_bounded_message_and_stack(self):
        namespace, output = load_functions()
        try:
            raise RuntimeError("training failed")
        except RuntimeError as error:
            namespace["log_worker_failure"]("training_subprocess", error)
        self.assertIn('"message": "training failed"', output[0])
        self.assertIn('"frames":', output[0])

    def test_multiline_failure_keeps_one_readable_log_line(self):
        namespace, output = load_functions()
        try:
            try:
                raise RuntimeError("storage \r\n unavailable")
            except RuntimeError as cause:
                raise ValueError("training  failed \n retry") from cause
        except ValueError as error:
            namespace["log_worker_failure"](
                "training_subprocess", error, tail="worker \n failed"
            )
        line = output[0]
        self.assertIn('"message": "training  failed \\n retry"', line)
        self.assertIn('"cause_message": "storage \\r\\n unavailable"', line)
        self.assertIn('"worker_tail": "worker \\n failed"', line)
        self.assertIn('"frames":', line)
        self.assertIn("\\n", line)
        self.assertNotIn("\n", line)

    def test_expanded_worker_tail_stays_bounded(self):
        namespace, output = load_functions()
        diagnostic = namespace["log_worker_failure"]("training_subprocess", tail="a\n" * 70000)
        line = output[0]
        record = json.loads(line[line.index("{"):])
        self.assertEqual(len(record["worker_tail"]), 750)
        self.assertEqual(record["worker_tail_chars"], 131072)
        self.assertEqual(len(diagnostic["worker_tail"]), 131072)
        self.assertIn("\\n", line)
        self.assertNotIn("\n", line)

    def test_training_tail_redacts_credentials_before_restricted_diagnostics(self):
        fake_subprocess = types.SimpleNamespace(
            PIPE=subprocess.PIPE,
            STDOUT=subprocess.STDOUT,
            run=Mock(return_value=types.SimpleNamespace(
                returncode=1,
                stdout="error secret-token https://private.example.test/log avr_loss=nan",
            )),
        )
        namespace, output = load_functions(
            subprocess=fake_subprocess,
            sys=types.SimpleNamespace(platform="win32"),
        )
        code, saw_nan, tail = namespace["_run_training_subprocess"](
            ["train", "--http-log-token", "secret-token", "--http-log-endpoint", "https://private.example.test/log"],
            10,
        )
        self.assertEqual(code, 1)
        self.assertTrue(saw_nan)
        self.assertNotIn("secret-token", tail)
        self.assertNotIn("private.example.test", tail)
        self.assertIn("[credential-omitted]", tail)
        diagnostic = namespace["log_worker_failure"](
            "training_subprocess", tail=tail, exit_code=code, job_id="job_123"
        )
        self.assertEqual(diagnostic["exit_code"], 1)
        self.assertNotIn("secret-token", json.dumps(diagnostic))
        self.assertNotIn("private.example.test", json.dumps(diagnostic))
        self.assertEqual(len(output), 2)

    def test_timeout_keeps_redacted_subprocess_output(self):
        fake_subprocess = types.SimpleNamespace(
            PIPE=subprocess.PIPE,
            STDOUT=subprocess.STDOUT,
            TimeoutExpired=subprocess.TimeoutExpired,
            run=Mock(side_effect=subprocess.TimeoutExpired(
                ["train"], 10, output="failed secret-token\nTraceback: startup error"
            )),
        )
        namespace, _ = load_functions(
            subprocess=fake_subprocess,
            sys=types.SimpleNamespace(platform="win32"),
        )
        with self.assertRaises(namespace["TrainingSubprocessTimeout"]) as caught:
            namespace["_run_training_subprocess"](["train", "--http-log-token", "secret-token"], 10)
        self.assertNotIn("secret-token", caught.exception.safe_tail)
        self.assertIn("Traceback: startup error", caught.exception.safe_tail)

    def test_failure_labels_preserve_webhook_guidance_without_private_text(self):
        namespace, _ = load_functions()
        label = namespace["safe_training_failure_label"]
        examples = [
            ("No training images found in archive", "No training images found"),
            ("CUDA out of memory while reading secret-input", "CUDA out of memory"),
            ("CUDA error: an illegal memory access was encountered secret-input", "Training process failed: 1"),
            ("CUDA out of memory followed by illegal memory access secret-input", "Training process failed: 1"),
            ("training process timed out secret-input", "Training timed out"),
            ("_pickle.UnpicklingError: invalid load key secret-input", "Base model download failed"),
            ("OSError: [Errno 122] Disk quota exceeded secret-input", "Disk quota exceeded"),
            ("Failed to write accelerate config secret-input", "Failed to write accelerate config"),
            ("training data invalid secret-input", "Training data invalid"),
            ("OSError: File name too long secret-input", "File name too long"),
            ("unexpected failure secret-input", "Training process failed: 1"),
        ]
        for tail, expected in examples:
            self.assertEqual(label(tail, 1), expected)
            self.assertNotIn("secret-input", label(tail, 1))


if __name__ == "__main__":
    unittest.main()

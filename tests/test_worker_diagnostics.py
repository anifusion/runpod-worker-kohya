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
    if isinstance(node, ast.FunctionDef)
    and node.name in {"log_worker_failure", "_run_training_subprocess", "safe_training_failure_label"}
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
        "print": lambda *args: output.append(" ".join(map(str, args))),
    }
    namespace.update(overrides)
    exec(compile(ast.Module(body=FUNCTIONS, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace, output


class WorkerDiagnosticsTests(unittest.TestCase):
    def test_credential_failure_keeps_stage_without_exception_text(self):
        namespace, output = load_functions()
        error = RuntimeError("signed-url-and-private-content")
        namespace["log_worker_failure"]("model_download", error, credential_boundary=True, job_id="job_123")
        self.assertIn('"stage": "model_download"', output[0])
        self.assertIn('"job_id": "job_123"', output[0])
        self.assertNotIn("signed-url-and-private-content", output[0])

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
                raise RuntimeError("storage\r\nunavailable")
            except RuntimeError as cause:
                raise ValueError("training failed\nretry") from cause
        except ValueError as error:
            namespace["log_worker_failure"](
                "training_subprocess", error, tail="worker\nfailed"
            )
        line = output[0]
        self.assertIn('"message": "training failed | retry"', line)
        self.assertIn('"cause_message": "storage | unavailable"', line)
        self.assertIn('"worker_tail": "worker | failed"', line)
        self.assertIn('"frames":', line)
        self.assertNotIn("\\n", line)
        self.assertNotIn("\n", line)

    def test_expanded_worker_tail_stays_bounded(self):
        namespace, output = load_functions()
        namespace["log_worker_failure"]("training_subprocess", tail="a\n" * 2000)
        line = output[0]
        record = json.loads(line[line.index("{"):])
        self.assertEqual(len(record["worker_tail"]), 2000)
        self.assertNotIn("\\n", line)

    def test_training_tail_stays_out_of_live_stdout_and_job_output(self):
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
        self.assertEqual(output, [])

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

import _requires_python  # noqa: F401
import ast
from collections import defaultdict
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingContractTests(unittest.TestCase):
    def test_handler_preserves_legacy_text_encoder_targets(self):
        tree = ast.parse((ROOT / "src/handler.py").read_text())
        handler = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "_handle_job")
        assignment = next(node for node in handler.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "cmd_args"
                                  for target in node.targets))
        values = {
            "accelerate_config_path": "/tmp/synthetic-config.yaml",
            "downloaded_model": {"file_path": "/tmp/synthetic.safetensors"},
            "job_input": defaultdict(lambda: "1"), "sanitized_params": {},
            "mixed_precision": "bf16", "save_precision": "fp16", "out_id": "synthetic",
        }
        arguments = eval(compile(ast.Expression(assignment.value), "handler-command", "eval"), values)
        self.assertEqual(arguments.count("--network_args"), 1)
        self.assertEqual(arguments[arguments.index("--network_args") + 1],
                         "text_encoder_mlp_only=true")
        self.assertEqual(arguments[arguments.index("--network_module") + 1], "networks.lora")

    def test_training_source_pins_match(self):
        setup = (ROOT / "builder/setup.sh").read_text()
        requirements = (ROOT / "builder/requirements.txt").read_text()
        script_pin = re.search(r"git checkout ([0-9a-f]{40})", setup).group(1)
        library_pin = re.search(r"sd-scripts.git@([0-9a-f]{40})#egg=library", requirements).group(1)
        self.assertEqual(script_pin, library_pin)


if __name__ == "__main__":
    unittest.main()

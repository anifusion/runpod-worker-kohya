"""Installed-library checks using synthetic data and no provider connections."""

import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from fastapi.testclient import TestClient
from runpod.serverless.modules import rp_fastapi
from transformers import CLIPTokenizer


class WorkerCompatibility(unittest.TestCase):
    def test_cached_tokenizers_preserve_token_ids(self):
        texts = ['a small blue bird', 'café, trees!', '']
        prefixes = [[49406, 320, 2442, 1746, 3329, 49407],
                    [49406, 15304, 267, 4682, 256, 49407], [49406, 49407]]
        expected = [row + [49407] * (16 - len(row)) for row in prefixes]
        for name in ('openai_clip-vit-large-patch14', 'laion_CLIP-ViT-bigG-14-laion2B-39B-b160k'):
            with self.subTest(tokenizer=name):
                tokenizer = CLIPTokenizer.from_pretrained('/tokenizer_cache/' + name, local_files_only=True)
                actual = tokenizer(texts, padding='max_length', max_length=16, truncation=True)
                self.assertEqual(actual['input_ids'], expected)

    def test_sdk_api_synthetic_request_and_validation(self):
        with patch.object(rp_fastapi.heartbeat, 'start_ping'), patch.dict(os.environ, {}, clear=True):
            api = rp_fastapi.WorkerAPI({'handler': lambda job: {'value': job['input']['value'] + 1}})
            with TestClient(api.rp_app) as client:
                response = client.post('/runsync', json={'input': {'value': 2}})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['status'], 'COMPLETED')
                self.assertEqual(response.json()['output'], {'value': 3})
                self.assertEqual(client.post('/runsync', json={}).status_code, 422)

    def test_real_handler_registration_and_input_rejection(self):
        handler_path = Path('/sd-scripts/handler.py')
        sys.path.insert(0, str(handler_path.parent))
        try:
            with patch.dict(os.environ, {}, clear=True), patch('runpod.serverless.start') as start:
                runpy.run_path(str(handler_path), run_name='__main__')
                start.assert_called_once()
                handler = start.call_args.args[0]['handler']
                self.assertEqual(handler({'id': 'synthetic', 'input': {}})['failure_stage'],
                                 'input_validation')
        finally:
            sys.path.pop(0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

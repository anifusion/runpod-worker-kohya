"""Synthetic compatibility checks. No model downloads, provider calls or real datasets."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import unittest

parser = argparse.ArgumentParser()
parser.add_argument('--sd-scripts-root', type=Path, required=True)
parser.add_argument('--require-cuda', action='store_true')
options = parser.parse_args()
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(options.sd_scripts_root.resolve()))

import numpy as np
from PIL import Image
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler
from safetensors.torch import save_file, load_file
from transformers import CLIPTextConfig, CLIPTextModel
import sdxl_train_network
from library import train_util, utils, sdxl_original_unet
from networks.lora import LoRAModule


class RuntimeCompatibility(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_training_cli_contract(self):
        args = sdxl_train_network.setup_parser().parse_args([
            '--network_module', 'networks.lora', '--save_model_as', 'safetensors',
            '--gradient_checkpointing', '--no_half_vae', '--cache_latents',
            '--tokenizer_cache_dir', '/tokenizer_cache', '--http-log',
            '--http-log-endpoint', 'https://example.test/log', '--http-log-token', 'synthetic',
            '--optimizer_type', 'AdamW8bit', '--resolution', '1024,1024'])
        self.assertTrue(args.gradient_checkpointing)
        self.assertEqual(args.optimizer_type, 'AdamW8bit')

    def test_dataset_image_loading_and_resizing(self):
        pixels = np.arange(32 * 24 * 3, dtype=np.uint8).reshape(24, 32, 3)
        with tempfile.TemporaryDirectory() as directory:
            for extension in ['png', 'jpg', 'webp', 'gif', 'bmp', 'tif', 'tiff']:
                with self.subTest(format=extension):
                    path = Path(directory, 'synthetic.' + extension)
                    Image.fromarray(pixels).save(path)
                    image = train_util.load_image(str(path))
                    self.assertEqual(image.shape, (24, 32, 3))
                    self.assertEqual(image.dtype, np.uint8)
                    resized = utils.pil_resize(image, (16, 12))
                    self.assertEqual(resized.shape, (12, 16, 3))
                    transformed, original, crop = train_util.trim_and_resize_if_required(False, image, (16, 16), (24, 24))
                    self.assertEqual(transformed.shape, (16, 16, 3))
                    tensor = train_util.IMAGE_TRANSFORMS(transformed)
                    self.assertEqual(tuple(tensor.shape), (3, 16, 16))
                    self.assertTrue(torch.isfinite(tensor).all())
                    self.assertGreaterEqual(tensor.min().item(), -1)
                    self.assertLessEqual(tensor.max().item(), 1)
            for mode in ['RGBA', 'L', 'I;16', 'P']:
                with self.subTest(mode=mode):
                    path = Path(directory, 'synthetic.png')
                    Image.new(mode, (32, 24)).save(path)
                    self.assertEqual(train_util.load_image(str(path)).shape, (24, 32, 3))
                    self.assertEqual(train_util.load_image(str(path), alpha=True).shape, (24, 32, 4))

    def test_sdxl_checkpointed_backward(self):
        block = sdxl_original_unet.ResnetBlock2D(32, 32).train()
        block.gradient_checkpointing = True
        image = torch.randn(1, 32, 8, 8, requires_grad=True)
        embedding = torch.randn(1, sdxl_original_unet.TIME_EMBED_DIM, requires_grad=True)
        block(image, embedding).square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        attention = sdxl_original_unet.CrossAttention(32, heads=2, dim_head=16)
        values = torch.randn(1, 8, 32, requires_grad=True)
        attention(values).square().mean().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_clip_and_vae_forward_backward(self):
        clip = CLIPTextModel(CLIPTextConfig(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
            max_position_embeddings=8))
        clip(torch.tensor([[1, 2, 3, 4]])).last_hidden_state.square().mean().backward()
        vae = AutoencoderKL(block_out_channels=(32,), down_block_types=('DownEncoderBlock2D',),
            up_block_types=('UpDecoderBlock2D',), latent_channels=4, norm_num_groups=32)
        latent = vae.encode(torch.randn(1, 3, 16, 16)).latent_dist.sample()
        decoded = vae.decode(latent).sample
        self.assertTrue(torch.isfinite(decoded).all())
        decoded.square().mean().backward()
        scheduler = DDPMScheduler()
        noisy = scheduler.add_noise(latent.detach(), torch.randn_like(latent), torch.tensor([1]))
        self.assertTrue(torch.isfinite(noisy).all())

    def test_lora_optimizer_and_safe_output(self):
        accelerator = Accelerator(cpu=True)
        original = torch.nn.Linear(8, 8).requires_grad_(False)
        lora = LoRAModule('synthetic', original, lora_dim=4)
        lora.apply_to()
        args = sdxl_train_network.setup_parser().parse_args(['--optimizer_type', 'AdamW'])
        _, _, optimizer = train_util.get_optimizer(args, lora.parameters())
        lora, optimizer = accelerator.prepare(lora, optimizer)
        before = lora.lora_up.weight.detach().clone()
        loss = lora(torch.randn(2, 8)).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        self.assertFalse(torch.equal(before, lora.lora_up.weight))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, 'synthetic.safetensors'))
            tensors = {k: v.detach().contiguous().cpu() for k, v in lora.state_dict().items()}
            save_file(tensors, path)
            restored = load_file(path)
            for key, value in tensors.items():
                self.assertTrue(torch.equal(value, restored[key]))

    @unittest.skipUnless(options.require_cuda, 'GPU checks require --require-cuda')
    def test_gpu_native_kernels_and_optimizer(self):
        self.assertTrue(torch.cuda.is_available(), 'CUDA is required for release acceptance')
        self.assertEqual(torch.version.cuda, '13.2')
        self.assertGreaterEqual(torch.cuda.get_device_capability(), (7, 5))
        import bitsandbytes as bnb
        import xformers.ops
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    continue
                q = torch.randn(1, 16, 2, 32, device='cuda', dtype=dtype, requires_grad=True)
                xformers.ops.memory_efficient_attention(q, q, q).float().square().mean().backward()
                self.assertTrue(torch.isfinite(q.grad).all())
        parameter = torch.nn.Parameter(torch.randn(8192, device='cuda'))
        optimizer = bnb.optim.AdamW8bit([parameter], lr=0.001)
        before = parameter.detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            parameter.square().mean().backward()
            optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertFalse(torch.equal(before, parameter))
        torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], verbosity=2)

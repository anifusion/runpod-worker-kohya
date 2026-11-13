"""Synthetic compatibility checks. No model downloads, provider calls or real datasets."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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
from transformers import CLIPTextConfig, CLIPTextModelWithProjection
import sdxl_train_network
from library import train_util, utils, sdxl_model_util, sdxl_original_unet
from networks.lora import LoRAModule, create_network

CLIPTextModel = sdxl_model_util.CLIPTextModel


class RuntimeCompatibility(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_training_cli_contract(self):
        args = sdxl_train_network.setup_parser().parse_args([
            '--network_module', 'networks.lora', '--network_args', 'text_encoder_mlp_only=true',
            '--save_model_as', 'safetensors',
            '--gradient_checkpointing', '--no_half_vae', '--cache_latents',
            '--tokenizer_cache_dir', '/tokenizer_cache', '--http-log',
            '--http-log-endpoint', 'https://example.test/log', '--http-log-token', 'synthetic',
            '--optimizer_type', 'AdamW8bit', '--resolution', '1024,1024'])
        self.assertTrue(args.gradient_checkpointing)
        self.assertEqual(args.optimizer_type, 'AdamW8bit')
        self.assertEqual(args.network_args, ['text_encoder_mlp_only=true'])

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
            max_position_embeddings=8, bos_token_id=1, eos_token_id=2, pad_token_id=0))
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

    def test_sdxl_text_checkpoint_conversion_and_conditioning(self):
        config = CLIPTextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=12, num_attention_heads=4, max_position_embeddings=8,
            projection_dim=32, eos_token_id=2, bos_token_id=1, pad_token_id=0)
        encoder1 = CLIPTextModel(config)
        encoder2 = CLIPTextModelWithProjection(config)
        checkpoint = sdxl_model_util.convert_text_encoder_2_state_dict_to_sdxl(
            encoder2.state_dict(), None)
        checkpoint = {'conditioner.embedders.1.model.' + k: v for k, v in checkpoint.items()}
        restored, _ = sdxl_model_util.convert_sdxl_text_encoder_2_checkpoint(checkpoint, 8)
        copy = CLIPTextModelWithProjection(config)
        sdxl_model_util._load_state_dict_on_device(copy, restored, 'cpu')
        for key, value in encoder2.state_dict().items():
            torch.testing.assert_close(copy.state_dict()[key], value, rtol=0, atol=0)
        tokens = torch.tensor([[[1, 5, 7, 2, 0, 0, 0, 0]]])
        tokenizer = SimpleNamespace(model_max_length=8, eos_token_id=2)
        hidden1, hidden2, pooled = train_util.get_hidden_states_sdxl(
            None, tokens, tokens, tokenizer, tokenizer, encoder1, copy)
        self.assertEqual(tuple(hidden1.shape), (1, 8, 32))
        self.assertEqual(tuple(hidden2.shape), (1, 8, 32))
        self.assertEqual(tuple(pooled.shape), (1, 32))
        sum(value.square().mean() for value in (hidden1, hidden2, pooled)).backward()
        for encoder in (encoder1, copy):
            gradient = encoder.text_model.embeddings.token_embedding.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_text_encoder_lora_targets_and_backward(self):
        config = CLIPTextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=8,
            projection_dim=32, eos_token_id=2, bos_token_id=1, pad_token_id=0)
        config._attn_implementation = 'sdpa'
        encoders = [CLIPTextModel(config), CLIPTextModelWithProjection(config)]
        for encoder in encoders:
            encoder.requires_grad_(False)
        network = create_network(1.0, 4, 1.0, None, encoders, torch.nn.Identity(),
            text_encoder_mlp_only='true')
        expected = {f'lora_te{i}_text_model_encoder_layers_0_mlp_fc{j}'
                    for i in (1, 2) for j in (1, 2)}
        self.assertEqual({module.lora_name for module in network.text_encoder_loras}, expected)
        network.apply_to(encoders, None, apply_text_encoder=True, apply_unet=False)
        for encoder in encoders:
            encoder.gradient_checkpointing_enable()
            encoder.text_model.embeddings.requires_grad_(True)
            encoder.train()
            encoder(torch.tensor([[1, 5, 7, 2, 0, 0, 0, 0]])).last_hidden_state.square().mean().backward()
        for module in network.text_encoder_loras:
            self.assertIsNotNone(module.lora_up.weight.grad)
            self.assertTrue(torch.isfinite(module.lora_up.weight.grad).all())

    def test_sdxl_text_encoder_lora_inventory(self):
        # Full-size SDXL encoders built the way the checkpoint loader builds them,
        # on the meta device. Saved LoRAs must keep the MLP-only key inventory.
        from accelerate import init_empty_weights
        shapes = [(1, 768, 3072, 12, 12, 'quick_gelu'), (2, 1280, 5120, 32, 20, 'gelu')]
        encoders = []
        with init_empty_weights():
            for index, hidden, intermediate, layers, heads, activation in shapes:
                config = CLIPTextConfig(vocab_size=49408, hidden_size=hidden,
                    intermediate_size=intermediate, num_hidden_layers=layers,
                    num_attention_heads=heads, max_position_embeddings=77, hidden_act=activation,
                    projection_dim=hidden, pad_token_id=1, bos_token_id=0, eos_token_id=2)
                model_class = CLIPTextModel if index == 1 else CLIPTextModelWithProjection
                encoders.append(model_class._from_config(config) if index == 1 else model_class(config))
        network = create_network(1.0, 4, 1.0, None, encoders, torch.nn.Identity(),
            text_encoder_mlp_only='true')
        expected = {f'lora_te{index}_text_model_encoder_layers_{layer}_mlp_fc{j}'
                    for index, _, _, layers, _, _ in shapes for layer in range(layers) for j in (1, 2)}
        self.assertEqual(len(expected), 88)
        self.assertEqual({module.lora_name for module in network.text_encoder_loras}, expected)

    def test_training_cosine_scheduler(self):
        args = sdxl_train_network.setup_parser().parse_args([
            '--optimizer_type', 'AdamW', '--learning_rate', '0.001',
            '--lr_scheduler', 'cosine', '--lr_warmup_steps', '2', '--max_train_steps', '8'])
        parameter = torch.nn.Parameter(torch.ones(1))
        _, _, optimizer = train_util.get_optimizer(args, [parameter])
        scheduler = train_util.get_scheduler_fix(args, optimizer, 1)
        rates = [scheduler.get_last_lr()[0]]
        for _ in range(8):
            optimizer.step()
            scheduler.step()
            rates.append(scheduler.get_last_lr()[0])
        self.assertEqual(rates[0], 0)
        self.assertAlmostEqual(rates[2], 0.001)
        self.assertAlmostEqual(rates[-1], 0)
        self.assertTrue(all(a >= b for a, b in zip(rates[2:], rates[3:])))

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

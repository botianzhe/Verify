"""CPU tests for I/O, simulated forgery and independent copyright extraction."""

import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from verify.forgery import blend_forgery
from verify.image_io import load_image, save_image, load_mask, mask_to_pil
from verify.pipeline import VerifyConfig, VerifyPipeline


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.image = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(42)) * 2 - 1
        self.mask = torch.zeros(1, 1, 256, 256)
        self.mask[:, :, 70:180, 80:190] = 1

    def test_numeric_files_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for suffix in ('.pt','.npz'):
                path = Path(temporary)/('image'+suffix)
                with self.assertRaises(ValueError):save_image(self.image,path)
                with self.assertRaises(ValueError):load_image(path)
                with self.assertRaises(ValueError):load_mask(path)
                self.assertFalse(path.exists())

    def test_png_mask_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'mask.png'
            mask_to_pil(self.mask,binary=True).save(path)
            self.assertTrue(torch.equal(load_mask(path), self.mask))

    def test_png_error_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'image.png'
            save_image(self.image, path)
            self.assertLessEqual(float((load_image(path) - self.image).abs().max()), 1 / 255 + 1e-7)

    def test_positive_tensor_range_is_never_guessed(self):
        bright = torch.full_like(self.image, .75)
        self.assertTrue(torch.equal(load_image(bright), bright))
        self.assertTrue(torch.equal(load_image(bright, input_range='zero_one'), bright * 2 - 1))
        with self.assertRaises(ValueError):
            load_image(self.image, input_range='zero_one')

    def test_forge_single_image_background_seed_and_rng(self):
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        for seed in range(12):
            for method in ('hard', 'feather', 'random'):
                result = blend_forgery(self.image, mask=self.mask, seed=seed, method=method)
                repeated = blend_forgery(self.image, mask=self.mask, seed=seed, method=method)
                self.assertTrue(torch.equal(result['image'], repeated['image']))
                self.assertEqual(float(((result['image'] - self.image) * (1-self.mask)).abs().max()), 0)
                self.assertTrue(torch.isfinite(result['image']).all())
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_empty_mask_identity(self):
        result = blend_forgery(self.image, mask=torch.zeros_like(self.mask))
        self.assertTrue(torch.equal(result['image'], self.image))

    def test_invalid_mask(self):
        for mask in (self.mask * .5, self.mask + float('nan'), self.mask[:, :, :100]):
            with self.assertRaises(ValueError):
                blend_forgery(self.image, mask=mask)

    def test_extract_never_calls_localizer_or_vae(self):
        class Decoder(nn.Module):
            def forward(self, x):
                return torch.cat([torch.ones(x.shape[0], 100), torch.zeros(x.shape[0], 1024)], 1)
        class Unavailable(nn.Module):
            def forward(self, *args):
                raise AssertionError('extract must not invoke localizer')
        pipe = VerifyPipeline.__new__(VerifyPipeline)
        pipe.config = VerifyConfig()
        pipe.device = torch.device('cpu')
        pipe.dtype = torch.float32
        pipe.watermark = nn.Module()
        pipe.watermark.decoder = Decoder()
        pipe.localizer = Unavailable()
        pipe.vae = None
        result = pipe.extract(self.image)
        self.assertEqual(result['watermark'].shape, (1, 100))
        self.assertTrue(torch.all(result['watermark'] == 1))


if __name__ == '__main__':
    unittest.main()

"""Small, CPU-only regression tests; no checkpoint or VAE download required."""

import tempfile
import unittest
import argparse
from pathlib import Path

import torch
from torch import nn

from verify.cli import _bits
from verify.pipeline import VerifyConfig, VerifyPipeline


class CheckpointTests(unittest.TestCase):
    def pipeline(self, use_ema=None):
        pipe = VerifyPipeline.__new__(VerifyPipeline)
        pipe.config = VerifyConfig()
        pipe.use_ema = use_ema
        pipe.strict = False  # these tests also inspect deliberately partial states
        pipe.watermark = nn.Linear(3, 2)
        pipe.localizer = nn.Linear(2, 1)
        pipe.recovery = nn.Linear(2, 2)
        return pipe

    def state(self, pipe):
        return {
            prefix + key: value.clone()
            for prefix, module in (
                ('CopyRightWMmodel.', pipe.watermark),
                ('segmodel.', pipe.localizer),
                ('recoverymodel.', pipe.recovery),
            )
            for key, value in module.state_dict().items()
        }

    def load(self, pipe, payload):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            torch.save(payload, path)
            pipe._load_checkpoint(path)

    def test_config_and_bits(self):
        self.assertEqual(VerifyConfig().watermark_bits, 100)
        self.assertEqual(_bits('01' * 50), [0, 1] * 50)
        for value in ('01' * 32, 'x' * 100):
            with self.assertRaises(argparse.ArgumentTypeError):
                _bits(value)
        for kwargs in ({'image_size': 128}, {'latent_bits': 512},
                       {'wm_epsilon_px': float('nan')}, {'watermark_bits': 0}):
            with self.assertRaises(ValueError):
                VerifyConfig(**kwargs)
        with self.assertRaises(ValueError):
            VerifyConfig(wm_epsilon_px=-1)

    def test_plain_and_wrapped(self):
        for wrapper in (None, 'model', 'state_dict'):
            pipe = self.pipeline()
            state = {key: torch.full_like(value, 2) for key, value in self.state(pipe).items()}
            self.load(pipe, state if wrapper is None else {wrapper: state})
            self.assertTrue(pipe.checkpoint_load_report['complete_model_load'])
            self.assertFalse(pipe.ema_applied)
            self.assertTrue(torch.all(pipe.watermark.weight == 2))

    def test_partial_load_never_applies_ema(self):
        pipe = self.pipeline()
        original = pipe.watermark.bias.detach().clone()
        state = {key: torch.full_like(value, 2) for key, value in self.state(pipe).items()}
        shadows = [torch.full_like(value, 9) for value in state.values()]
        state['CopyRightWMmodel.bias'] = torch.zeros(5)
        state['unknown'] = torch.zeros(1)
        state['segmodel.bias'] = 'not a tensor'
        self.load(pipe, {'model': state, 'ema': {'shadow_params': shadows}})
        self.assertFalse(pipe.ema_applied)
        self.assertEqual(len(pipe.checkpoint_load_report['skipped_keys']), 3)
        self.assertEqual(len(pipe.checkpoint_load_report['missing_keys']), 2)
        self.assertTrue(torch.equal(pipe.watermark.bias, original))
        self.assertTrue(torch.all(pipe.watermark.weight == 2))

    def test_ema_metadata_and_order(self):
        for model_is_ema, override, expected in (
            (False, None, True), (True, None, False),
            (False, False, False), (True, True, True),
        ):
            pipe = self.pipeline(override)
            state = self.state(pipe)
            shadow = [torch.full_like(value, 9) for value in state.values()]
            self.load(pipe, {'model': state, 'model_is_ema': model_is_ema,
                             'ema': {'shadow_params': shadow}})
            self.assertEqual(pipe.ema_applied, expected)
            if expected:
                self.assertTrue(torch.all(pipe.watermark.weight == 9))
        pipe = self.pipeline(True)
        state = self.state(pipe)
        shadow = [torch.zeros_like(value) for value in state.values()]
        with self.assertRaises(ValueError):
            self.load(pipe, {'model': dict(reversed(list(state.items()))),
                             'ema': {'shadow_params': shadow}})
        self.assertFalse(pipe.ema_applied)

    def test_strict_rejects_partial(self):
        pipe = self.pipeline()
        pipe.strict = True
        state = self.state(pipe)
        del state['CopyRightWMmodel.bias']
        with self.assertRaisesRegex(ValueError, 'Refusing inference'):
            self.load(pipe, {'model': state})


if __name__ == '__main__':
    unittest.main()

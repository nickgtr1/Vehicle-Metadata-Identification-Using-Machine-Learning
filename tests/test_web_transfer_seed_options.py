"""Tests for the seed option, the run-config overlay and the validation device.

These cover the three behaviours added to the make/body runner:

* one prepared split can be trained with several optimisation seeds, while the split
  construction seed stays fixed so every seed sees the same holdout;
* the fresh-process reload check reads the seed and device the run recorded;
* validation during training runs on the same device as that reload check.

No training, no GPU and no dataset access is required.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_web_transfer as runner
from vehicle_id.attributes.modelling import predict_loader


class SeedOptionTests(unittest.TestCase):
    def test_split_seed_is_fixed_at_the_pilot_value(self):
        self.assertEqual(runner.SPLIT_SEED, 36127)

    def test_parser_defaults_the_run_seed_to_the_split_seed(self):
        args = runner.build_parser().parse_args(
            ['prepare', '--root', '.', '--audit', '.', '--prepared', 'out'])
        self.assertEqual(args.seed, runner.SPLIT_SEED)

    def test_parser_accepts_an_overriding_run_seed(self):
        args = runner.build_parser().parse_args(
            ['train', '--root', '.', '--audit', '.', '--prepared', 'prep',
             '--output', 'out', '--weights', 'w', '--seed', '36129'])
        self.assertEqual(args.seed, 36129)

    def test_prepare_keeps_the_split_seed_and_records_the_run_seed(self):
        """prepare() must build the split with SPLIT_SEED but store args.seed."""
        seen = {}
        frame = pd.DataFrame({
            'relative_key': ['a', 'b', 'c', 'd'],
            'sha256': ['1', '2', '3', '4'],
            'make_name': ['X', 'X', 'Y', 'Y'],
            'split': ['train', 'train', 'train', 'train'],
        })

        def fake_pilot_split(df, label, cap, seed, reserved):
            seen['seed'] = seed
            fit = df.iloc[:2].assign(experiment_split='train')
            val = df.iloc[2:].assign(experiment_split='validation')
            return fit, val, df.iloc[:0], ['X', 'Y'], {'official_train_rows': len(df)}

        original = (runner.audited_frame, runner.pilot_split, runner.validate_split)
        runner.audited_frame = lambda audit: frame
        runner.pilot_split = fake_pilot_split
        runner.validate_split = lambda *args, **kwargs: None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                prepared = Path(tmp) / 'prepared'
                args = runner.build_parser().parse_args(
                    ['prepare', '--root', tmp, '--audit', tmp,
                     '--prepared', str(prepared), '--task', 'make', '--cap', '10',
                     '--seed', '36128'])
                runner.prepare(args)
                config = json.loads((prepared / 'config.json').read_text(encoding='utf-8'))
        finally:
            runner.audited_frame, runner.pilot_split, runner.validate_split = original

        self.assertEqual(seen['seed'], runner.SPLIT_SEED,
                         'the split must be built with the fixed split seed')
        self.assertEqual(config['seed'], 36128,
                         'the run seed must be recorded in the prepared config')
        self.assertEqual(config['task'], 'make')


class RunConfigOverlayTests(unittest.TestCase):
    def test_seed_and_device_come_from_the_run_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / 'config.json').write_text(
                json.dumps({'seed': 36129, 'device': 'cuda'}), encoding='utf-8')
            config = runner.resolve_run_config({'seed': 36127, 'task': 'make'}, output)
        self.assertEqual(config['seed'], 36129)
        self.assertEqual(config['device'], 'cuda')
        self.assertEqual(config['task'], 'make')

    def test_prepared_values_are_kept_when_the_run_record_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = runner.resolve_run_config({'seed': 36127}, Path(tmp))
        self.assertEqual(config, {'seed': 36127})

    def test_only_seed_and_device_are_overlaid(self):
        """A run record must not be able to rewrite the prepared experiment."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / 'config.json').write_text(json.dumps({
                'seed': 36128, 'device': 'cpu', 'task': 'body_type',
                'manifest_sha256': 'not-the-prepared-manifest',
            }), encoding='utf-8')
            config = runner.resolve_run_config(
                {'seed': 36127, 'task': 'make', 'manifest_sha256': 'abc'}, output)
        self.assertEqual(config['task'], 'make')
        self.assertEqual(config['manifest_sha256'], 'abc')
        self.assertEqual(config['seed'], 36128)
        self.assertEqual(config['device'], 'cpu')


class ValidationDeviceTests(unittest.TestCase):
    def _model_and_loader(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 3))
        inputs = torch.randn(6, 1, 2, 2)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
        return model, loader

    def test_predictions_match_a_direct_cpu_pass(self):
        model, loader = self._model_and_loader()
        expected = predict_loader(model, loader, 'cpu')
        got = runner.validation_predictions(model, loader)
        self.assertEqual(expected[0], got[0])
        self.assertEqual(expected[1], got[1])
        self.assertEqual(len(got[2]), 6)

    def test_model_is_left_on_the_device_it_started_from(self):
        model, loader = self._model_and_loader()
        runner.validation_predictions(model, loader)
        self.assertEqual(next(model.parameters()).device.type, 'cpu')
        if torch.cuda.is_available():
            model = model.to('cuda')
            runner.validation_predictions(model, loader)
            self.assertEqual(next(model.parameters()).device.type, 'cuda',
                             'a CUDA model must be returned to CUDA after CPU validation')


if __name__ == '__main__':
    unittest.main()

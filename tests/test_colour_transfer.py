import sys
from pathlib import Path
import unittest
import pandas as pd
from torchvision.models import resnet18
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_colour_transfer import validate_frozen_split, configure_trainable


class ColourTransferTests(unittest.TestCase):
    def fixture(self):
        rows = [{'relative_key': str(i), 'sha256': str(i), 'color_name': 'black',
                 'image_path': str(i) + '.jpg', 'experiment_split': 'train' if i == 0 else 'validation',
                 'split': 'train'} for i in range(2)]
        return pd.DataFrame(rows)

    def test_fixed_split_passes(self):
        f = self.fixture(); train, val, classes = validate_frozen_split(f, f)
        self.assertEqual((len(train), len(val), classes), (1, 1, ['black']))

    def test_official_test_rejected(self):
        f = self.fixture(); official = f.copy(); official.loc[1, 'split'] = 'test'
        with self.assertRaises(ValueError): validate_frozen_split(f, official)

    def test_duplicate_content_rejected(self):
        f = self.fixture(); f.loc[1, 'sha256'] = '0'
        with self.assertRaises(ValueError): validate_frozen_split(f, f)

    def test_path_or_label_change_rejected(self):
        f = self.fixture(); tampered = f.copy(); tampered.loc[0, 'image_path'] = 'different.jpg'
        with self.assertRaises(ValueError): validate_frozen_split(tampered, f)

    def test_phase_and_batchnorm_freezing(self):
        m = resnet18(weights=None); configure_trainable(m, False)
        self.assertTrue(all(p.requires_grad == name.startswith('fc.') for name, p in m.named_parameters()))
        configure_trainable(m, True)
        self.assertTrue(all(p.requires_grad == name.startswith(('fc.', 'layer4.')) for name, p in m.named_parameters()))
        self.assertTrue(all(not x.training for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)))

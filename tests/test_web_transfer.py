import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_local_compcars
from train_web_transfer import loader_view, validate_split
from vehicle_id.attributes.dataset import CompCarsMakeDataset
from vehicle_id.attributes.modelling import AttributePredictor, build_model, normalise_image
from vehicle_id.pipeline import analyse_vehicle_image


class WebTransferTests(unittest.TestCase):
    def test_official_zero_requires_missing_body_label(self):
        match = audit_local_compcars.body_label_matches
        self.assertTrue(match(0, float('nan'), ['sedan', 'convertible']))
        self.assertFalse(match(0, 'convertible', ['sedan', 'convertible']))
        self.assertTrue(match(2, 'convertible', ['sedan', 'convertible']))
        self.assertFalse(match(3, 'convertible', ['sedan', 'convertible']))
        self.assertFalse(match(-1, float('nan'), ['sedan', 'convertible']))

    def test_loader_crop_matches_predictor_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
            Image.fromarray(pixels).save(root / 'car.png')
            source = pd.DataFrame([{'bbox': '(1, 2, 12, 13)', 'image_path': 'car.png', 'make_name': 'A'}])
            view = loader_view(source)
            self.assertEqual(source.bbox.iloc[0], '(1, 2, 12, 13)')
            self.assertEqual(view.bbox.iloc[0], (0, 1, 12, 13))
            dataset = CompCarsMakeDataset(view, root=root, image_size=16,
                                         transform=normalise_image, classes=['A', 'B'])
            checkpoint = root / 'synthetic.pt'
            model = build_model('tiny_cnn', 2)
            torch.save({'config': {'task': 'make', 'architecture': 'tiny_cnn', 'image_size': 16},
                        'classes': ['A', 'B'], 'synthetic': True, 'state_dict': model.state_dict()}, checkpoint)
            predictor = AttributePredictor(checkpoint, allow_synthetic=True)
            captured = []
            hook = predictor.model.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].clone()))
            result = analyse_vehicle_image(root / 'car.png', predictors={'make': predictor},
                                           detections=[{'bbox': [0, 1, 12, 13], 'confidence': .9}])
            hook.remove()
            self.assertTrue(torch.equal(dataset[0][0], captured[0][0]))
            profile = result['vehicles'][0]['vehicle_profile']
            self.assertEqual(profile['status']['make'], 'synthetic_smoke_only')
            self.assertEqual(profile['status']['body_type'], 'not_assessed')
            with self.assertRaises(ValueError):
                analyse_vehicle_image(root / 'car.png', predictors={'body_type': predictor}, detections=[])

    def test_split_guards_source_and_test_content(self):
        frame = pd.DataFrame([{'relative_key': str(i), 'image_path': f'{i}.png',
                               'sha256': str(i), 'make_name': 'A', 'bbox': '(1, 1, 4, 4)',
                               'split': 'train', 'experiment_split': 'train' if i == 0 else 'validation'}
                              for i in range(2)])
        fit, val, classes = validate_split(frame, frame, 'make_name', {'test'})
        self.assertEqual((len(fit), len(val), classes), (1, 1, ['A']))
        with self.assertRaises(ValueError):
            validate_split(frame, frame, 'make_name', {'1'})
        for column, value in [('bbox', '(2, 1, 4, 4)'), ('split', 'test'),
                              ('image_path', 'wrong.png'), ('make_name', 'B'), ('sha256', 'other')]:
            changed = frame.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_split(changed, frame, 'make_name', set())

    def test_invalid_official_bbox_is_rejected(self):
        for bbox in ['(0, 1, 10, 10)', '(1, 1, 10)', '(1.5, 1, 10, 10)']:
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                loader_view(pd.DataFrame({'bbox': [bbox]}))


if __name__ == '__main__':
    unittest.main()

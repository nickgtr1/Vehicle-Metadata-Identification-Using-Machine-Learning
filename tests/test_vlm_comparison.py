"""Independent answers and local predictions must remain distinguishable."""
import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from vehicle_id.schema import VehicleProfile
from vehicle_id.vlm_fallback import apply_vlm_fallback, comparison_rows


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.image = Path(self.tmp.name) / 'car.png'
        Image.new('RGB', (100, 80), 'white').save(self.image)
        self.vehicle = {
            'vehicle_crop_id': 'vehicle_001', 'bbox_xyxy': [20, 20, 60, 60],
            'vehicle_profile': {
                'make': 'CV make', 'model': None, 'body_type': 'sedan', 'colour': 'white',
                'confidence': {'make': .7, 'body_type': .8, 'colour': .9},
                'status': {'make': 'predicted', 'body_type': 'predicted',
                           'colour': 'predicted', 'model': 'not_assessed'},
            },
        }

    def run_with(self, vehicles, predictors, response, mode='tuned'):
        client = types.ModuleType('vehicle_id.vlm.client')
        client.predict_profile_tuned = Mock(side_effect=response)
        client.predict_profile = Mock(side_effect=response)
        with patch.dict(sys.modules, {'vehicle_id.vlm.client': client}):
            result, error = apply_vlm_fallback(
                self.image, vehicles, predictors, enabled=True, mode=mode)
        return result, error, client

    def test_one_call_preserves_cv_and_retains_independent_vlm(self):
        before = copy.deepcopy(self.vehicle['vehicle_profile'])
        crop_sizes, temporary_paths = [], []
        def reply(path):
            temporary_paths.append(Path(path))
            with Image.open(path) as crop:
                crop_sizes.append(crop.size)
            return VehicleProfile(make='VLM make', model='VLM model', body_type='SUV', colour='red'), None
        vehicles, error, client = self.run_with(
            [self.vehicle], {'make': object(), 'body_type': object(), 'colour': object()}, reply)
        self.assertIsNone(error)
        client.predict_profile_tuned.assert_called_once()
        self.assertEqual(crop_sizes, [(52, 52)])
        self.assertTrue(all(not p.exists() for p in temporary_paths))
        rec = vehicles[0]
        self.assertEqual(rec['attribute_profile'], before)
        self.assertEqual(rec['vehicle_profile']['make'], 'CV make')
        self.assertEqual(rec['vehicle_profile']['body_type'], 'sedan')
        self.assertEqual(rec['vehicle_profile']['model'], 'VLM model')
        self.assertNotIn('colour', rec['vlm_profile'])
        rows = {row['attribute']: row for row in comparison_rows(vehicles)}
        self.assertEqual(rows['make']['VLM'], 'VLM make')
        self.assertEqual(rows['model']['attribute model / baseline'], 'not assessed')
        self.assertEqual(rows['colour']['VLM'], 'not assessed')

    def test_missing_local_fields_fill_without_becoming_local_predictions(self):
        self.vehicle['vehicle_profile']['make'] = None
        vehicles, _, _ = self.run_with([self.vehicle], {}, [
            (VehicleProfile(make='VLM make', model='C', body_type='SUV'), None)])
        self.assertEqual(vehicles[0]['vehicle_profile']['make'], 'VLM make')
        self.assertIsNone(vehicles[0]['attribute_profile']['make'])
        self.assertEqual(comparison_rows(vehicles)[0]['attribute model / baseline'], 'not assessed')

    def test_failure_keeps_cv_continues_batch_and_clears_old_vlm(self):
        before = copy.deepcopy(self.vehicle['vehicle_profile'])
        self.vehicle['vlm_profile'] = {'model': 'stale'}
        other = copy.deepcopy(self.vehicle)
        other['vehicle_crop_id'] = 'vehicle_002'
        paths = []
        def reply(path):
            paths.append(Path(path))
            if len(paths) == 1:
                raise RuntimeError('simulated failure')
            return VehicleProfile(model='new model'), None
        vehicles, error, client = self.run_with([self.vehicle, other], {}, reply)
        self.assertIn('simulated failure', error)
        self.assertEqual(client.predict_profile_tuned.call_count, 2)
        self.assertEqual(vehicles[0]['vehicle_profile'], before)
        self.assertEqual(vehicles[0]['vlm_profile'], {})
        self.assertEqual(vehicles[0]['vlm_status'], 'failed')
        self.assertEqual(vehicles[1]['vlm_profile']['model'], 'new model')
        self.assertTrue(all(not p.exists() for p in paths))

    def test_disabled_or_empty_needs_no_client(self):
        before = copy.deepcopy(self.vehicle)
        apply_vlm_fallback(self.image, [self.vehicle], {}, enabled=False)
        self.assertEqual(self.vehicle, before)
        self.assertEqual(apply_vlm_fallback(self.image, [], {}, enabled=True), ([], None))

    def test_unavailable_client_is_reported_without_losing_cv(self):
        before = copy.deepcopy(self.vehicle['vehicle_profile'])
        with patch.dict(sys.modules, {'vehicle_id.vlm.client': None}):
            vehicles, error = apply_vlm_fallback(self.image, [self.vehicle], {}, enabled=True)
        self.assertIn('ModuleNotFoundError', error)
        self.assertEqual(vehicles[0]['vehicle_profile'], before)
        self.assertEqual(vehicles[0]['attribute_profile'], before)
        self.assertEqual(vehicles[0]['vlm_profile'], {})
        self.assertEqual(vehicles[0]['vlm_status'], 'failed')

    def test_zero_shot_uses_constrained_prompt_and_can_compare_colour(self):
        vehicles, _, client = self.run_with([self.vehicle], {}, [
            (VehicleProfile(colour='red'), None)], mode='zero-shot')
        self.assertEqual(client.predict_profile.call_args.kwargs, {'prompt': 'constrained'})
        client.predict_profile_tuned.assert_not_called()
        self.assertEqual(vehicles[0]['vlm_profile']['colour'], 'red')

    def test_model_predictor_keeps_local_model_and_still_retains_vlm_answer(self):
        self.vehicle['vehicle_profile']['model'] = 'CV model'
        self.vehicle['vehicle_profile']['confidence']['model'] = .5
        vehicles, error, _client = self.run_with([self.vehicle], {'model': object()}, [
            (VehicleProfile(make='VLM make', model='VLM model', body_type='SUV'), None)])
        self.assertIsNone(error)
        profile = vehicles[0]['vehicle_profile']
        self.assertEqual(profile['model'], 'CV model')
        self.assertEqual(profile['make'], 'VLM make')
        self.assertEqual(vehicles[0]['vlm_profile']['model'], 'VLM model')
        rows = {row['attribute']: row for row in comparison_rows(vehicles)}
        self.assertEqual(rows['model']['attribute model / baseline'], 'CV model')
        self.assertEqual(rows['model']['VLM'], 'VLM model')

    def test_fully_covered_fields_make_no_vlm_call(self):
        vehicles, error, client = self.run_with(
            [self.vehicle], {'make': object(), 'model': object(), 'body_type': object()},
            [(VehicleProfile(), None)])
        self.assertIsNone(error)
        client.predict_profile_tuned.assert_not_called()
        self.assertEqual(vehicles[0]['vlm_profile'], {})


if __name__ == '__main__':
    unittest.main()

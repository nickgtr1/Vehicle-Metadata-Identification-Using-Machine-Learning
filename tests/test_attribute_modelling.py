import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from vehicle_id.attributes.dataset import CompCarsMakeDataset, resolve_image_path, parse_bbox
from vehicle_id.attributes.modelling import audit_manifest, split_training, build_model, metrics
from vehicle_id.pipeline import analyse_vehicle_image


class ModellingTests(unittest.TestCase):
    def test_published_notebook_path(self):
        root = Path(tempfile.gettempdir()).resolve()
        self.assertEqual(resolve_image_path('../../data/raw/compcars/image\\a.jpg', root), root / 'data/raw/compcars/image/a.jpg')

    def test_path_traversal_rejected(self):
        for p in ['../../../private.jpg', 'C:/private.jpg']:
            with self.assertRaises(ValueError): resolve_image_path(p, '.')

    def test_invalid_boxes(self):
        for b in ['(2,2,1,1)', '(0,0,1)', '(-1,0,2,2)', (0,0,float('nan'),3)]:
            with self.assertRaises(ValueError): parse_bbox(b)

    def test_shared_mapping_crop_and_no_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); Image.new('RGB', (16,16), 'red').save(p / 'a.png')
            frame = pd.DataFrame([{'image_path':'a.png', 'make_name':'B', 'bbox':'(1,1,12,12)'}])
            ds = CompCarsMakeDataset(frame, root=p, classes=['A','B'], image_size=8)
            x,y = ds[0]; self.assertEqual(y,1); self.assertEqual(tuple(x.shape), (3,8,8))
            self.assertAlmostEqual(float(x[0].mean()),1)
            ds2 = CompCarsMakeDataset(frame.drop(columns='bbox'), root=p, classes=['A','B'])
            self.assertEqual(ds2[0][1],1)
            with self.assertRaises(ValueError): CompCarsMakeDataset(frame, root=p, classes=['A'])

    @staticmethod
    def frame():
        return pd.DataFrame([{'image_path':f'data/{c}/{i}.png','make_name':c,
                              'split':'train' if i < 10 else 'test', 'group': f'{c}-{i}'}
                             for c in ['A','B'] for i in range(12)])

    def test_repeatable_split_keeps_test(self):
        df = self.frame()
        a,b,t,c = split_training(df,'make_name',.2,36127)
        a2,b2,_,_ = split_training(df,'make_name',.2,36127)
        self.assertEqual(a.image_path.tolist(),a2.image_path.tolist())
        self.assertEqual(b.image_path.tolist(),b2.image_path.tolist())
        self.assertEqual(set(t.image_path),set(df[df.split=='test'].image_path))
        self.assertFalse(set(a.image_path)&set(b.image_path))

    def test_group_conflict_rejected(self):
        df=self.frame(); df['group']='same'
        with self.assertRaises(ValueError): split_training(df,'make_name',.2,36127,'group')

    def test_duplicate_path_audit(self):
        df=self.frame(); df.loc[11,'image_path']=df.loc[0,'image_path']
        result=audit_manifest(df,'make_name','.')
        self.assertEqual(result['train_test_path_overlap'],1)

    def test_metrics(self):
        result=metrics([0,1],[0,0],['A','B'])
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['confusion_matrix'],[[1,0],[1,0]])

    def test_resnet_heads_no_download(self):
        torch.set_num_threads(2)
        for n in [75,12,10]:
            model=build_model('resnet18',n).eval()
            with torch.inference_mode(): self.assertEqual(tuple(model(torch.zeros(1,3,32,32)).shape),(1,n))

    def test_pipeline_no_weights_and_no_detections(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.png'; Image.new('RGB',(40,40)).save(p)
            with self.assertRaises(FileNotFoundError): analyse_vehicle_image(p)
            self.assertEqual(analyse_vehicle_image(p,detections=[])['vehicles'],[])
            result=analyse_vehicle_image(p,detections=[{'bbox':[0,0,20,20],'confidence':.9}])
            self.assertEqual(result['vehicles'][0]['vehicle_profile']['status']['make'],'not_assessed')


if __name__=='__main__': unittest.main()

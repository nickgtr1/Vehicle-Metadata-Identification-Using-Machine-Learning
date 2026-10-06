import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import pandas as pd
import numpy as np
from scipy.io import savemat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from train_real_compcars_pilots import pilot_split, body_eligibility
from audit_local_compcars import mapped_frame


class PilotTests(unittest.TestCase):
    def test_missing_body_type_not_convertible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); misc=root/'data/data/misc'; misc.mkdir(parents=True)
            pd.DataFrame({'model_id':[1,2],'type':[0,12]}).to_csv(misc/'attributes.txt',sep=' ',index=False)
            names=np.empty((1,12),dtype=object)
            for i in range(12): names[0,i]=f'class{i+1}'
            names[0,11]='convertible'; savemat(misc/'car_type.mat',{'types':names})
            df=pd.DataFrame({'model_id':[1,2],'car_type_name':['convertible','convertible']})
            valid,excluded=body_eligibility(df,root)
            self.assertEqual(valid.model_id.tolist(),[2]); self.assertEqual(excluded.model_id.tolist(),[1])

    def test_hash_group_split_and_official_test_exclusion(self):
        rows=[]
        for label in ['A','B']:
            for i in range(20):
                rows.append({'relative_key':f'{label}-{i}','sha256':f'{label}-{i}',
                             'make_name':label,'split':'train' if i<18 else 'test'})
        rows.append({**rows[18],'relative_key':'duplicate_of_test','split':'train'})
        df=pd.DataFrame(rows)
        fit,val,excluded,classes,info=pilot_split(df,'make_name',10,36127)
        self.assertEqual(classes,['A','B']); self.assertEqual(len(excluded),1)
        self.assertEqual(len(fit),16); self.assertEqual(len(val),4)
        self.assertFalse(set(fit.sha256)&set(val.sha256))
        self.assertFalse(set(pd.concat([fit,val]).sha256)&set(df[df.split=='test'].sha256))
        again=pilot_split(df,'make_name',10,36127)
        self.assertEqual(fit.relative_key.tolist(),again[0].relative_key.tolist())

    def test_manifest_mapping_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.csv'
            pd.DataFrame([{'image_path':'../../data/raw/compcars/sv_data\\image\\1\\a.jpg',
                           'color_name':'red','split':'train'}]).to_csv(path,index=False)
            before=path.read_bytes(); df=mapped_frame(path,'surveillance')
            self.assertEqual(df.image_path.iloc[0],'sv_data/sv_data/image/1/a.jpg')
            self.assertEqual(path.read_bytes(),before)

    def test_archive_crc_checker(self):
        script=REPO/'scripts/compcars_archive_integrity.py'
        spec=importlib.util.spec_from_file_location('crc_audit',script)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); archive=root/'a.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                z.writestr('fixture/a.txt','abc')
            entry=list(mod.entries(archive))[0]
            (root/'fixture').mkdir(); (root/'fixture/a.txt').write_text('abc')
            self.assertEqual(mod.check((root,entry))['status'],'ok')
            (root/'fixture/a.txt').write_text('abd')
            self.assertEqual(mod.check((root,entry))['status'],'mismatch')
            self.assertEqual(mod.check((root/'missing',entry))['status'],'missing_or_unreadable')

if __name__=='__main__': unittest.main()

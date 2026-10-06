"""Audit original team manifests against local public images and official labels.

No source manifests/images modified. SHA-256s are reused from the complete CRC audit.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from vehicle_id.attributes.modelling import save_json, sha256, metadata

COLOURS={-1:'unrecognized',0:'black',1:'white',2:'red',3:'yellow',4:'blue',5:'green',6:'purple',7:'brown',8:'champagne',9:'silver'}

def mapped_frame(path,kind):
    df=pd.read_csv(path)
    df['source_image_path']=df.image_path
    prefix='../../data/raw/compcars/'
    keys=[]; paths=[]
    for raw in df.image_path:
        text=raw.replace('\\','/')
        if not text.startswith(prefix) or '..' in text[len(prefix):].split('/'):
            raise ValueError('Unexpected manifest path')
        tail=text[len(prefix):]
        if kind=='surveillance':
            if not tail.startswith('sv_data/image/'): raise ValueError('Unexpected surveillance path')
            keys.append(tail[len('sv_data/image/'):]); paths.append('sv_data/'+tail)
        else:
            if not tail.startswith('image/'): raise ValueError('Unexpected web path')
            keys.append(tail[len('image/'):]); paths.append('data/data/'+tail)
    df['relative_key']=keys; df['image_path']=paths
    return df

def inspect_image(args):
    root,row=args
    result={'relative_key':row['relative_key'],'split':row['split'],'status':'ok'}
    try:
        # Read metadata/decodability only. No model predictions on official test images.
        image=cv2.imdecode(np.fromfile(root/row['image_path'],dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None: raise ValueError('decode failed')
        h,w=image.shape[:2]; result.update(width=w,height=h)
        if 'bbox' in row:
            x1,y1,x2,y2=ast.literal_eval(row['bbox'])
            if not(1<=x1<x2<=w and 1<=y1<y2<=h):
                raise ValueError('bbox outside image under official 1-based convention')
    except Exception as exc:
        result['status']='error'; result['error']=str(exc)
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--archives-root',required=True); p.add_argument('--integrity',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); root=Path(args.archives_root); integrity=Path(args.integrity); out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    proofs=json.loads((integrity/'integrity_summary.json').read_text())
    if set(proofs)!={'web','surveillance'} or not all(x['crc_matches_archive'] for x in proofs.values()):
        raise ValueError('Both full extraction CRC checks must pass first')
    save_json(out/'run_metadata.json',metadata({'operation':'real_image_and_label_audit','seed':36127}))
    summaries={}
    for kind,filename in [('web','compcars_classification.csv'),('surveillance','compcars_surveillance_classification.csv')]:
        source=repo/'data/manifests'/filename; df=mapped_frame(source,kind)
        base=root/('data/data' if kind=='web' else 'sv_data/sv_data')
        splits={s:set((base/('train_test_split/classification/'+s+'.txt' if kind=='web' else s+'_surveillance.txt')).read_text().splitlines()) for s in ['train','test']}
        split_mismatch=[]; label_mismatch=[]
        if kind=='surveillance':
            lookup={str(row[0][0]):int(row[1][0][0]) for row in loadmat(base/'color_list.mat')['color_list']}
        else:
            attrs=pd.read_csv(base/'misc/attributes.txt',sep=r'\s+').set_index('model_id')['type'].to_dict()
            types=[str(x[0]) for x in loadmat(base/'misc/car_type.mat')['types'][0]]
            makes=[str(x[0][0]) for x in loadmat(base/'misc/make_model_name.mat')['make_names']]
        for row in df.to_dict('records'):
            key=row['relative_key']
            if key not in splits[row['split']]: split_mismatch.append(key)
            if kind=='surveillance':
                if lookup.get(key)!=row['color_id'] or COLOURS.get(lookup.get(key))!=row['color_name']:
                    label_mismatch.append({'key':key,'field':'colour'})
            else:
                label=(base/'label'/Path(key).with_suffix('.txt')).read_text().splitlines()
                expected=tuple(map(int,label[2].split()))
                body_id=attrs.get(int(row['model_id']),0)
                if expected!=ast.literal_eval(row['bbox']): label_mismatch.append({'key':key,'field':'bbox'})
                if makes[int(row['make_id'])-1]!=row['make_name']: label_mismatch.append({'key':key,'field':'make'})
                if not body_id or types[body_id-1]!=row['car_type_name']: label_mismatch.append({'key':key,'field':'body_type'})
        checked=pd.read_csv(integrity/f'{kind}_file_integrity.csv',usecols=['name','sha256','status'])
        hashes=checked.set_index('name').sha256.to_dict()
        prefix='data/' if kind=='web' else 'sv_data/'
        df['sha256']=[hashes[p[len(prefix):]] for p in df.image_path]
        groups=df.groupby('sha256').agg(rows=('relative_key','size'),split_count=('split','nunique'))
        crosses=set(groups[groups.split_count>1].index)
        duplicates=df[df.sha256.isin(set(groups[groups.rows>1].index))]
        duplicates[['relative_key','split','sha256']].to_csv(out/f'{kind}_exact_duplicates.csv',index=False)
        records=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,result in enumerate(pool.map(inspect_image,((root,r) for r in df.to_dict('records'))),1):
                records.append(result)
                if i%10000==0: print(f'{kind}: {i}/{len(df)} images decoded',flush=True)
        pd.DataFrame(records).to_csv(out/f'{kind}_image_checks.csv',index=False)
        df[['source_image_path','image_path','relative_key','split','sha256']].to_csv(out/f'{kind}_resolved_index.csv',index=False)
        errors=[r for r in records if r['status']!='ok']
        summary={'rows':len(df),'manifest_sha256':sha256(source),'split_mismatches':split_mismatch,
                 'official_label_mismatches':label_mismatch,'image_errors':errors,
                 'exact_duplicate_groups':int((groups.rows>1).sum()),'cross_official_split_duplicate_groups':len(crosses),
                 'colour_classes':sorted(df.color_name.unique()) if kind=='surveillance' else None,
                 'scope':'Full manifest image decode + official labels/split membership + exact-byte duplicates; not a near-duplicate or physical-identity audit.'}
        summaries[kind]=summary
        save_json(out/'audit_summary.json',summaries)
        print(kind,json.dumps({k:v for k,v in summary.items() if k not in ['split_mismatches','official_label_mismatches','image_errors']}),flush=True)
    if any(x['split_mismatches'] or x['official_label_mismatches'] or x['image_errors'] for x in summaries.values()):
        raise SystemExit(2)

if __name__=='__main__': main()

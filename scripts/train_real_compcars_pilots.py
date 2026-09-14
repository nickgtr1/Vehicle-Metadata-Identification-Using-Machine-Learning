"""Bounded real-data pilots; official test never used for model evaluation.

Keeps team manifests unchanged. Exact-byte groups stay together and training copies
of test-duplicate content are excluded. This is NOT physical-vehicle disjointness.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import ast
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import DataLoader
from vehicle_id.attributes.dataset import CompCarsMakeDataset
from vehicle_id.attributes.modelling import (TASKS, build_model, normalise_image, metrics,
                                            predict_loader, save_json, metadata, sha256, AttributePredictor)
from audit_local_compcars import mapped_frame


def pilot_split(df,label,cap,seed,reserved_test_hashes=None):
    original_test_hashes=set(df.loc[df.split=='test','sha256']) | set(reserved_test_hashes or [])
    conflicting=set(df.groupby('sha256')[label].nunique().loc[lambda x:x>1].index)
    train=df[df.split=='train'].copy()
    excluded=train[train.sha256.isin(original_test_hashes|conflicting)].copy()
    eligible=train[~train.sha256.isin(original_test_hashes|conflicting)].copy()
    representatives=eligible.sort_values('relative_key').drop_duplicates('sha256')
    classes=sorted(train[label].unique().tolist())
    rng=np.random.default_rng(seed); fits=[]; vals=[]
    for name in classes:
        group=representatives[representatives[label]==name].sort_values('relative_key')
        if len(group)<2: raise ValueError(f'Not enough non-overlapping examples for {name}')
        selected=group.iloc[rng.permutation(len(group))[:cap]]
        nval=max(1,math.ceil(len(selected)*.2))
        vals.append(selected.iloc[:nval]); fits.append(selected.iloc[nval:])
    fit,val=pd.concat(fits).copy(),pd.concat(vals).copy()
    fit['experiment_split']='train'; val['experiment_split']='validation'
    assert not(set(fit.sha256)&set(val.sha256))
    assert not((set(fit.sha256)|set(val.sha256))&original_test_hashes)
    info={'official_train_rows':len(train),'official_test_rows_reserved':int((df.split=='test').sum()),
          'excluded_train_rows_duplicate_with_test_or_conflicting':len(excluded),
          'train_rows':len(fit),'validation_rows':len(val),'classes':len(classes),
          'train_validation_exact_hash_overlap':0,'pilot_official_test_exact_hash_overlap':0,
          'duplicate_representative_policy':'one image per identical-byte SHA256 group',
          'validation_distribution':'up to a per-class cap, NOT the natural full-dataset distribution',
          'physical_identity_disjointness_verified':False,'near_duplicate_audit_completed':False}
    return fit,val,excluded,classes,info


def body_eligibility(df,root):
    """Official type 0 means unavailable, not types[-1] (convertible)."""
    misc=root/'data/data/misc'
    attrs=pd.read_csv(misc/'attributes.txt',sep=r'\s+').set_index('model_id')['type']
    names=[str(x[0]) for x in loadmat(misc/'car_type.mat')['types'][0]]
    frame=df.copy(); frame['official_type_id']=frame.model_id.map(attrs)
    if frame.official_type_id.isna().any(): raise ValueError('Missing model in official attributes')
    unavailable=frame[frame.official_type_id==0].copy()
    unavailable['exclusion_reason']='official type=0: body-type label unavailable; not convertible'
    valid=frame[frame.official_type_id!=0].copy()
    for row in valid.to_dict('records'):
        identifier=int(row['official_type_id'])
        if not 1<=identifier<=len(names) or row['car_type_name']!=names[identifier-1]:
            raise ValueError('A nonzero body label still conflicts with official metadata')
    return valid,unavailable


def run_task(task,root,audit,out,device):
    result_dir=out/task; result_dir.mkdir(parents=True,exist_ok=False)
    kind='surveillance' if task=='colour' else 'web'
    repo=Path(__file__).resolve().parents[1]
    source=repo/'data/manifests'/('compcars_surveillance_classification.csv' if kind=='surveillance' else 'compcars_classification.csv')
    df=mapped_frame(source,kind)
    index=pd.read_csv(audit/f'{kind}_resolved_index.csv').set_index('relative_key')
    df['sha256']=df.relative_key.map(index.sha256)
    if df.sha256.isna().any(): raise ValueError('Missing verified image hashes')
    reserved_test_hashes=set(df.loc[df.split=='test','sha256'])
    eligibility={'original_manifest_rows':len(df),'missing_body_labels_excluded':0}
    if task=='body_type':
        df,unavailable=body_eligibility(df,root)
        unavailable[['relative_key','split','model_id','car_type_name','official_type_id','exclusion_reason']].to_csv(result_dir/'unavailable_body_labels.csv',index=False)
        eligibility.update(missing_body_labels_excluded=len(unavailable),
                           missing_body_labels_by_split=unavailable.split.value_counts().to_dict(),
                           eligible_rows=len(df))
    save_json(result_dir/'label_eligibility.json',eligibility)
    seed=36127; label=TASKS[task]; cap={'make':30,'body_type':150,'colour':200}[task]
    fit,val,excluded,classes,split_info=pilot_split(df,label,cap,seed,reserved_test_hashes)
    pd.concat([fit,val]).to_csv(result_dir/'pilot_split.csv',index=False)
    excluded[['relative_key','sha256',label]].to_csv(result_dir/'excluded_training_rows.csv',index=False)
    save_json(result_dir/'split_summary.json',split_info)
    pd.concat([fit,val]).groupby(['experiment_split',label]).size().rename('count').reset_index().to_csv(result_dir/'class_counts.csv',index=False)
    if kind=='web':
        # Official labels are 1-based xyxy. Convert only the in-memory loader view.
        def box_to_zero(value):
            x1,y1,x2,y2=ast.literal_eval(value); return (x1-1,y1-1,x2,y2)
        fit['bbox']=fit.bbox.map(box_to_zero); val['bbox']=val.bbox.map(box_to_zero)
    config={'task':task,'architecture':'tiny_cnn','image_size':64,'batch_size':32,
            'epochs':5,'learning_rate':.001,'seed':seed,'device':device,'synthetic':False,
            'pretrained_weights':None,'per_class_cap_before_split':cap,'workers':0,
            'input_root':str(root),'source_manifest':str(source),'manifest_sha256':sha256(source),
            'bbox_policy':'official 1-based -> 0-based half-open at load time' if kind=='web' else 'already cropped',
            'pilot_only':True,'purpose':'bounded feasibility pilot, not a final benchmark model'}
    save_json(result_dir/'config.json',config)
    save_json(result_dir/'run_metadata.json',metadata(config))
    save_json(result_dir/'class_to_idx.json',{c:i for i,c in enumerate(classes)})
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    if device=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=False
    ds=CompCarsMakeDataset(fit,label,64,normalise_image,root,classes)
    vd=CompCarsMakeDataset(val,label,64,normalise_image,root,classes)
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(ds,batch_size=32,shuffle=True,num_workers=0,generator=gen)
    vl=DataLoader(vd,batch_size=32,shuffle=False,num_workers=0)
    model=build_model('tiny_cnn',len(classes)).to(device)
    optimiser=torch.optim.AdamW(model.parameters(),lr=.001)
    criterion=nn.CrossEntropyLoss(); history=[]; best=-1.; start=time.perf_counter()
    for epoch in range(1,6):
        model.train(); total_loss=0.
        for x,y in loader:
            optimiser.zero_grad(); loss=criterion(model(x.to(device)),y.to(device))
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward(); optimiser.step(); total_loss+=float(loss.detach())*len(y)
        targets,pred,scores=predict_loader(model,vl,device)
        score=metrics(targets,pred,classes)
        row={'epoch':epoch,'train_loss':total_loss/len(fit),'validation_accuracy':score['accuracy'],
             'validation_macro_f1':score['macro_f1']}
        print(task,json.dumps(row),flush=True); history.append(row)
        if score['macro_f1']>best:
            best=score['macro_f1']
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'state_dict':state,'classes':classes,'config':config,'synthetic':False,
                        'manifest_sha256':sha256(source),'best_epoch':epoch},result_dir/'best.pt')
            save_json(result_dir/'validation_metrics.json',score)
    pd.DataFrame(history).to_csv(result_dir/'history.csv',index=False)
    # Reload the saved checkpoint and recompute validation metrics independently.
    saved=AttributePredictor(result_dir/'best.pt')
    y,pred,scores=predict_loader(saved.model,vl,'cpu')
    score=metrics(y,pred,classes)
    saved_score=json.loads((result_dir/'validation_metrics.json').read_text())
    if score['confusion_matrix']!=saved_score['confusion_matrix']:
        raise RuntimeError('Checkpoint reload changed validation predictions')
    pd.DataFrame({'relative_key':val.relative_key,'target':[classes[i] for i in y],
                  'prediction':[classes[i] for i in pred],'uncalibrated_score':scores}).to_csv(result_dir/'validation_predictions.csv',index=False)
    pd.DataFrame(score['confusion_matrix'],index=classes,columns=classes).to_csv(result_dir/'confusion_matrix.csv')
    majority=fit[label].value_counts().idxmax(); baseline=[classes.index(majority)]*len(y)
    save_json(result_dir/'majority_baseline_validation.json',metrics(y,baseline,classes))
    summary={'task':task,'synthetic':False,'training_completed':True,'epochs':5,
             'train_rows':len(fit),'validation_rows':len(val),'classes':len(classes),
             'best_epoch':int(torch.load(result_dir/'best.pt',weights_only=True)['best_epoch']),
             'validation_accuracy':score['accuracy'],'validation_macro_f1':score['macro_f1'],
             'official_test_model_evaluation_performed':False,'checkpoint_reload_verified':True,
             'elapsed_seconds':time.perf_counter()-start,'checkpoint_sha256':sha256(result_dir/'best.pt'),
             'model_bytes':(result_dir/'best.pt').stat().st_size,
             'model_parameters':sum(p.numel() for p in model.parameters()),
             'notice':'Preliminary validation on a capped image-level pilot. No final test, near-duplicate or identity-safe claim.'}
    save_json(result_dir/'pilot_summary.json',summary); return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archives-root',required=True); p.add_argument('--audit',required=True)
    p.add_argument('--output',required=True); p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    audit=Path(args.audit); checks=json.loads((audit/'audit_summary.json').read_text())
    for kind in ['web','surveillance']:
        if any(checks[kind][key] for key in ['split_mismatches','image_errors']):
            raise ValueError('Raw image and label audit must pass first')
        mismatches=checks[kind]['official_label_mismatches']
        if kind=='surveillance' and mismatches:
            raise ValueError('Surveillance labels must match official metadata')
        if kind=='web' and any(m['field']!='body_type' for m in mismatches):
            raise ValueError('Web make and bbox labels must match official metadata')
    save_json(out/'preflight_review.json',{'audit_body_type_mismatches':len(checks['web']['official_label_mismatches']),
        'policy':'Body task excludes official type 0, checks every remaining label against official metadata. Original manifests untouched.'})
    summaries=[]
    for task in ['colour','make','body_type']:
        summaries.append(run_task(task,Path(args.archives_root),audit,out,args.device))
        save_json(out/'pilot_summaries.json',summaries)
    print('All three real-data pilots completed; official test metrics NOT computed.',flush=True)

if __name__=='__main__': main()

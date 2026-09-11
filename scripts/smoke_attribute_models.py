"""Synthetic software test: never report these metrics as vehicle performance."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from vehicle_id.attributes.modelling import train, evaluate, AttributePredictor, save_json, metadata
from vehicle_id.pipeline import analyse_vehicle_image


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); args=p.parse_args()
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    root=out/'synthetic_fixture'; root.mkdir()
    (root/'data').mkdir()
    rows=[]; rng=np.random.default_rng(36127)
    for cls in range(2):
        for i in range(16):
            pixels=np.zeros((40,40,3),dtype=np.uint8)
            pixels[:,:,cls*2]=rng.integers(160,256,size=(40,40),dtype=np.uint8)
            name=f'data/toy_{cls}_{i}.png'; Image.fromarray(pixels).save(root/name)
            rows.append({'image_path':name,'make_name':f'toy_make_{cls}',
                         'car_type_name':f'toy_body_{cls}', 'color_name':['red','blue'][cls],
                         'bbox':'(2, 2, 38, 38)', 'split':'train' if i<12 else 'test'})
    df=pd.DataFrame(rows)
    df.to_csv(root/'web.csv',index=False)
    df.drop(columns='bbox').to_csv(root/'surveillance.csv',index=False)
    predictors={}; summaries={}
    for task in ['make','body_type','colour']:
        config={'task':task,'manifest':'surveillance.csv' if task=='colour' else 'web.csv',
                'architecture':'tiny_cnn','image_size':32,'batch_size':4,'epochs':2,
                'learning_rate':.005,'validation_fraction':.25,'seed':36127,'device':'cpu',
                'cpu_threads':2,'group_col':None,'synthetic':True}
        summaries[task]=train(config,root,out/task)
        evaluate(out/task/'best.pt',root,out/f'{task}_test',allow_synthetic=True)
        try:
            AttributePredictor(out/task/'best.pt')
        except ValueError:
            pass
        else:
            raise AssertionError('Synthetic weights must be rejected by default')
        predictors[task]=AttributePredictor(out/task/'best.pt',allow_synthetic=True)
    image=root/'synthetic_display.png'
    canvas=Image.new('RGB',(1600,240),'white')
    canvas.paste(Image.open(root/rows[0]['image_path']).resize((780,200)),(0,30))
    canvas.paste(Image.open(root/rows[16]['image_path']).resize((780,200)),(800,30))
    canvas.save(image)
    # These are test-supplied boxes, not detector predictions.
    result=analyse_vehicle_image(image,predictors,detections=[{'bbox':[0,30,780,230],'confidence':1.},
                                                             {'bbox':[800,30,1580,230],'confidence':1.}])
    result.pop('annotated_image').save(out/'synthetic_pipeline.png')
    assert len(result['vehicles'])==2
    assert all(result['vehicles'][0]['vehicle_profile']['status'][t]=='synthetic_smoke_only' for t in predictors)
    save_json(out/'pipeline_result.json',result)
    save_json(out/'smoke_summary.json',{'tasks':summaries,'synthetic':True,'all_three_checkpoints_reloaded':True,
               'synthetic_deployment_guard_passed':True,'actual_detector_executed':False,
               'notice':'Two toy classes per task, NOT the CompCars label spaces. This verifies software execution only.'})
    save_json(out/'run_metadata.json',metadata({'seed':36127,'device':'cpu','fixture':'synthetic coloured noise patches'}))
    print('Three synthetic training/evaluation/checkpoint/pipeline smoke tests completed.')


if __name__=='__main__': main()

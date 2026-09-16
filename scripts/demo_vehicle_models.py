"""Run trusted local attribute models and export a self-contained demo page."""
import argparse
import base64
import html
import io
import json
import os
import time
from pathlib import Path

from PIL import Image
import torch
from vehicle_id.attributes.modelling import AttributePredictor, sha256, save_json
from vehicle_id.pipeline import analyse_vehicle_image


def image_uri(image):
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=90)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def render_page(result, source_name, elapsed):
    cards = []
    for vehicle in result['vehicles']:
        profile = vehicle['vehicle_profile']
        attributes = []
        for task in ['make', 'body_type', 'colour', 'model']:
            status = profile['status'].get(task, 'not_assessed')
            status = {'pilot_prediction': 'Pilot estimate', 'predicted': 'Model estimate',
                      'not_assessed': 'Not assessed'}.get(status, status.replace('_', ' '))
            value = profile.get(task) or 'Not assessed'
            score = profile['confidence'].get(task)
            suffix = f' · score {score:.2f}' if score is not None else ''
            attributes.append(f'<div class="attribute"><span>{html.escape(task.replace("_", " ").title())}</span>'
                              f'<strong>{html.escape(value)}</strong><small>{html.escape(status + suffix)}</small></div>')
        cards.append(f'<article><h2>{html.escape(vehicle["vehicle_crop_id"].replace("_", " ").title())}</h2>'
                     f'<p>Detection score {vehicle["detection_confidence"]:.2f}</p>{"".join(attributes)}</article>')
    details = ''.join(cards) if cards else '<article><h2>No vehicles detected</h2><p>No attribute predictions were produced.</p></article>'
    picture = image_uri(result['annotated_image'])
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vehicle metadata demo</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#edf1f5;color:#172638}}
main{{max-width:1250px;margin:36px auto;padding:0 24px}}h1{{font-size:32px;margin:8px 0}}
.eyebrow{{color:#315d81;font-weight:600;letter-spacing:1px;text-transform:uppercase;font-size:12px}}
.note{{max-width:900px;line-height:1.5;color:#4b5f72}}.photo{{width:100%;max-height:620px;object-fit:contain;background:#142331;border-radius:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:16px;margin-top:24px}}
article{{background:white;border:1px solid #dbe3ec;border-radius:12px;padding:20px}}h2{{font-size:19px;margin:0}}article p{{font-size:12px;color:#5f7182}}
.attribute{{display:grid;gap:3px;padding:11px 0;border-top:1px solid #e5eaf0}}.attribute span,small{{font-size:12px;color:#5f7182}}strong{{font-size:18px}}
footer{{margin:28px 0;font-size:12px;color:#5f7182;line-height:1.5}}
</style></head><body><main><div class="eyebrow">Local research prototype · v0.1</div>
<h1>Vehicle metadata</h1><p class="note">{html.escape(source_name)} · {len(result['vehicles'])} vehicles · {elapsed:.2f} s end to end</p>
<img class="photo" src="{picture}" alt="Vehicle detections and predicted attributes">
<p class="note">Model predictions require review. Scores are uncalibrated model scores, not accuracy estimates.
Unavailable attributes are shown as Not assessed. Damage and accessories are not assessed by this version.</p>
<section class="grid">{details}</section><footer>CompCars-based attribute pilots with a pretrained YOLO detector.
Public benchmark validation does not establish performance on operational police imagery. This page works locally without an external service.</footer>
</main></body></html>'''


def run_demo(image_path, output, checkpoints, detector_weights=None, detections=None):
    if not checkpoints:
        raise ValueError('Provide at least one trained attribute checkpoint for a model demo')
    started = time.perf_counter()
    predictors = {}
    for task, checkpoint in checkpoints.items():
        predictor = AttributePredictor(checkpoint)
        if predictor.config['task'] != task:
            raise ValueError(f'{task} checkpoint has a different task')
        predictors[task] = predictor
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = analyse_vehicle_image(image_path, predictors, detections=detections, detector_weights=detector_weights)
    elapsed = time.perf_counter() - started
    result['annotated_image'].save(output / 'annotated.jpg', quality=95)
    payload = {key: result[key] for key in ['vehicles', 'confidence_semantics']}
    payload['elapsed_seconds'] = elapsed
    save_json(output / 'result.json', payload)
    save_json(output / 'provenance.json', {
        'image_sha256': sha256(image_path),
        'attribute_checkpoints': {task: {'sha256': sha256(path), 'config': predictors[task].config}
                                  for task, path in checkpoints.items()},
        'detector_sha256': sha256(detector_weights) if detector_weights else None,
        'detection_source': 'supplied_boxes' if detections is not None else 'local_YOLO',
        'timing_scope': 'checkpoint loading, image loading, detection, crops and attribute inference; export excluded'})
    (output / 'demo.html').write_text(render_page(result, Path(image_path).name, elapsed), encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--make', type=Path)
    parser.add_argument('--body-type', type=Path)
    parser.add_argument('--colour', type=Path)
    detector = parser.add_mutually_exclusive_group(required=True)
    detector.add_argument('--detector', type=Path)
    detector.add_argument('--boxes', type=Path, help='JSON list of supplied detections for a crop-level check')
    args = parser.parse_args()
    # Keep library settings beside the local outputs, outside the source tree.
    settings = args.output.resolve().parent / 'runtime_settings'
    settings.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(settings))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    torch.set_num_threads(2)
    checkpoints = {k: v for k, v in [('make', args.make), ('body_type', args.body_type), ('colour', args.colour)] if v}
    boxes = json.loads(args.boxes.read_text(encoding='utf-8')) if args.boxes else None
    result = run_demo(args.image, args.output, checkpoints, args.detector, boxes)
    print(json.dumps({'vehicles': len(result['vehicles']), 'page': str(args.output / 'demo.html')}))


if __name__ == '__main__':
    main()

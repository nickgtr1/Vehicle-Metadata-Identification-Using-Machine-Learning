"""Bounded pretrained ResNet-18 comparison on the frozen TinyCNN colour split.

No downloads or official-test evaluation. Only explicitly supplied local weights.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from vehicle_id.attributes.dataset import CompCarsMakeDataset, resolve_image_path
from vehicle_id.attributes.modelling import (
    normalise_image, predict_loader, metrics, save_json, metadata, sha256, AttributePredictor)
from audit_local_compcars import mapped_frame


def validate_frozen_split(frame, official):
    required = {'relative_key', 'sha256', 'color_name', 'image_path', 'experiment_split', 'split'}
    if not required <= set(frame):
        raise ValueError('Missing frozen split fields')
    if frame[list(required)].isna().any().any():
        raise ValueError('Null split fields')
    if frame.relative_key.duplicated().any() or frame.sha256.duplicated().any():
        raise ValueError('Repeated path/content in frozen split')
    if set(frame.experiment_split) != {'train', 'validation'} or set(frame.split) != {'train'}:
        raise ValueError('Only official TRAIN may populate this comparison')
    lookup = official.set_index('relative_key')
    if lookup.index.duplicated().any():
        raise ValueError('Official source has duplicate keys')
    for r in frame.to_dict('records'):
        if r['relative_key'] not in lookup.index:
            raise ValueError('Unknown image key')
        source = lookup.loc[r['relative_key']]
        if source['split'] != 'train' or source['color_name'] != r['color_name'] or source['image_path'] != r['image_path']:
            raise ValueError('Frozen split disagrees with official task manifest')
    fit = frame[frame.experiment_split == 'train'].copy()
    val = frame[frame.experiment_split == 'validation'].copy()
    if set(fit.color_name) != set(val.color_name):
        raise ValueError('Class coverage mismatch')
    return fit, val, sorted(fit.color_name.unique().tolist())


def configure_trainable(model, fine_tune):
    # Keep all BatchNorm running statistics fixed, including those in layer4.
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    model.fc.train()
    if fine_tune:
        for p in model.layer4.parameters():
            p.requires_grad = True
        model.layer4.train()
        for layer in model.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archives-root', type=Path, required=True)
    p.add_argument('--previous-pilot', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--provenance', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    args = p.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    source = repo / 'data/manifests/compcars_surveillance_classification.csv'
    old_config = json.loads((args.previous_pilot / 'config.json').read_text())
    if sha256(source) != old_config['manifest_sha256']:
        raise ValueError('Source manifest changed since TinyCNN pilot')
    audit = json.loads((args.audit / 'audit_summary.json').read_text())['surveillance']
    if any(audit[k] for k in ['split_mismatches', 'image_errors', 'official_label_mismatches']):
        raise ValueError('Surveillance audit must pass')
    official = mapped_frame(source, 'surveillance')
    split_path = args.previous_pilot / 'pilot_split.csv'
    frame = pd.read_csv(split_path)
    fit, val, classes = validate_frozen_split(frame, official)
    index = pd.read_csv(args.audit / 'surveillance_resolved_index.csv').set_index('relative_key')
    test_hashes = set(index.loc[official.loc[official.split == 'test', 'relative_key'], 'sha256'])
    if set(frame.sha256) & test_hashes:
        raise ValueError('Content overlaps official test')
    for row in frame.to_dict('records'):
        path = resolve_image_path(row['image_path'], args.archives_root)
        if sha256(path) != row['sha256'] or row['sha256'] != index.loc[row['relative_key'], 'sha256']:
            raise ValueError('Selected image differs from recorded audit hash')
    provenance = json.loads(args.provenance.read_text())
    weight_hash = sha256(args.weights)
    if weight_hash != provenance['sha256'] or not weight_hash.startswith('f37072fd'):
        raise ValueError('Pretrained weight provenance mismatch')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    seed = 36127
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark = False
    config = {'task': 'colour', 'architecture': 'resnet18', 'image_size': 224,
              'batch_size': 32, 'epochs': 8, 'head_only_epochs': 2,
              'head_learning_rate': .001, 'layer4_learning_rate': .0001,
              'weight_decay': .01, 'seed': seed, 'device': args.device, 'workers': 0,
              'synthetic': False, 'pilot_only': True, 'pretrained_weights': 'IMAGENET1K_V1',
              'pretrained_sha256': weight_hash, 'manifest_sha256': sha256(source),
              'split_sha256': sha256(split_path), 'batchnorm_statistics': 'frozen all epochs',
              'preprocessing': 'OpenCV whole-crop resize 224x224, RGB, ImageNet mean/std',
              'augmentation': 'none; no colour-altering augmentation',
              'preprocessing_deviation': 'Whole-crop square resize, not official 256-resize/224-center-crop',
              'selection': 'best validation macro F1; no threshold or test tuning',
              'comparison_limit': 'Same data, but architecture/pretraining/resolution/epochs all change; not a causal ablation',
              'official_test_evaluated': False, 'near_duplicate_or_identity_disjointness_verified': False}
    # Persist the complete recipe and exact source snapshots before optimisation.
    save_json(out / 'config.json', config)
    save_json(out / 'run_metadata.json', metadata(config))
    save_json(out / 'pretrained_provenance.json', provenance)
    save_json(out / 'class_to_idx.json', {name: i for i, name in enumerate(classes)})
    frame.to_csv(out / 'split_assignments.csv', index=False)
    frame.groupby(['experiment_split', 'color_name']).size().rename('count').reset_index().to_csv(out / 'class_counts.csv', index=False)
    source_files = [Path(__file__), repo / 'src/vehicle_id/attributes/modelling.py',
                    repo / 'src/vehicle_id/attributes/dataset.py', repo / 'scripts/audit_local_compcars.py']
    snapshots = out / 'executed_source'; snapshots.mkdir()
    records = []
    for f in source_files:
        dest = snapshots / f.name; dest.write_bytes(f.read_bytes())
        records.append({'source': str(f.resolve()), 'snapshot': dest.name, 'sha256': sha256(f)})
    save_json(out / 'source_hashes.json', records)
    train_ds = CompCarsMakeDataset(fit, 'color_name', 224, normalise_image, args.archives_root, classes)
    val_ds = CompCarsMakeDataset(val, 'color_name', 224, normalise_image, args.archives_root, classes)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    vl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(args.device)
    optimiser = torch.optim.AdamW([
        {'params': model.fc.parameters(), 'lr': .001},
        {'params': model.layer4.parameters(), 'lr': .0001}], weight_decay=.01)
    criterion = nn.CrossEntropyLoss(); history = []; best = -1.
    start = time.perf_counter()
    print('Starting fixed-recipe ResNet18: 2 head-only + 6 layer4/head epochs', flush=True)
    for epoch in range(1, 9):
        configure_trainable(model, epoch > 2)
        started_epoch = time.perf_counter(); total_loss = 0.
        for x, y in loader:
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(args.device)), y.to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward(); optimiser.step(); total_loss += float(loss.detach()) * len(y)
        y, pred, scores = predict_loader(model, vl, args.device)
        result = metrics(y, pred, classes)
        row = {'epoch': epoch, 'phase': 'head' if epoch <= 2 else 'layer4_and_head',
               'train_loss': total_loss / len(fit), 'validation_accuracy': result['accuracy'],
               'validation_macro_f1': result['macro_f1'], 'seconds': time.perf_counter() - started_epoch}
        history.append(row); pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        if result['macro_f1'] > best:
            best = result['macro_f1']
            torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        'classes': classes, 'config': config, 'synthetic': False,
                        'manifest_sha256': sha256(source), 'best_epoch': epoch}, out / 'best.pt')
            save_json(out / 'validation_metrics.json', result)
    training_seconds = time.perf_counter() - start
    # A fresh CPU reload, using the existing pipeline-compatible predictor.
    saved = AttributePredictor(out / 'best.pt')
    eval_start = time.perf_counter()
    y, pred, scores = predict_loader(saved.model, vl, 'cpu')
    eval_seconds = time.perf_counter() - eval_start
    result = metrics(y, pred, classes)
    if result['confusion_matrix'] != json.loads((out / 'validation_metrics.json').read_text())['confusion_matrix']:
        raise RuntimeError('CPU checkpoint reload changed predictions')
    pd.DataFrame({'relative_key': val.relative_key, 'target': [classes[i] for i in y],
                  'prediction': [classes[i] for i in pred], 'uncalibrated_score': scores}).to_csv(out / 'validation_predictions.csv', index=False)
    pd.DataFrame(result['confusion_matrix'], index=classes, columns=classes).to_csv(out / 'confusion_matrix.csv')
    pd.DataFrame(list(result['per_class_recall'].items()), columns=['colour', 'recall']).to_csv(out / 'per_class_recall.csv', index=False)
    # Independent reconstruction from persisted CSV, without using metrics().
    csv = pd.read_csv(out / 'validation_predictions.csv')
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for r in csv.itertuples():
        cm[classes.index(r.target), classes.index(r.prediction)] += 1
    denom = cm.sum(0) + cm.sum(1)
    macro = np.divide(2 * cm.diagonal(), denom, out=np.zeros(len(classes)), where=denom != 0).mean()
    assert cm.tolist() == result['confusion_matrix']
    assert abs(macro - result['macro_f1']) < 1e-12
    previous = json.loads((args.previous_pilot / 'validation_metrics.json').read_text())
    assert previous['rows'] == len(val) and previous['classes'] == classes
    comparison = [{'model': 'TinyCNN_random_64px_5epochs', 'accuracy': previous['accuracy'], 'macro_f1': previous['macro_f1']},
                  {'model': 'ResNet18_ImageNet_224px_8epochs', 'accuracy': result['accuracy'], 'macro_f1': result['macro_f1']}]
    pd.DataFrame(comparison).to_csv(out / 'baseline_comparison.csv', index=False)
    # Forward-only latency: one actual validation input, no repeated image reads.
    sample = val_ds[0][0].unsqueeze(0).to(args.device)
    model.load_state_dict(torch.load(out / 'best.pt', map_location='cpu', weights_only=True)['state_dict'])
    model.eval()
    with torch.inference_mode():
        for _ in range(10): model(sample)
        if args.device == 'cuda': torch.cuda.synchronize()
        timings = []
        for _ in range(50):
            t = time.perf_counter(); model(sample)
            if args.device == 'cuda': torch.cuda.synchronize()
            timings.append((time.perf_counter() - t) * 1000)
    pd.DataFrame({'forward_ms_batch1': timings}).to_csv(out / 'forward_latency.csv', index=False)
    summary = {'training_completed': True, 'train_rows': len(fit), 'validation_rows': len(val),
               'best_epoch': torch.load(out / 'best.pt', map_location='cpu', weights_only=True)['best_epoch'],
               'validation_accuracy': result['accuracy'], 'validation_macro_f1': result['macro_f1'],
               'accuracy_delta_percentage_points': (result['accuracy'] - previous['accuracy']) * 100,
               'macro_f1_delta': result['macro_f1'] - previous['macro_f1'],
               'checkpoint_reload_verified': True, 'csv_metrics_independently_verified': True,
               'selected_image_hashes_rechecked': len(frame), 'official_test_evaluated': False,
               'training_seconds': training_seconds, 'cpu_validation_seconds_including_image_loading': eval_seconds,
               'cpu_validation_images_per_second_including_loading': len(val) / eval_seconds,
               'forward_latency_median_ms': float(np.median(timings)), 'latency_device': args.device,
               'latency_scope': 'batch1, 224px, 10 warmups, 50 trials, forward only; excludes preprocessing/detection/UI',
               'model_bytes': (out / 'best.pt').stat().st_size, 'checkpoint_sha256': sha256(out / 'best.pt'),
               'scope': 'Capped same-split validation comparison; not full training, independent test, or police-domain validation.'}
    save_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

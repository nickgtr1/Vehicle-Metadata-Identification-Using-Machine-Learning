"""Audited make/body ResNet18 pilots using the shared attribute checkpoint format.

Prepare freezes the split. Train reads only that split. Verify reloads on CPU.
Raw data, pretrained weights and outputs are explicit local paths.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import ast
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from audit_local_compcars import mapped_frame
from train_colour_transfer import configure_trainable
from train_real_compcars_pilots import body_eligibility, pilot_split
from vehicle_id.attributes.dataset import CompCarsMakeDataset, resolve_image_path
from vehicle_id.attributes.modelling import (
    AttributePredictor, TASKS, metadata, metrics, normalise_image,
    predict_loader, save_json, sha256,
)

PRETRAINED_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'data/manifests/compcars_classification.csv'


def loader_view(frame):
    def convert(value):
        box = ast.literal_eval(value) if isinstance(value, str) else value
        if len(box) != 4 or not all(isinstance(x, int) for x in box):
            raise ValueError('Expected integer official xyxy bbox')
        x1, y1, x2, y2 = box
        if not (1 <= x1 < x2 and 1 <= y1 < y2):
            raise ValueError('Expected 1-based official bbox')
        return (x1 - 1, y1 - 1, x2, y2)
    result = frame.copy()
    # Persist original annotations; convert exactly once in the loader view.
    result['bbox'] = result.bbox.map(convert)
    return result


def validate_split(frame, official, label, test_hashes):
    fields = ['relative_key', 'image_path', 'sha256', 'bbox', label, 'split', 'experiment_split']
    if not set(fields) <= set(frame) or frame[fields].isna().any().any():
        raise ValueError('Incomplete frozen split')
    if frame.relative_key.duplicated().any() or frame.sha256.duplicated().any():
        raise ValueError('Duplicate path/content in frozen split')
    if set(frame.split) != {'train'} or set(frame.experiment_split) != {'train', 'validation'}:
        raise ValueError('Only official train may supply train/validation')
    if set(frame.sha256) & set(test_hashes):
        raise ValueError('Frozen split overlaps official test content')
    lookup = official.set_index('relative_key')
    if lookup.index.duplicated().any():
        raise ValueError('Repeated official key')
    for row in frame.to_dict('records'):
        if row['relative_key'] not in lookup.index:
            raise ValueError('Unknown image key')
        original = lookup.loc[row['relative_key']]
        if any(original[k] != row[k] for k in ['split', label, 'bbox', 'image_path', 'sha256']):
            raise ValueError('Frozen split differs from audited manifest')
    fit = frame[frame.experiment_split == 'train'].copy()
    val = frame[frame.experiment_split == 'validation'].copy()
    classes = sorted(fit[label].unique().tolist())
    if set(classes) != set(val[label]):
        raise ValueError('Train/validation class coverage differs')
    return fit, val, classes


def audited_frame(audit):
    summary = json.loads((audit / 'audit_summary.json').read_text(encoding='utf-8'))['web']
    if summary['manifest_sha256'] != sha256(SOURCE):
        raise ValueError('Manifest changed since audit')
    if any(summary[k] for k in ['split_mismatches', 'image_errors', 'official_label_mismatches']):
        raise ValueError('Web image, split and official label audit must pass')
    frame = mapped_frame(SOURCE, 'web')
    index = pd.read_csv(audit / 'web_resolved_index.csv')
    if index.relative_key.duplicated().any() or set(index.relative_key) != set(frame.relative_key):
        raise ValueError('Audit index coverage differs from manifest')
    index = index.set_index('relative_key')
    for column in ['image_path', 'split']:
        if not frame[column].eq(frame.relative_key.map(index[column])).all():
            raise ValueError('Audit index differs from source paths/splits')
    frame['sha256'] = frame.relative_key.map(index.sha256)
    if not frame.sha256.str.fullmatch(r'[0-9a-f]{64}').fillna(False).all():
        raise ValueError('Missing or malformed image hashes')
    return frame


def prepare(args):
    frame = audited_frame(args.audit)
    reserved = set(frame.loc[frame.split == 'test', 'sha256'])
    out = args.prepared
    out.mkdir(parents=True, exist_ok=False)
    unavailable = frame.iloc[:0].copy()
    if args.task == 'body_type':
        frame, unavailable = body_eligibility(frame, args.root)
    label = TASKS[args.task]
    if frame[label].isna().any():
        raise ValueError('Missing eligible target')
    fit, val, excluded, classes, info = pilot_split(frame, label, args.cap, 36127, reserved)
    split = pd.concat([fit, val], ignore_index=True)
    validate_split(split, frame, label, reserved)
    split.to_csv(out / 'split_assignments.csv', index=False)
    excluded.to_csv(out / 'excluded_training_rows.csv', index=False)
    unavailable.to_csv(out / 'unavailable_body_labels.csv', index=False)
    split.groupby(['experiment_split', label]).size().rename('count').reset_index().to_csv(out / 'class_counts.csv', index=False)
    info['missing_body_labels_by_split'] = unavailable.split.value_counts().to_dict()
    save_json(out / 'split_summary.json', info)
    config = {'task': args.task, 'architecture': 'resnet18', 'image_size': 224,
              'batch_size': 32, 'epochs': 8, 'head_only_epochs': 2, 'seed': 36127,
              'head_learning_rate': .001, 'layer4_learning_rate': .0001, 'weight_decay': .01,
              'workers': 0, 'per_class_cap_before_split': args.cap,
              'pretrained_sha256': PRETRAINED_SHA256, 'pretrained_weights': 'IMAGENET1K_V1',
              'synthetic': False, 'pilot_only': True, 'manifest_sha256': sha256(SOURCE),
              'split_sha256': sha256(out / 'split_assignments.csv'),
              'batchnorm_statistics': 'frozen all epochs', 'augmentation': 'none',
              'bbox_policy': 'official 1-based xyxy converted once to zero-based half-open',
              'preprocessing': 'OpenCV whole-crop square resize, RGB, ImageNet mean/std',
              'selection': 'highest validation macro F1; earliest epoch wins ties',
              'official_test_evaluated': False, 'near_duplicate_or_identity_disjointness_verified': False}
    save_json(out / 'config.json', config)
    save_json(out / 'classes.json', classes)
    save_json(out / 'run_metadata.json', metadata(config))
    print(json.dumps(info, indent=2), flush=True)


def load_prepared(prepared, root, audit):
    config = json.loads((prepared / 'config.json').read_text(encoding='utf-8'))
    if config['manifest_sha256'] != sha256(SOURCE) or config['split_sha256'] != sha256(prepared / 'split_assignments.csv'):
        raise ValueError('Prepared source or split has changed')
    official = audited_frame(audit)
    reserved = set(official.loc[official.split == 'test', 'sha256'])
    if config['task'] == 'body_type':
        official, _ = body_eligibility(official, root)
    frame = pd.read_csv(prepared / 'split_assignments.csv')
    fit, val, classes = validate_split(frame, official, TASKS[config['task']], reserved)
    if classes != json.loads((prepared / 'classes.json').read_text(encoding='utf-8')):
        raise ValueError('Prepared class mapping has changed')
    for row in frame.to_dict('records'):
        if sha256(resolve_image_path(row['image_path'], root)) != row['sha256']:
            raise ValueError('Selected image changed since audit')
    return config, fit, val, classes


def make_loader(frame, config, root, classes, shuffle=False):
    dataset = CompCarsMakeDataset(loader_view(frame), TASKS[config['task']],
                                 config['image_size'], normalise_image, root, classes)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle,
                      num_workers=0, generator=torch.Generator().manual_seed(config['seed']))


def verify(args):
    config, fit, val, classes = load_prepared(args.prepared, args.root, args.audit)
    payload = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
    if payload['classes'] != classes or any(payload['config'].get(k) != v for k, v in config.items()):
        raise ValueError('Checkpoint differs from prepared experiment')
    torch.set_num_threads(2)
    predictor = AttributePredictor(args.output / 'best.pt')
    y, pred, scores = predict_loader(predictor.model, make_loader(val, config, args.root, classes), 'cpu')
    result = metrics(y, pred, classes)
    expected = json.loads((args.output / 'validation_metrics.json').read_text(encoding='utf-8'))
    if result['confusion_matrix'] != expected['confusion_matrix']:
        raise ValueError('Fresh-process CPU reload changed validation predictions')
    predictions = pd.DataFrame({'relative_key': val.relative_key, 'target': [classes[i] for i in y],
                                'prediction': [classes[i] for i in pred], 'uncalibrated_score': scores})
    path = args.output / 'validation_predictions.csv'
    predictions.to_csv(path, index=False)
    saved = pd.read_csv(path)
    cm = pd.crosstab(saved.target, saved.prediction).reindex(index=classes, columns=classes, fill_value=0).to_numpy()
    denominator = cm.sum(0) + cm.sum(1)
    macro = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(len(classes)), where=denominator != 0).mean()
    if cm.tolist() != result['confusion_matrix'] or abs(float(macro) - result['macro_f1']) > 1e-12:
        raise ValueError('Persisted predictions do not reproduce metrics')
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(args.output / 'confusion_matrix.csv')
    pd.DataFrame({'class': classes, 'support': cm.sum(1), 'recall': list(result['per_class_recall'].values())}).to_csv(args.output / 'per_class_recall.csv', index=False)
    majority = fit[TASKS[config['task']]].value_counts().idxmax()
    save_json(args.output / 'majority_baseline_validation.json', metrics(y, [classes.index(majority)] * len(y), classes))
    save_json(args.output / 'verification.json', {
        'fresh_process_cpu_reload_passed': True, 'predictions_csv_metrics_verified': True,
        'selected_image_hashes_rechecked': len(fit) + len(val),
        'checkpoint_sha256': sha256(args.output / 'best.pt'), 'official_test_evaluated': False})


def train(args):
    config, fit, val, classes = load_prepared(args.prepared, args.root, args.audit)
    if sha256(args.weights) != PRETRAINED_SHA256:
        raise ValueError('Expected official ResNet18 IMAGENET1K_V1 weights')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config['device'] = args.device
    save_json(out / 'config.json', config)
    save_json(out / 'run_metadata.json', metadata(config))
    save_json(out / 'class_to_idx.json', {c: i for i, c in enumerate(classes)})
    snapshots = out / 'executed_source'
    snapshots.mkdir()
    files = [Path(__file__), REPO / 'scripts/train_colour_transfer.py',
             REPO / 'scripts/train_real_compcars_pilots.py', REPO / 'scripts/audit_local_compcars.py',
             REPO / 'src/vehicle_id/attributes/dataset.py', REPO / 'src/vehicle_id/attributes/modelling.py']
    for source in files:
        (snapshots / source.name).write_bytes(source.read_bytes())
    save_json(out / 'source_hashes.json', {p.name: sha256(p) for p in files})
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(args.device)
    optimiser = torch.optim.AdamW([
        {'params': model.fc.parameters(), 'lr': config['head_learning_rate']},
        {'params': model.layer4.parameters(), 'lr': config['layer4_learning_rate']}],
        weight_decay=config['weight_decay'])
    loader = make_loader(fit, config, args.root, classes, True)
    validation = make_loader(val, config, args.root, classes)
    best, history, started = -1., [], time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        configure_trainable(model, epoch > config['head_only_epochs'])
        loss_sum, epoch_start = 0., time.perf_counter()
        for x, y in loader:
            optimiser.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.to(args.device)), y.to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward(); optimiser.step()
            loss_sum += float(loss.detach()) * len(y)
        y, pred, _ = predict_loader(model, validation, args.device)
        result = metrics(y, pred, classes)
        row = {'epoch': epoch, 'train_loss': loss_sum / len(fit),
               'validation_accuracy': result['accuracy'], 'validation_macro_f1': result['macro_f1'],
               'seconds': time.perf_counter() - epoch_start}
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print(json.dumps(row), flush=True)
        if result['macro_f1'] > best:
            best = result['macro_f1']
            torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        'classes': classes, 'config': config, 'synthetic': False,
                        'manifest_sha256': config['manifest_sha256'], 'best_epoch': epoch}, out / 'best.pt')
            save_json(out / 'validation_metrics.json', result)
    save_json(out / 'training_summary.json', {
        'training_completed': True, 'training_seconds': time.perf_counter() - started,
        'peak_cuda_reserved_MiB': torch.cuda.max_memory_reserved() / 2**20 if args.device == 'cuda' else None,
        'official_test_evaluated': False, 'verification_pending': True})
    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'verify',
                    '--prepared', str(args.prepared), '--root', str(args.root),
                    '--audit', str(args.audit), '--output', str(out)], check=True)
    summary = json.loads((out / 'training_summary.json').read_text(encoding='utf-8'))
    summary['verification_pending'] = False
    save_json(out / 'training_summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'train', 'verify'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--task', choices=['make', 'body_type'])
    parser.add_argument('--cap', type=int, default=100)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.mode == 'prepare' and (not args.task or args.cap < 2):
        parser.error('prepare requires --task and --cap >= 2')
    if args.mode in ['train', 'verify'] and args.output is None:
        parser.error('train/verify requires --output')
    if args.mode == 'train' and args.weights is None:
        parser.error('train requires local --weights')
    {'prepare': prepare, 'train': train, 'verify': verify}[args.mode](args)


if __name__ == '__main__':
    main()

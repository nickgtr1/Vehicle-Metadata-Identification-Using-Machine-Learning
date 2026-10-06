"""Three supervised tasks; no downloads and no implicit test-set tuning."""
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from .dataset import CompCarsMakeDataset, resolve_image_path, parse_bbox

TASKS = {'make': 'make_name', 'body_type': 'car_type_name', 'colour': 'color_name'}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(config):
    return {'utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
            'config': config, 'seed': config.get('seed', 36127),
            'python': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'cuda_available': torch.cuda.is_available(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'packages': {p: importlib.metadata.version(p) for p in
                         ['numpy', 'pandas', 'torch', 'torchvision', 'scikit-learn', 'Pillow', 'opencv-python']}}


def audit_manifest(df, label_col, root):
    required = {'image_path', 'split', label_col}
    if not required <= set(df):
        raise ValueError(f'Missing columns: {required - set(df)}')
    paths = [resolve_image_path(p, root) for p in df.image_path]
    keys = [str(p).casefold() for p in paths]
    train_keys = {p for p, s in zip(keys, df.split) if s == 'train'}
    test_keys = {p for p, s in zip(keys, df.split) if s == 'test'}
    bad_boxes = 0
    if 'bbox' in df:
        for b in df.bbox:
            try:
                if parse_bbox(b) is None:
                    bad_boxes += 1
            except (ValueError, TypeError, SyntaxError):
                bad_boxes += 1
    train_labels = set(df.loc[df.split == 'train', label_col].dropna())
    test_labels = set(df.loc[df.split == 'test', label_col].dropna())
    return {'rows': len(df), 'classes': int(df[label_col].nunique()),
            'split_counts': {str(k): int(v) for k, v in df.split.value_counts().items()},
            'invalid_split_rows': int((~df.split.isin(['train', 'test'])).sum()),
            'missing_labels': int(df[label_col].isna().sum()), 'invalid_bbox_rows': bad_boxes,
            'duplicate_path_rows': len(keys) - len(set(keys)),
            'train_test_path_overlap': len(train_keys & test_keys),
            'test_labels_absent_from_train': sorted(test_labels - train_labels),
            'existing_image_paths': sum(p.is_file() for p in paths),
            'content_duplicates_checked': False, 'physical_vehicle_overlap_checked': False,
            'notice': 'Metadata/path audit only; no proof of image quality, official split provenance or identity-level leakage safety.'}


def split_training(df, label_col, fraction, seed, group_col=None):
    """Keep published test membership fixed; validation only from official train."""
    if not 0 < fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    train, test = df[df.split == 'train'].copy(), df[df.split == 'test'].copy()
    if train.empty or test.empty:
        raise ValueError('Both official train and test subsets are required')
    if group_col:
        if group_col not in df or df[group_col].isna().any():
            raise ValueError('Grouping requires complete approved identifiers')
        if set(train[group_col]) & set(test[group_col]):
            raise ValueError('A group crosses official train/test; resolve before training')
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(train, groups=train[group_col]))
        fit, val = train.iloc[a].copy(), train.iloc[b].copy()
    else:
        fit, val = train_test_split(train, test_size=fraction, random_state=seed, stratify=train[label_col])
        fit, val = fit.copy(), val.copy()
    if set(fit[label_col]) != set(train[label_col]) or set(val[label_col]) != set(train[label_col]):
        raise ValueError('Training/validation must cover all train classes; revise grouping or fraction')
    classes = sorted(fit[label_col].unique().tolist())
    if set(test[label_col]) - set(classes):
        raise ValueError('Official test contains unseen classes')
    fit['experiment_split'], val['experiment_split'], test['experiment_split'] = 'train', 'validation', 'test'
    return fit, val, test, classes


def build_model(architecture, classes):
    if architecture == 'tiny_cnn':
        return nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                             nn.Flatten(), nn.Linear(24, classes))
    if architecture == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    raise ValueError(f'Unsupported architecture: {architecture}')


def normalise_image(image):
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
    return (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def metrics(y, pred, classes):
    labels = list(range(len(classes)))
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
            'per_class_recall': dict(zip(classes, recall_score(y, pred, labels=labels, average=None, zero_division=0).tolist())),
            'confusion_matrix': confusion_matrix(y, pred, labels=labels).tolist(), 'classes': classes, 'rows': len(y)}


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    targets, predictions, scores = [], [], []
    for x, y in loader:
        probs = model(x.to(device)).softmax(1).cpu()
        targets.extend(y.tolist()); predictions.extend(probs.argmax(1).tolist())
        scores.extend(probs.max(1).values.tolist())
    return targets, predictions, scores


def require_audit_ok(audit):
    if any(audit[k] for k in ['invalid_split_rows', 'missing_labels', 'invalid_bbox_rows',
                              'duplicate_path_rows', 'train_test_path_overlap', 'test_labels_absent_from_train']):
        raise ValueError(f'Manifest validation failed: {audit}')
    if audit['existing_image_paths'] != audit['rows']:
        raise FileNotFoundError(f"Only {audit['existing_image_paths']}/{audit['rows']} images available; no training started")


def train(config, root, output):
    output, root = Path(output), Path(root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'run_metadata.json', metadata(config))
    seed = int(config['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    torch.use_deterministic_algorithms(True)
    device = config.get('device', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be explicitly cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    manifest = root / config['manifest']
    df = pd.read_csv(manifest)
    label = TASKS[config['task']]
    audit = audit_manifest(df, label, root)
    save_json(output / 'data_audit.json', audit)
    require_audit_ok(audit)
    fit, val, test, classes = split_training(df, label, config['validation_fraction'], seed, config.get('group_col'))
    pd.concat([fit, val, test]).to_csv(output / 'split_assignments.csv', index=False)
    save_json(output / 'class_to_idx.json', {c: i for i, c in enumerate(classes)})
    gen = torch.Generator().manual_seed(seed)
    def loader(frame, shuffle=False):
        ds = CompCarsMakeDataset(frame, label, config['image_size'], normalise_image, root, classes)
        return DataLoader(ds, batch_size=config['batch_size'], shuffle=shuffle, num_workers=0, generator=gen)
    fit_loader, val_loader = loader(fit, True), loader(val)
    model = build_model(config['architecture'], len(classes)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    history, best = [], -1.
    if int(config['epochs']) < 1:
        raise ValueError('epochs must be positive')
    started = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        model.train(); loss_sum = 0.
        for x, y in fit_loader:
            optimiser.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward(); optimiser.step()
            loss_sum += float(loss.detach()) * len(y)
        y, pred, _ = predict_loader(model, val_loader, device)
        score = metrics(y, pred, classes)
        row = {'epoch': epoch, 'train_loss': loss_sum / len(fit), 'validation_accuracy': score['accuracy'], 'validation_macro_f1': score['macro_f1']}
        history.append(row); print(json.dumps(row), flush=True)
        if score['macro_f1'] > best:
            best = score['macro_f1']
            torch.save({'state_dict': model.cpu().state_dict(), 'classes': classes, 'config': config,
                        'manifest_sha256': sha256(manifest), 'best_epoch': epoch,
                        'synthetic': bool(config.get('synthetic', False))}, output / 'best.pt')
            model.to(device)
            save_json(output / 'validation_metrics.json', score)
    pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
    summary = {'status': 'completed', 'task': config['task'], 'synthetic': bool(config.get('synthetic', False)),
               'train_rows': len(fit), 'validation_rows': len(val), 'test_rows_reserved': len(test),
               'test_images_loaded_during_training': 0, 'elapsed_seconds': time.perf_counter() - started,
               'model_bytes': (output / 'best.pt').stat().st_size, 'checkpoint_sha256': sha256(output / 'best.pt'),
               'manifest_sha256': sha256(manifest), 'pretrained_weights_used': False,
               'group_column': config.get('group_col'),
               'limitation': 'Image-level validation is not identity-disjoint unless approved groups are supplied. Synthetic runs do not demonstrate vehicle recognition.'}
    save_json(output / 'training_summary.json', summary)
    return summary


class AttributePredictor:
    """Only load trusted local checkpoints; softmax scores are not calibrated."""
    def __init__(self, checkpoint, allow_synthetic=False):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['synthetic'] and not allow_synthetic:
            raise ValueError('Synthetic smoke-test weights are not deployable vehicle models')
        self.config, self.classes = payload['config'], payload['classes']
        self.synthetic = payload['synthetic']
        self.model = build_model(self.config['architecture'], len(self.classes))
        self.model.load_state_dict(payload['state_dict']); self.model.eval()

    @torch.inference_mode()
    def predict(self, crop):
        image = cv2.resize(np.asarray(crop.convert('RGB')), (self.config['image_size'], self.config['image_size']))
        p = self.model(normalise_image(image).unsqueeze(0)).softmax(1)[0]
        i = int(p.argmax())
        return {'label': self.classes[i], 'score': float(p[i]),
                'status': ('synthetic_smoke_only' if self.synthetic else
                           'pilot_prediction' if self.config.get('pilot_only') else 'predicted'),
                'score_semantics': 'uncalibrated softmax score'}


def evaluate(checkpoint, root, output, allow_synthetic=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    predictor = AttributePredictor(checkpoint, allow_synthetic)
    config, classes = predictor.config, predictor.classes
    manifest = Path(root) / config['manifest']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if sha256(manifest) != payload['manifest_sha256']:
        raise ValueError('Manifest changed since training')
    df = pd.read_csv(manifest)
    require_audit_ok(audit_manifest(df, TASKS[config['task']], root))
    test = df[df.split == 'test'].copy()
    ds = CompCarsMakeDataset(test, TASKS[config['task']], config['image_size'], normalise_image, root, classes)
    started = time.perf_counter()
    y, pred, scores = predict_loader(predictor.model, DataLoader(ds, batch_size=config['batch_size'], num_workers=0), 'cpu')
    elapsed = time.perf_counter() - started
    result = metrics(y, pred, classes)
    result.update({'synthetic': predictor.synthetic, 'checkpoint_sha256': sha256(checkpoint),
                   'elapsed_seconds_including_loading': elapsed, 'images_per_second_including_loading': len(y)/elapsed,
                   'device': 'cpu', 'notice': 'Synthetic metrics are software checks only.' if predictor.synthetic else 'Public benchmark; not NSW Police performance.'})
    save_json(output / 'metrics.json', result); save_json(output / 'run_metadata.json', metadata(config))
    pd.DataFrame({'image_path': test.image_path, 'target': [classes[i] for i in y],
                  'prediction': [classes[i] for i in pred], 'uncalibrated_score': scores}).to_csv(output / 'predictions.csv', index=False)
    pd.DataFrame(result['confusion_matrix'], index=classes, columns=classes).to_csv(output / 'confusion_matrix.csv')
    return result

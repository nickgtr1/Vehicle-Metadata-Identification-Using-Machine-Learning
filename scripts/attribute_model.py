"""Run from repo root after pip install -e . --no-deps --no-build-isolation."""
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from vehicle_id.attributes.modelling import TASKS, audit_manifest, metadata, save_json, sha256, train, evaluate, split_training


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['audit', 'train', 'evaluate'])
    p.add_argument('--config', required=True)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint')
    args = p.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    output = Path(args.output)
    if args.mode == 'audit':
        output.mkdir(parents=True, exist_ok=False)
        path = Path(args.root) / config['manifest']
        df = pd.read_csv(path)
        label = TASKS[config['task']]
        result = audit_manifest(df, label, args.root)
        result['manifest_sha256'] = sha256(path)
        try:
            fit, val, test, classes = split_training(df, label, config['validation_fraction'], config['seed'], config.get('group_col'))
            result['proposed_split_counts'] = {'train': len(fit), 'validation': len(val), 'test_unchanged': len(test)}
            pd.concat([fit, val, test])[['image_path', label, 'split', 'experiment_split']].to_csv(output / 'split_design.csv', index=False)
        except ValueError as exc:
            result['split_design_error'] = str(exc)
        save_json(output / 'audit.json', result)
        save_json(output / 'run_metadata.json', metadata(config))
        df.groupby(['split', label]).size().rename('count').reset_index().to_csv(output / 'class_counts.csv', index=False)
    elif args.mode == 'train':
        result = train(config, args.root, output)
    else:
        if not args.checkpoint:
            p.error('--checkpoint is required for evaluate')
        result = evaluate(args.checkpoint, args.root, output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Build validation figures and a model card from verified make/body runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vehicle_id.attributes.modelling import save_json, sha256


def checked_run(path):
    proof = json.loads((path / 'verification.json').read_text(encoding='utf-8'))
    if not proof['fresh_process_cpu_reload_passed'] or not proof['predictions_csv_metrics_verified']:
        raise ValueError('Run verification is incomplete')
    if proof['checkpoint_sha256'] != sha256(path / 'best.pt'):
        raise ValueError('Checkpoint changed after verification')
    score = json.loads((path / 'validation_metrics.json').read_text(encoding='utf-8'))
    baseline = json.loads((path / 'majority_baseline_validation.json').read_text(encoding='utf-8'))
    if score['classes'] != baseline['classes'] or score['rows'] != baseline['rows']:
        raise ValueError('Reference and model were not evaluated on the same labels/rows')
    return score, baseline, pd.read_csv(path / 'history.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--make', type=Path, required=True)
    parser.add_argument('--body-type', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = {'make': args.make, 'body_type': args.body_type}
    data = {task: checked_run(path) for task, path in runs.items()}
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    rows, confusion_rows = [], []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'figure.facecolor': 'white'}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for ax, (task, (score, baseline, history)) in zip(axes, data.items()):
            label = task.replace('_', ' ').title()
            ax.plot(history.epoch, history.validation_accuracy, marker='o', color='#185e91', label='Accuracy')
            ax.plot(history.epoch, history.validation_macro_f1, marker='s', linestyle='--', color='#a64a00', label='Macro F1')
            ax.set(title=f'{label}: validation during training', xlabel='Epoch', ylabel='Score', ylim=(0, 1), xticks=history.epoch)
            ax.legend(loc='lower right'); ax.grid(axis='y', alpha=.2)
            cm = np.asarray(score['confusion_matrix'])
            support = cm.sum(1)
            recall = np.divide(cm.diagonal(), support, out=np.zeros(len(support)), where=support != 0)
            per_class = pd.DataFrame({'class': score['classes'], 'support': support, 'recall': recall})
            per_class.to_csv(out / f'{task}_per_class.csv', index=False)
            for i, target in enumerate(score['classes']):
                for j, prediction in enumerate(score['classes']):
                    if i != j and cm[i, j]:
                        confusion_rows.append({'task': task, 'target': target, 'prediction': prediction, 'count': int(cm[i, j])})
            for model_name, values in [('majority', baseline), ('ResNet18', score)]:
                rows.append({'task': task, 'model': model_name, 'rows': values['rows'],
                             'classes': len(values['classes']), 'accuracy': values['accuracy'], 'macro_f1': values['macro_f1']})
            height = max(4, len(support) * .25)
            recall_fig, recall_ax = plt.subplots(figsize=(9, height), layout='constrained')
            order = per_class.sort_values(['recall', 'class'])
            recall_ax.barh(np.arange(len(order)), order.recall, color='#185e91')
            recall_ax.set(yticks=np.arange(len(order)), yticklabels=[f'{r["class"]} (n={r["support"]})' for r in order.to_dict('records')],
                          xlim=(0, 1), xlabel='Validation recall', title=f'{label}: every class, with validation support')
            recall_ax.tick_params(axis='y', labelsize=8)
            recall_ax.grid(axis='x', alpha=.2)
            recall_fig.savefig(out / f'{task}_recall.png', dpi=160)
            recall_fig.savefig(out / f'{task}_recall.svg')
            plt.close(recall_fig)
            if task == 'body_type':
                proportions = cm / support[:, None]
                matrix_fig, matrix_ax = plt.subplots(figsize=(11, 10), layout='constrained')
                heatmap = matrix_ax.imshow(proportions, vmin=0, vmax=1, cmap='Blues', interpolation='nearest')
                names = score['classes']
                matrix_ax.set(xticks=range(len(names)), xticklabels=names, yticks=range(len(names)), yticklabels=names,
                              xlabel='Predicted label', ylabel='True label', title='Body type: row-normalised validation confusion matrix')
                matrix_ax.tick_params(axis='x', rotation=55)
                for i in range(len(names)):
                    for j in range(len(names)):
                        if cm[i, j]:
                            rgb = np.asarray(heatmap.cmap(proportions[i, j])[:3])
                            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                            luminance = float(linear @ np.asarray([.2126, .7152, .0722]))
                            matrix_ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                                           fontsize=8, color='black' if luminance > .179 else 'white')
                matrix_fig.colorbar(heatmap, ax=matrix_ax, label='Fraction of true class; cells show image counts', shrink=.75)
                matrix_fig.savefig(out / 'body_type_confusion.png', dpi=160)
                matrix_fig.savefig(out / 'body_type_confusion.svg')
                plt.close(matrix_fig)
        fig.savefig(out / 'validation_history.png', dpi=160)
        fig.savefig(out / 'validation_history.svg')
        plt.close(fig)
        table = pd.DataFrame(rows)
        compare_fig, compare_axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for ax, measure in zip(compare_axes, ['accuracy', 'macro_f1']):
            for i, (name, color, hatch) in enumerate([('majority', '#727f89', '//'), ('ResNet18', '#185e91', '')]):
                chosen = table[table.model == name].set_index('task').loc[list(runs)]
                positions = np.arange(2) + (i - .5) * .35
                bars = ax.bar(positions, chosen[measure], .35, label=name, color=color, hatch=hatch)
                ax.bar_label(bars, labels=['<0.001' if 0 < v < .001 else f'{v:.3f}' for v in chosen[measure]], padding=3, fontsize=9)
            ax.set(xticks=[0, 1], xticklabels=['Make', 'Body type'], ylim=(0, 1), ylabel=measure.replace('_', ' ').title())
            ax.legend(loc='upper left'); ax.grid(axis='y', alpha=.2)
        compare_fig.savefig(out / 'baseline_comparison.png', dpi=160)
        compare_fig.savefig(out / 'baseline_comparison.svg')
        plt.close(compare_fig)
    table.to_csv(out / 'baseline_comparison.csv', index=False)
    confusions = pd.DataFrame(confusion_rows, columns=['task', 'target', 'prediction', 'count'])
    confusions = confusions.sort_values(['task', 'count', 'target', 'prediction'], ascending=[True, False, True, True])
    confusions.to_csv(out / 'all_confusions.csv', index=False)
    report = ['# Vehicle metadata v0.1 — make and body type', '',
              'Local CompCars validation pilots built on Yuchen\'s shared model framework and Nicholas\'s manifests.', '',
              '## Validation results', '', '| Task | Images | Classes | ResNet18 accuracy | Macro F1 | Majority accuracy |',
              '|---|---:|---:|---:|---:|---:|']
    for task, (score, baseline, history) in data.items():
        report.append(f'| {task} | {score["rows"]} | {len(score["classes"])} | {score["accuracy"]:.2%} | {score["macro_f1"]:.4f} | {baseline["accuracy"]:.2%} |')
    report += ['', '![ResNet18 and majority baseline on the same task validation splits](baseline_comparison.png)', '',
               'Both checkpoints passed a fresh-process CPU reload and independent metric reconstruction from saved predictions. '
               'These are capped, image-level validation results from one fixed seed, not official-test results or estimates for operational police imagery.', '',
               '## Recipe and coverage', '',
               'ResNet18 ImageNet initialization; RGB 224×224 square crops and ImageNet normalization; batch 32; '
               'two head-only epochs then six layer4/head epochs; frozen BatchNorm statistics; seed 36127. '
               'Validation macro F1 selects the checkpoint, with earliest-epoch tie breaking. No augmentation. '
               'Official 1-based boxes are converted once for loading. The make cap is 100 and body-type cap is 500 images per class before the 80/20 split.', '',
               '![Validation accuracy and macro F1 for every epoch](validation_history.png)', '',
               '## Error analysis', '',
               'Full class support/recall and every nonzero off-diagonal confusion are retained in the companion CSV files. '
               'The following are the largest observed confusion counts, selected by count rather than example appearance:', '',
               '| Task | True label | Predicted label | Images |', '|---|---|---|---:|']
    for task in runs:
        for row in confusions[confusions.task == task].head(5).to_dict('records'):
            report.append(f'| {task} | {row["target"]} | {row["prediction"]} | {row["count"]} |')
    report += ['', '![Body-type confusion: fractions by true class, annotated with image counts](body_type_confusion.png)', '',
               '## Limits and contribution', '',
               '- Exact-byte duplicates crossing official train/test are excluded from training. One representative per remaining hash is used. Near-duplicate and physical-vehicle independence are not established.',
               '- Official type 0 is unavailable. It is excluded only from the body task; valid make targets are retained.',
               '- Scores are uncalibrated softmax scores. Weak, small or occluded crops can produce incorrect high scores. No abstention threshold has been validated.',
               '- Class-name spelling follows the source manifest, including names such as Chrey and Wealeak. These labels have not been standardised for a user-facing taxonomy.',
               '- Vehicle-model recognition, damage and accessories remain not assessed. Colour requires Yuchen\'s trained local checkpoint; synthetic software checks do not reproduce his colour experiment.',
               '- Yuchen supplied the shared dataset/predictor/pipeline, pilot split and colour transfer recipe. This work adds the make/body transfer runner, verified outputs, bbox/interface checks and local demo/export workflow.',
               '- CompCars web data came from a separately recorded mirror. All extracted files were CRC-checked against that archive; 100 images matched decoded official bytes. That sample is not full publisher authentication.',
               '', 'AI assistance was used for implementation, verification scripts and drafting. These artifacts remain subject to Yuxiang\'s review before any coursework submission.', '']
    (out / 'MODEL_CARD.md').write_text('\n'.join(report), encoding='utf-8')
    save_json(out / 'figure_provenance.json', {
        'script_sha256': sha256(Path(__file__)),
        'runs': {task: {'checkpoint_sha256': sha256(path / 'best.pt'),
                        'metrics_sha256': sha256(path / 'validation_metrics.json')} for task, path in runs.items()},
        'scope': 'Coursework/team review; no journal compliance claim',
        'transformations': ['confusion matrix divided by row support for heatmap color', 'all-class recall sorted ascending', 'largest confusion counts shown; all retained in CSV'],
        'uncertainty': 'single-seed image-level validation; no independent-vehicle confidence intervals claimed'})


if __name__ == '__main__':
    main()

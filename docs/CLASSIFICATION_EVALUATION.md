# Unified attribute classification evaluation

One evaluation entry point for the three supervised attribute tasks, so that
colour, make and body-type numbers are produced under a single metric
definition instead of three local variants.

Added by Yuchen MENG for the attribute workstream. The make and body-type
figures below are computed from the confusion matrices published in PR #5 by
Yuxiang Wang; the colour figure is recomputed from the saved colour pilot
predictions retained locally (not published). Public colour aggregate evidence is
in `docs/evidence/colour_transfer_2026-09-11/`.

## Why a single entry point

Two runs can both report "macro F1" and still be incomparable when they differ
in class list, label encoding, zero-division handling, or whether classes with
no support are included in the average. This tool fixes those choices in one
place, writes the choice into every output file, and refuses input it cannot
evaluate under those rules.

## Metric specification

| Decision | Value |
|---|---|
| Class list | Supplied by the caller, saved in declared order, never inferred from the data |
| Raw vs display labels | Raw class IDs and labels stay authoritative; display names may be trimmed for presentation and never replace them |
| Label normalisation | `none` by default. `strip` is opt-in, is applied identically to the class list, targets and predictions, and aborts on any many-to-one collapse |
| Confusion counts | Integers only in effect; integral floats such as `1.0` are accepted; fractional, negative, non-finite, boolean, half-boolean, textual and out-of-range values are refused |
| Class indices | JSON integers only; floats (even `0.0`), booleans, `null` and strings are refused |
| Label encoding | Integer indices into the declared class list |
| Precision / recall / F1 | `zero_division=0`; an undefined denominator yields `0.0` rather than an error |
| Macro F1 (primary) | Mean over **all** declared classes, including classes with zero support |
| Macro F1 (secondary) | `macro_f1_excluding_zero_support`, reported separately and never substituted silently |
| Weighted F1 | Support-weighted over the declared classes |
| Uncalibrated scores | Recorded and hashed, never used by any metric |
| Unusable rows | Abort by default; with `--allow-unknown-labels` they are excluded **and** written to `excluded_rows.csv` |

`accuracy`, `macro_f1`, `macro_precision`, `macro_recall` and `weighted_f1` come
from scikit-learn calls that are identical to
`vehicle_id.attributes.modelling.metrics`, the implementation that produced the
saved colour pilot, so a recomputation reproduces the historical numbers exactly
rather than approximately.

## Running it

### Row-level predictions

```powershell
python scripts/evaluate_classification.py `
  --task colour --split validation `
  --predictions outputs/2026-09-11_colour_transfer/validation_predictions.csv `
  --class-mapping outputs/2026-09-11_colour_transfer/class_to_idx.json `
  --target-column target --prediction-column prediction --id-column relative_key `
  --label-normalisation none --expected-rows 386 `
  --seed not_applicable --device unused `
  --output outputs/<new-folder>/colour_validation
```

### Aggregate confusion matrix

First retrieve the pinned PR #5 inputs using the links and SHA-256 checks in
[inputs/README.md](evidence/classification_eval_2026-09-16/inputs/README.md).
The original files are deliberately not duplicated in this PR. The command
below assumes you saved the two make files under the indicated local names.

Use this only when raw per-sample predictions are unavailable. The numbers are
then derived from somebody else's aggregate and cannot be re-audited row by row,
which the output records as `input_mode: aggregate_confusion_matrix`.

```powershell
python scripts/evaluate_classification.py `
  --task make --split validation `
  --confusion-matrix docs/evidence/classification_eval_2026-09-16/inputs/pr5_make_validation_metrics.json `
  --class-mapping docs/evidence/classification_eval_2026-09-16/inputs/pr5_make_class_to_idx.json `
  --label-normalisation none --expected-rows 1120 `
  --output outputs/<new-folder>/make_validation
```

### Exit codes

| Code | Meaning |
|---:|---|
| 0 | completed, metrics written |
| 2 | refused input or output; existing non-empty directories remain unchanged; an owned fresh directory may contain a failure log |

## Input contract

| Column | Meaning |
|---|---|
| sample identifier (name is configurable) | unique per row; duplicates abort the run |
| target | the true label, spelled exactly as in the class list |
| prediction | the predicted label, spelled exactly as in the class list |
| score (optional) | uncalibrated score; recorded, never scored |

Plus a class list as a JSON array or as `class_to_idx.json`, and a note saying
which split the rows belong to.

## Outputs

| File | Contents |
|---|---|
| `metrics.json` | full result: policy, inputs and hashes, environment, per-class scores, confusion matrix, both checks |
| `per_class_metrics.csv` | precision, recall, F1, support and predicted count per class |
| `label_mapping.csv` | `class_id`, `raw_label`, `display_label`, `changed` |
| `confusion_matrix.csv` | rows are targets, columns are predictions, same class order |
| `excluded_rows.csv` | present only when rows could not be scored; lists every one with a reason |
| `environment.json` | interpreter, platform and package versions |
| `run_log.txt` | the exact command and each validation step |

## Two checks, and what each one means

`arithmetic_check` runs in both modes. It recomputes accuracy, macro
precision/recall/F1, weighted F1 and every per-class value with exact rational
arithmetic (`fractions.Fraction`) using no shared code with the numpy path, and
compares the results.

`matrix_cross_check` runs only in row-level mode, where two representations of
the same rows exist: sklearn scores from the labels versus the closed-form
reconstruction from the aggregated matrix. In aggregate mode there is no second
representation, so it is written as `not_performed` with a reason rather than
being reported as a check that did not happen.

## Safety behaviour

* No path is chosen automatically; `--predictions` or `--confusion-matrix` is
  always explicit.
* Nothing is downloaded and no network access is used.
* A split name aborts unless `--allow-official-test` is passed deliberately when,
  after lower-casing and removing separators, it is one of: `test`, `tests`,
  `testing`, `officialtest`, `testset`, `testdata`, `finaltest`, `holdout`,
  `holdouts`, `holdoutset`. This is an explicit list rather than a substring
  search, so `validation` and `latest` are not guarded.
* Reserved tokens (`unknown`, `not_assessed`, `unassessed`, `synthetic`,
  `synthetic_smoke_only`) are treated as unusable rows, never as classes.
* The output directory must be fresh, and the tool never writes into a
  directory it does not own: a refused run leaves the target byte-for-byte
  unchanged, including any existing `run_log.txt`. Failure details go to
  `stderr` as well as to the structured message on `stdout`.

## Results

Annotated-crop validation results. None of these is an official-test score.

| Task | Split | Classes | Rows | Accuracy | Macro F1 | Weighted F1 | Input mode |
|---|---|---:|---:|---:|---:|---:|---|
| colour | validation | 10 | 386 | 0.725389 | 0.729908 | 0.726483 | row-level predictions |
| make | validation | 75 | 1,120 | 0.546429 | 0.589046 | 0.553911 | aggregate matrix |
| body type | validation | 12 | 782 | 0.670077 | 0.696090 | 0.673483 | aggregate matrix |

The three tasks are reported separately. A combined "overall F1" is not
produced: the class counts and sample counts differ, the tasks answer different
questions, and no agreed weighting between them exists, so a single averaged
figure would not correspond to anything the project is asking.

Source: `docs/evidence/classification_eval_2026-09-16/`.

## Limitations

1. These are annotated-crop validation numbers, not detector, end-to-end or
   official-test results.
2. Make has 19 classes with fewer than ten validation images and two classes
   that are never predicted (`SAAB`, `Wealeak`). Macro F1 weights every class
   equally, so those estimates carry the same weight as the large classes while
   being far less stable.
3. The aggregate input mode cannot detect duplicate identifiers or per-row
   label errors; it trusts the supplied matrix. A per-sample audit for make and
   body type is awaiting the underlying prediction files.
4. The make and body-type validation splits were created for those runs and have
   not been verified for near-duplicate or same-physical-vehicle disjointness,
   so identical metric definitions do not make the three splits equally
   trustworthy.

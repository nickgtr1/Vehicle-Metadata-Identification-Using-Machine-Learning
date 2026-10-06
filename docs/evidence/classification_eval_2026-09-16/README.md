# Unified classification evaluation — aggregate evidence

16 September 2026 · annotated-crop validation results for three attribute tasks

Everything here is produced by `scripts/evaluate_classification.py` with the
metric definition described in [CLASSIFICATION_EVALUATION.md](../../CLASSIFICATION_EVALUATION.md).
No model was trained for this evidence. No official test set was evaluated.

## Results

| Task | Split | Classes | Rows | Accuracy | Macro F1 | Weighted F1 | Input mode |
|---|---|---:|---:|---:|---:|---:|---|
| colour | validation | 10 | 386 | 0.7253886010362695 | 0.7299082471403191 | 0.7264826819937054 | row-level predictions |
| make | validation | 75 | 1,120 | 0.5464285714285714 | 0.5890462077274896 | 0.5539105856593011 | aggregate matrix |
| body type | validation | 12 | 782 | 0.6700767263427110 | 0.6960901786007714 | 0.6734825891010982 | aggregate matrix |

The three tasks are reported separately, and no combined total is produced. The
class counts and sample counts differ, the tasks answer different questions, and
no agreed weighting between them exists.

## Where each number comes from

**colour** is recomputed from the saved per-sample validation predictions of the
ResNet-18 colour pilot. The prediction file is not duplicated here; its SHA-256
is recorded in `colour_validation/metrics.json` and it is retained locally,
not in the public repository. Public historical aggregate evidence is in
`docs/evidence/colour_transfer_2026-09-11/`. The recomputation reproduces the
stored pilot result exactly, which is the check that the metric definition did
not drift.

**make** and **body type** are computed from the confusion matrices published in
PR #5. They are recorded as `aggregate_confusion_matrix` because no per-sample
prediction file is available, so the numbers cannot be re-audited row by row.
The four source files are not duplicated; pinned source links and hashes are in
[inputs/README.md](inputs/README.md).

## Reading the evidence

| Path | Contents |
|---|---|
| `<run>/metrics.json` | full result: policy, inputs and hashes, environment, per-class scores, confusion matrix, both checks |
| `<run>/per_class_metrics.csv` | precision, recall, F1, support and predicted count per class |
| `<run>/confusion_matrix.csv` | rows are targets, columns are predictions, same class order |
| `<run>/label_mapping.csv` | `class_id`, `raw_label`, `display_label`, `changed` |
| `<run>/environment.json` | interpreter, platform and package versions |
| `unified_metrics_comparison.md` | the three tasks side by side, with the statement that no total is produced |
| `display_trim_control.csv` | control run showing that a collision-free rename cannot change a metric |
| `verification.json` | what was checked, with counts, and what was not checked |
| `inputs/README.md` | pinned references to the four original inputs, with hashes; no duplicate input files |

## Known limits

1. These are annotated-crop validation numbers. They are not detector,
   end-to-end or official-test results.
2. The make and body-type validation splits were created for those runs and were
   not verified for near-duplicate or same-physical-vehicle disjointness. Equal
   metric definitions do not make the three splits equally trustworthy.
3. Make has 19 classes with fewer than ten validation images, and `SAAB` and
   `Wealeak` are never predicted. Macro F1 weights every class equally, so those
   estimates carry full weight while being far less stable than the large
   classes.
4. The aggregate mode trusts the supplied matrix. It cannot detect duplicate
   identifiers, per-row label errors or leakage. A per-sample audit for make and
   body type is awaiting the underlying prediction files.
5. The colour pilot used an RTX 3060 Laptop GPU and the make/body pilots used an
   RTX 4070 Ti, on different machines. This evidence is arithmetic over saved
   files and used no accelerator, so it says nothing about hardware comparisons.

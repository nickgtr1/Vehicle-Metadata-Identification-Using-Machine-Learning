Three tasks, three separate results. No combined "overall F1" is produced. The class counts (10 / 75 / 12) and sample counts (386 / 1,120 / 782) differ, the tasks answer different questions, and no agreed weighting between them exists, so a single averaged figure would not correspond to anything the project is asking.

| Task | Split | Classes | Rows | Accuracy | Macro F1 | Macro F1 (no zero-support) | Weighted F1 | Labels | Input mode |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| colour | validation | 10 | 386 | 0.725389 | 0.729908 | 0.729908 | 0.726483 | none | row_level_predictions |
| make | validation | 75 | 1120 | 0.546429 | 0.589046 | 0.589046 | 0.553911 | none | aggregate_confusion_matrix |
| body_type | validation | 12 | 782 | 0.670077 | 0.696090 | 0.696090 | 0.673483 | none | aggregate_confusion_matrix |

## Control: display-only label trimming

Running the same make matrix with `--label-normalisation strip` reproduces the identical values; only the reported display names differ, for class IDs 31 (`Hyundai `) and 41 (`Lamorghini `).

| Task | Split | Classes | Rows | Accuracy | Macro F1 | Macro F1 (no zero-support) | Weighted F1 | Labels | Input mode |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| make (display trim) | validation | 75 | 1120 | 0.546429 | 0.589046 | 0.589046 | 0.553911 | strip | aggregate_confusion_matrix |

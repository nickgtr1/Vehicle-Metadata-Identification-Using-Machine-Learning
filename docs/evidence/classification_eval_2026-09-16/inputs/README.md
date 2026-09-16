# Aggregate input references from PR #5

These four inputs belong to Yuxiang Wang's PR #5, not Yuchen's training work.
No copies are included in this PR. Retrieve the originals from the pinned
commit `130917ec77c02bf5897f0d4919b262fb07803093`, then verify SHA-256 against
the table below before evaluating. A future PR #5 revision may differ.

- [make metrics](https://raw.githubusercontent.com/nickgtr1/Vehicle-Metadata-Identification-Using-Machine-Learning/130917ec77c02bf5897f0d4919b262fb07803093/docs/evidence/web_transfer_2026-09-16/make/validation_metrics.json)
- [make class mapping](https://raw.githubusercontent.com/nickgtr1/Vehicle-Metadata-Identification-Using-Machine-Learning/130917ec77c02bf5897f0d4919b262fb07803093/docs/evidence/web_transfer_2026-09-16/make/class_to_idx.json)
- [body-type metrics](https://raw.githubusercontent.com/nickgtr1/Vehicle-Metadata-Identification-Using-Machine-Learning/130917ec77c02bf5897f0d4919b262fb07803093/docs/evidence/web_transfer_2026-09-16/body_type/validation_metrics.json)
- [body-type class mapping](https://raw.githubusercontent.com/nickgtr1/Vehicle-Metadata-Identification-Using-Machine-Learning/130917ec77c02bf5897f0d4919b262fb07803093/docs/evidence/web_transfer_2026-09-16/body_type/class_to_idx.json)

For the documented CLI example, save them locally under the `File here` names
below, inside this folder; do not commit those downloaded copies. Alternatively,
pass their actual location explicitly to the CLI. Verify with
`Get-FileHash -Algorithm SHA256 <path>` in PowerShell.

| File here | Path in PR #5 | SHA-256 |
|---|---|---|
| `pr5_make_validation_metrics.json` | `docs/evidence/web_transfer_2026-09-16/make/validation_metrics.json` | `abc2b82623e38ef5b85272085a4e5de19bb3cf4fe92f6193d6d2385f31f2ce29` |
| `pr5_make_class_to_idx.json` | `docs/evidence/web_transfer_2026-09-16/make/class_to_idx.json` | `1e926b809db4e25ea2b26f650da63331a98d4782eb14c73272c96ed5b23ba82a` |
| `pr5_body_type_validation_metrics.json` | `docs/evidence/web_transfer_2026-09-16/body_type/validation_metrics.json` | `98f3f1d8468b194726eee0f33ba326e15ce6327a7a86a37bed6b4a97493e2541` |
| `pr5_body_type_class_to_idx.json` | `docs/evidence/web_transfer_2026-09-16/body_type/class_to_idx.json` | `1e7bb60f406856e4d85c7d4990564030b92226d743de25a817785e0e71afeb3c` |

The two `make` hashes match the values recorded in PR #5's own
`copy_provenance.json`, which is how the copies were confirmed to be faithful.

## What the files contain

`*_validation_metrics.json` hold the accuracy, macro F1, per-class recall,
confusion matrix, class order and row count for that pilot. A confusion matrix
is a sufficient statistic for precision, recall, F1 and accuracy, which is why
the make and body-type figures can be recomputed in the shared metric
definition without the per-sample predictions.

`*_class_to_idx.json` hold the class order the matrix rows and columns follow.

## Two things to know before comparing these with other tasks

The make class list contains `"Hyundai "` and `"Lamorghini "` with a trailing
space, inherited from the source manifest. The evaluation tool keeps those raw
strings as authoritative and treats trimming as an optional display adapter; a
collision-free rename cannot change a metric. `label_mapping.csv` in each run
folder records exactly which names differ and their class ids.

The body-type class list has 12 entries and excludes official type 0, which the
source manifest uses for an unavailable body label. PR #5 excludes those 721
rows from body-type work (369 fit, 352 test) while keeping them for make.

## Publication decision

On 17 September 2026, Yuchen chose references and hashes rather than duplicate
inputs, with the evaluation tool and its derived evidence in one PR.

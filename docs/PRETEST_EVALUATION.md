# Pre-test evaluation gates

Two entry points that run an attribute model over an explicit set of samples and
refuse anything they cannot evaluate honestly. They exist so that a formal
evaluation is a declared, frozen operation rather than a sequence of convenience
runs.

| Entry point | Purpose |
|---|---|
| `scripts/attribute_inference_eval.py` | run one task checkpoint over an explicit member list, write per-sample predictions, hand them to the shared evaluator |
| `scripts/verify_attribute_artifact.py` | check a checkpoint before it is used, and print the values a freeze record needs |

Both reuse the training preprocessing (`CompCarsMakeDataset`, `normalise_image`)
and the shared evaluator (`vehicle_id.attributes.evaluation`). They never train,
never download and never read a dataset image that was not explicitly selected.

## What the gates refuse

| Gate | Refusal condition |
|---|---|
| checkpoint structure | missing `state_dict`, `classes`, `config` or `synthetic`; config missing task, architecture, image size or preprocessing |
| task | the checkpoint's task differs from `--task` |
| architecture and image size | unsupported architecture, or a size disagreeing with the transform |
| preprocessing identity | a string outside the supported transform registry |
| class list | empty, duplicated or non-string names; a supplied mapping that disagrees with the checkpoint order |
| synthetic weights | the checkpoint is marked synthetic |
| membership source | `--membership-source` and `--dataset` are required; the descriptor's hash-pinned lists must match; split labels must be exactly `train` or `test` |
| membership join | a missing key column, a row with no entry in the source, or a key listed in two splits |
| split naming | a train-like split value carrying test members, the reverse, or an unrecognised split value |
| official test | any test member, unless `--allow-official-test` is given **and** a valid freeze record is supplied; sampling is refused outright for test members |
| freeze record | not machine-readable JSON, status other than `frozen`, an incomplete record, a code-hash block that is not exactly the required file set, or any bound value that does not match the run |
| device | a CUDA request is refused when CUDA is unavailable, and the model's parameter devices are checked before inference |
| output directory | not fresh; the tool never writes into a directory it does not own |

## What a freeze record binds

The record is a JSON document with schema `36127.frozen_protocol/v1` and status
`frozen`. It is compared against the run about to happen:

task, dataset, transform id, image size, checkpoint hash, class-order hash,
membership-source hash, the complete code-hash set, the split-file hash, the
selection columns and value, the label and key columns, the sample size, the
seed, the selected count, and a digest of the canonical selected rows (key,
image path, target label and bounding box).

The tool can emit a record with the real hashes filled in, but it writes
`status: "draft"` and cannot write `frozen`. A person has to make that change.

## Running it

```bash
# check an incoming checkpoint
python scripts/verify_attribute_artifact.py \
  --checkpoint <file> --task make --expected-sha256 <recorded hash> \
  --class-mapping <class list> --report <new directory>/intake_make.json

# evaluate one task over an explicit member list
python scripts/attribute_inference_eval.py \
  --checkpoint <validated checkpoint> --data-root <approved data root> \
  --split-file <membership table> --split-column experiment_split --split-value validation \
  --key-column relative_key --membership-source <hash-pinned descriptor> \
  --dataset <dataset id> --label-column <task label column> --task <task> \
  --class-mapping <class list> --device cpu --output <new empty directory>
```

Exit codes: `0` completed, `2` refused, `1` unexpected error. A refused run writes
nothing outside its own output directory, and if it refused because the output
directory was not fresh, it writes nothing at all.

## Current readiness

| Item | State |
|---|---|
| colour | a real checkpoint runs through this code and reproduces its recorded validation result |
| make and body type | blocked: the checkpoints are not available in this repository |
| live detection | blocked: needs detector weights; not required for annotated-crop attribute evaluation |
| official test | **not authorised and not run** |

Passing these gates means an evaluation is technically well formed. It is not a
statement that a model is good, that a split is free of near-duplicates, or that
the project may be scored against the official test set.

## Limitations

* Membership is pinned by hash, which fixes the bytes used. It does not by itself
  establish that a list is the official distribution; that remains a review step.
* A freeze record binds a configuration. Bypassing the gates remains possible for
  anyone who edits the code, so the gates are an integrity aid, not a defence
  against a determined operator.
* The gates cover the annotated-crop evaluation path. Detection accuracy, end-to-end
  behaviour and the interface are separate concerns.
* Near-duplicate and same-physical-vehicle disjointness are not checked here.

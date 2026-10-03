# Portable command templates

Placeholders: `<ckpt>` a validated checkpoint, `<data-root>` the CompCars extraction root, `<split>` the membership table, `<descriptor>` the hash-pinned membership descriptor, `<label>` the task label column, `<task>` one of `colour`, `make`, `body_type`, `<out>` a new empty directory, `<frozen>` the frozen protocol record.

```text
# 1. intake a checkpoint before using it
python scripts/verify_attribute_artifact.py \
  --checkpoint <ckpt> --task <task> --expected-sha256 <recorded hash> \
  --report <new directory>/intake_<task>.json

# 2. emit a freeze-record draft (validates membership, labels and hashes;
#    writes status: 'draft' and never overwrites an existing file)
python scripts/attribute_inference_eval.py \
  --checkpoint <ckpt> --data-root <data-root> \
  --split-file <split> --split-column experiment_split --split-value test \
  --key-column relative_key --membership-source <descriptor> \
  --dataset <dataset> --label-column <label> --task <task> \
  --device cpu --output <out> --emit-freeze-record <task>_freeze.json

# 3. a person reviews the draft and changes status to 'frozen'

# 4. the single declared run
python scripts/attribute_inference_eval.py \
  --checkpoint <ckpt> --data-root <data-root> \
  --split-file <split> --split-column experiment_split --split-value test \
  --key-column relative_key --membership-source <descriptor> \
  --dataset <dataset> --label-column <label> --task <task> --device cpu \
  --allow-official-test --protocol-frozen <frozen> --output <out>
```

Without the opt-in flag the tool refuses; with a `draft` record it refuses ('a draft is not an approval'); a non-fresh output directory is refused; sampling over official test members is refused.

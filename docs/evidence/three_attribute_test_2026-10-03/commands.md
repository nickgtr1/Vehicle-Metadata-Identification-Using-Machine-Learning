# Portable command template

Placeholders are replaced per task using the table below. Commands are written as
single lines so that no shell-specific line-continuation syntax is implied; they
work unchanged in bash, PowerShell and cmd.

| Placeholder | Meaning |
|---|---|
| `<ckpt>` | a checkpoint that has passed intake |
| `<data-root>` | the CompCars extraction root |
| `<split>` | the membership table for the task |
| `<descriptor>` | the hash-pinned membership descriptor for the dataset |
| `<intake-out>` | a new directory for the intake report |
| `<draft-out>` | a **new** directory for the generated draft record |
| `<run-out>` | a **different, new, empty** directory for the scored run |

Per-task values:

| Task | `<dataset>` | `<label>` | key column | membership source |
|---|---|---|---|---|
| colour | `compcars_surveillance` | `color_name` | `relative_key` | official surveillance classification lists |
| make | `compcars_web` | `make_name` | `relative_key` | official web classification lists |
| body_type | `compcars_web` | `car_type_name` | `relative_key` | official web classification lists |

```text
python scripts/verify_attribute_artifact.py --checkpoint <ckpt> --task <task> --expected-sha256 <recorded hash> --report <intake-out>/intake_<task>.json

python scripts/attribute_inference_eval.py --checkpoint <ckpt> --data-root <data-root> --split-file <split> --split-column experiment_split --split-value test --key-column relative_key --membership-source <descriptor> --dataset <dataset> --label-column <label> --task <task> --device cpu --output <draft-out> --emit-freeze-record <task>_freeze.json

python scripts/attribute_inference_eval.py --checkpoint <ckpt> --data-root <data-root> --split-file <split> --split-column experiment_split --split-value test --key-column relative_key --membership-source <descriptor> --dataset <dataset> --label-column <label> --task <task> --device cpu --allow-official-test --protocol-frozen <draft-out>/<task>_freeze.json --output <run-out>
```

Between the second and third commands, a person reviews `<draft-out>/<task>_freeze.json`
and changes `status` from `draft` to `frozen`. The tool never approves its own record.

`<draft-out>` and `<run-out>` must be different directories: emitting a draft creates
a file in `<draft-out>`, and the scoring run refuses any output directory that is not
fresh and empty.

Behaviour the tool enforces, unchanged by this document:

* without `--allow-official-test` the run refuses to score official test members;
* with a record whose `status` is `draft` the run refuses ("a draft is not an approval");
* sampling over official test members is refused;
* a non-fresh output directory is refused;
* every selected row must be registered in the membership source, and its label must
  be inside the checkpoint's class list.

"""Batch attribute inference on an explicit member list, with a unified metric report.

This is the pretest entry point. It runs one task checkpoint over an explicit
set of samples, writes per-sample predictions, and hands those predictions to
the shared evaluator.

Every gate is enforced rather than documented:

* membership comes from a trusted, hash-pinned descriptor whose split labels are
  canonical, never from the split name and never from a column in the file being
  run;
* a freeze record binds the whole evaluation context - checkpoint, class order,
  transform, membership source, split file, selection settings and the canonical
  selected rows - and must carry the exact required set of code hashes;
* the model is moved to the selected device, so a CUDA request cannot silently
  run on CPU;
* nothing is dropped silently, no output directory is overwritten, and the
  freeze-record template is never written over an existing file.

It never trains, never downloads and never writes outside its output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vehicle_id.attributes.dataset import CompCarsMakeDataset, resolve_image_path  # noqa: E402
from vehicle_id.attributes.evaluation import (  # noqa: E402
    EvaluationError,
    evaluate_rows,
    load_class_list,
    sha256_file,
)
from vehicle_id.attributes.modelling import AttributePredictor, normalise_image, predict_loader  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

MEMBERSHIP_SCHEMA = "36127.split_membership/v1"
FREEZE_SCHEMA = "36127.frozen_protocol/v1"

CANONICAL_SPLIT_LABELS = ("train", "test")

REQUIRED_DESCRIPTOR_FIELDS = ("schema", "dataset", "version", "key_format", "provenance", "lists")
REQUIRED_LIST_FIELDS = ("split", "path", "sha256")
REQUIRED_FREEZE_FIELDS = (
    "task",
    "dataset",
    "transform_id",
    "image_size",
    "checkpoint_sha256",
    "class_order_sha256",
    "membership_source_sha256",
    "code_hashes",
    "split_file_sha256",
    "split_column",
    "split_value",
    "label_column",
    "key_column",
    "sample_size",
    "seed",
    "selected_count",
    "selected_rows_sha256",
)

SUPPORTED_TRANSFORMS = {
    "OpenCV whole-crop resize 224x224, RGB, ImageNet mean/std": {
        "transform_id": "whole_crop_square_resize_imagenet_v1",
        "image_size": 224,
    },
    "OpenCV whole-crop square resize, RGB, ImageNet mean/std": {
        "transform_id": "whole_crop_square_resize_imagenet_v1",
        "image_size": 224,
    },
}

REQUIRED_CHECKPOINT_KEYS = ("state_dict", "classes", "config", "synthetic")
REQUIRED_CONFIG_KEYS = ("task", "architecture", "image_size", "preprocessing")
SUPPORTED_ARCHITECTURES = ("resnet18", "tiny_cnn")

TEST_LIKE_SPLITS = {"test", "tests", "testing", "officialtest", "testset", "holdout", "holdouts", "holdoutset"}
TRAIN_LIKE_SPLITS = {"train", "training", "fit", "validation", "valid", "val", "dev"}


class InferenceError(RuntimeError):
    """Raised for any input the run refuses to infer on."""


def normalise_split_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch attribute inference plus unified metrics.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--split-column", required=True)
    parser.add_argument("--split-value", required=True)
    parser.add_argument("--key-column", default="relative_key")
    parser.add_argument("--membership-source", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--class-mapping", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=36127)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--allow-official-test", action="store_true")
    parser.add_argument("--protocol-frozen", type=Path, default=None)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--emit-freeze-record", nargs="?", const=Path("freeze_record_template.json"), type=Path,
                        default=None, help="write a draft record inside --output, then exit")
    return parser.parse_args(argv)


def prepare_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise InferenceError(f"Output directory is not fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)


def environment() -> dict:
    import importlib.metadata as metadata

    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "torchvision", "Pillow", "opencv-python"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "packages": packages,
    }


def code_hashes() -> dict:
    """The exact set of files a freeze record must cover."""
    return {
        "scripts/attribute_inference_eval.py": sha256_file(Path(__file__).resolve()),
        "src/vehicle_id/attributes/evaluation.py": sha256_file(
            REPO_ROOT / "src" / "vehicle_id" / "attributes" / "evaluation.py"
        ),
        "src/vehicle_id/attributes/dataset.py": sha256_file(
            REPO_ROOT / "src" / "vehicle_id" / "attributes" / "dataset.py"
        ),
        "src/vehicle_id/attributes/modelling.py": sha256_file(
            REPO_ROOT / "src" / "vehicle_id" / "attributes" / "modelling.py"
        ),
    }


def class_order_digest(classes: list[str]) -> str:
    return hashlib.sha256(json.dumps(list(classes), ensure_ascii=False).encode("utf-8")).hexdigest()


def selected_rows_digest(selected: pd.DataFrame, key_column: str, label_column: str) -> str:
    """Digest of the canonical selected rows: key, image path, target label, bbox.

    Order-independent, so the same member set hashes the same way whatever order
    the table is in, while any change to a row, its label or its box changes the
    digest.
    """
    rows = sorted(
        [
            [
                str(record[key_column]),
                str(record.get("image_path")),
                str(record.get(label_column)),
                str(record.get("bbox")),
            ]
            for record in selected.to_dict("records")
        ]
    )
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_checkpoint_payload(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - reported verbatim
        raise InferenceError(f"Checkpoint could not be read: {type(error).__name__}: {error}") from error
    if not isinstance(payload, dict):
        raise InferenceError("Checkpoint payload is not a dictionary")
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise InferenceError(
            f"Checkpoint is missing required key(s) {missing}; expected {list(REQUIRED_CHECKPOINT_KEYS)}"
        )
    if not isinstance(payload["config"], dict):
        raise InferenceError("Checkpoint config is not a dictionary")
    config_missing = [key for key in REQUIRED_CONFIG_KEYS if key not in payload["config"]]
    if config_missing:
        raise InferenceError(
            f"Checkpoint config is missing {config_missing}; the task, architecture, image size and "
            "transform must all be declared"
        )
    classes = payload["classes"]
    if not isinstance(classes, (list, tuple)) or not classes:
        raise InferenceError("Checkpoint classes must be a non-empty list")
    if not all(isinstance(name, str) and name for name in classes):
        raise InferenceError("Checkpoint class names must be non-empty strings")
    if len(set(classes)) != len(classes):
        raise InferenceError("Checkpoint class list contains duplicates")
    return payload


def validate_transform(config: dict) -> dict:
    declared = config["preprocessing"]
    if declared not in SUPPORTED_TRANSFORMS:
        raise InferenceError(
            f"Checkpoint declares an unsupported preprocessing contract ({declared!r}). "
            f"Supported: {sorted(SUPPORTED_TRANSFORMS)}"
        )
    entry = SUPPORTED_TRANSFORMS[declared]
    if config["image_size"] != entry["image_size"]:
        raise InferenceError(
            f"Checkpoint image_size {config['image_size']} disagrees with the transform contract "
            f"{entry['image_size']} for {entry['transform_id']}"
        )
    if config["architecture"] not in SUPPORTED_ARCHITECTURES:
        raise InferenceError(f"Unsupported checkpoint architecture '{config['architecture']}'")
    return {"declared": declared, **entry}


def _require_text(descriptor: dict, field: str) -> str:
    value = descriptor.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InferenceError(f"Membership descriptor field '{field}' must be a non-empty string")
    return value


def load_membership_source(path: Path, dataset: str) -> dict:
    """Read and verify a trusted membership descriptor, then build key -> split.

    The split labels are canonicalised by rejection: only the exact strings
    ``train`` and ``test`` are accepted. Anything else - whitespace, different
    case, an unknown word - stops the run, because a label the gates do not
    recognise would otherwise sit outside every check.
    """
    if not path.is_file():
        raise InferenceError(f"Membership source not found: {path}")
    try:
        descriptor = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InferenceError(f"Membership source is not valid JSON: {path} ({error})") from error
    if not isinstance(descriptor, dict):
        raise InferenceError("Membership descriptor must be a JSON object")
    if descriptor.get("schema") != MEMBERSHIP_SCHEMA:
        raise InferenceError(f"Membership source must declare schema {MEMBERSHIP_SCHEMA}")
    missing = [field for field in REQUIRED_DESCRIPTOR_FIELDS if field not in descriptor]
    if missing:
        raise InferenceError(f"Membership descriptor is missing {missing}")
    if descriptor.get("dataset") != dataset:
        raise InferenceError(
            f"Membership source is for dataset {descriptor.get('dataset')!r} but --dataset is {dataset!r}; "
            "datasets must not be conflated"
        )
    version = _require_text(descriptor, "version")
    provenance = _require_text(descriptor, "provenance")
    key_format = _require_text(descriptor, "key_format")
    lists = descriptor["lists"]
    if not isinstance(lists, list) or not lists:
        raise InferenceError("Membership descriptor must list at least one split file")

    seen_splits: set[str] = set()
    seen_paths: set[str] = set()
    members: dict[str, str] = {}
    conflicts: list[str] = []
    verified: list[dict] = []
    for entry in lists:
        if not isinstance(entry, dict):
            raise InferenceError(f"Membership list entry must be an object: {entry!r}")
        entry_missing = [field for field in REQUIRED_LIST_FIELDS if field not in entry]
        if entry_missing:
            raise InferenceError(f"Membership list entry is missing {entry_missing}: {entry}")
        label = entry["split"]
        if not isinstance(label, str) or label not in CANONICAL_SPLIT_LABELS:
            raise InferenceError(
                f"Membership split label {label!r} is not canonical. Only {list(CANONICAL_SPLIT_LABELS)} are "
                "accepted, exactly, so that no row can carry a label the gates do not recognise."
            )
        if label in seen_splits:
            raise InferenceError(f"Membership descriptor lists split '{label}' more than once")
        seen_splits.add(label)
        list_path = Path(entry["path"])
        if str(list_path) in seen_paths:
            raise InferenceError(f"Membership descriptor lists the same file twice: {list_path}")
        seen_paths.add(str(list_path))
        if not list_path.is_file():
            raise InferenceError(f"Membership list file not found: {list_path}")
        digest = entry["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise InferenceError(f"Membership list '{label}' must pin a 64-character hex sha256")
        actual = sha256_file(list_path)
        if actual != digest.lower():
            raise InferenceError(
                f"Membership list {list_path.name} hash mismatch: descriptor says {digest[:16]}..., "
                f"file is {actual[:16]}..."
            )
        keys = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for key in keys:
            if key in members and members[key] != label:
                conflicts.append(key)
            members[key] = label
        verified.append({"split": label, "path": str(list_path), "sha256": actual, "keys": len(keys)})
    if conflicts:
        raise InferenceError(
            f"{len(conflicts)} key(s) appear in more than one split list, which is a conflict, not a "
            f"permission: {conflicts[:5]}"
        )
    # Defence in depth: after the loop the label universe is guaranteed canonical.
    unexpected = sorted(set(members.values()) - set(CANONICAL_SPLIT_LABELS))
    if unexpected:
        raise InferenceError(f"Membership contains non-canonical labels: {unexpected}")
    return {
        "descriptor": descriptor,
        "descriptor_path": str(path),
        "descriptor_sha256": sha256_file(path),
        "version": version,
        "provenance": provenance,
        "key_format": key_format,
        "members": members,
        "verified_lists": verified,
    }


def validate_freeze_record(path: Path, context: dict) -> dict:
    """Validate a freeze record against the exact run about to happen."""
    if not path.is_file():
        raise InferenceError(f"Freeze record not found: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InferenceError(
            f"The freeze record is not machine-readable JSON: {path} ({error}). A text file, a note or a "
            "source file is not a freeze record."
        ) from error
    if not isinstance(record, dict):
        raise InferenceError("The freeze record must be a JSON object")
    if record.get("schema") != FREEZE_SCHEMA:
        raise InferenceError(f"The freeze record must declare schema {FREEZE_SCHEMA}")
    if record.get("status") != "frozen":
        raise InferenceError(
            f"The freeze record status is {record.get('status')!r}; only 'frozen' is accepted. "
            "A draft is not an approval."
        )
    missing = [field for field in REQUIRED_FREEZE_FIELDS if field not in record]
    if missing:
        raise InferenceError(f"The freeze record is incomplete: {missing}")

    mismatches = []
    for field in REQUIRED_FREEZE_FIELDS:
        if field == "code_hashes":
            continue
        expected = context.get(field)
        if record[field] != expected:
            mismatches.append({"field": field, "record": record[field], "actual": expected})

    # The code-hash block must cover exactly the required file set, no more and
    # no less: a subset would let a record omit the very files it claims to fix.
    recorded_hashes = record["code_hashes"]
    if not isinstance(recorded_hashes, dict):
        raise InferenceError("The freeze record's code_hashes must be a JSON object")
    required_hashes = context["code_hashes"]
    extra = sorted(set(recorded_hashes) - set(required_hashes))
    absent = sorted(set(required_hashes) - set(recorded_hashes))
    if absent:
        raise InferenceError(
            f"The freeze record's code_hashes is incomplete: {absent}. Every file the run depends on must "
            "be pinned, so an empty or partial block is not accepted."
        )
    if extra:
        raise InferenceError(f"The freeze record's code_hashes names unknown file(s): {extra}")
    for name, actual in required_hashes.items():
        if recorded_hashes[name] != actual:
            mismatches.append({"field": f"code_hashes.{name}", "record": recorded_hashes[name], "actual": actual})

    if mismatches:
        detail = "; ".join(
            f"{item['field']} record={str(item['record'])[:16]} actual={str(item['actual'])[:16]}"
            for item in mismatches[:6]
        )
        raise InferenceError(f"The freeze record does not match this run: {detail}")
    return {"path": str(path), "sha256": sha256_file(path), "record": record}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log: list[str] = [f"argv: {sys.argv}"]
    try:
        return _run(args, log)
    except (InferenceError, EvaluationError, FileNotFoundError, ValueError) as error:
        log.append(f"FAILED: {type(error).__name__}: {error}")
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, indent=2))
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, log: list[str]) -> int:
    for path, label in (
        (args.checkpoint, "Checkpoint"),
        (args.split_file, "Split file"),
        (args.membership_source, "Membership source"),
    ):
        if not path.is_file():
            raise InferenceError(f"{label} not found: {path}")
    if not args.data_root.is_dir():
        raise InferenceError(f"Data root not found: {args.data_root}")
    # The opt-in and freeze-record pairing is checked below, after the draft-emit
    # branch: emitting a record is how a record comes to exist, so it must not
    # require one first.

    payload = load_checkpoint_payload(args.checkpoint)
    config = payload["config"]
    classes = list(payload["classes"])
    checkpoint_hash = sha256_file(args.checkpoint)
    transform = validate_transform(config)
    if config["task"] != args.task:
        raise InferenceError(f"Checkpoint task '{config['task']}' does not match --task '{args.task}'")
    if payload["synthetic"] and not args.allow_synthetic:
        raise InferenceError("Synthetic checkpoints are not usable for pretest evidence")
    if args.class_mapping is not None:
        declared = load_class_list(args.class_mapping)
        if declared != classes:
            raise InferenceError(
                "Class mapping does not match the checkpoint's ordered classes; the checkpoint order is authoritative"
            )
    class_digest = class_order_digest(classes)
    log.append(f"checkpoint {args.checkpoint} sha256={checkpoint_hash}")
    log.append(f"task={config['task']} architecture={config['architecture']} image_size={config['image_size']}")
    log.append(f"transform={transform['transform_id']}")

    source = load_membership_source(args.membership_source, args.dataset)
    log.append(
        f"membership source {args.membership_source} sha256={source['descriptor_sha256']} "
        f"version={source['version']!r} keys={len(source['members'])}"
    )

    frame = pd.read_csv(args.split_file)
    if args.split_column not in frame.columns:
        raise InferenceError(f"Column '{args.split_column}' missing; available: {list(frame.columns)}")
    if args.label_column not in frame.columns:
        raise InferenceError(f"Label column '{args.label_column}' missing; available: {list(frame.columns)}")
    if args.key_column not in frame.columns:
        raise InferenceError(
            f"Key column '{args.key_column}' missing; membership cannot be joined without a canonical key. "
            f"Available: {list(frame.columns)}"
        )

    selected = frame[frame[args.split_column] == args.split_value].copy()
    log.append(f"rows selected by {args.split_column} == '{args.split_value}': {len(selected)}")
    if selected.empty:
        raise InferenceError(
            f"No rows have {args.split_column} == '{args.split_value}'. Membership is joined from the "
            "trusted source, so a name alone selects nothing."
        )

    keys = selected[args.key_column].astype(str).tolist()
    unknown = sorted({key for key in keys if key not in source["members"]})
    if unknown:
        raise InferenceError(
            f"{len(unknown)} selected row(s) have no entry in the membership source, so membership is "
            f"unknown and permission cannot be assumed. Examples: {unknown[:5]}"
        )
    counts = {label: 0 for label in CANONICAL_SPLIT_LABELS}
    for key in keys:
        counts[source["members"][key]] += 1
    log.append(f"membership of the selected rows: {counts}")

    split_name = normalise_split_name(args.split_value)
    if split_name in TRAIN_LIKE_SPLITS and counts["test"]:
        raise InferenceError(
            f"The split value '{args.split_value}' is train-like but {counts['test']} selected row(s) are "
            "official test members. A relabelled split is not permission."
        )
    if split_name in TEST_LIKE_SPLITS and counts["train"]:
        raise InferenceError(
            f"The split value '{args.split_value}' is test-like but {counts['train']} selected row(s) are "
            "not official test members. A relabelled split is not permission."
        )
    if split_name not in TRAIN_LIKE_SPLITS | TEST_LIKE_SPLITS:
        raise InferenceError(
            f"The split value '{args.split_value}' is neither train-like nor test-like; state which it is "
            f"so the membership rules can be applied. Train-like: {sorted(TRAIN_LIKE_SPLITS)}; "
            f"test-like: {sorted(TEST_LIKE_SPLITS)}"
        )

    if args.sample_size is not None:
        if args.sample_size < 1:
            raise InferenceError("--sample-size must be positive")
        if args.sample_size > len(selected):
            raise InferenceError(f"--sample-size {args.sample_size} exceeds the {len(selected)} available rows")
        if counts["test"]:
            raise InferenceError(
                "Sampling is refused when official test members are selected: a formal scoring run uses "
                "the whole declared member set."
            )
        selected = selected.sample(n=args.sample_size, random_state=args.seed).copy()
        log.append(f"seeded sample of {len(selected)} rows, seed={args.seed}")

    duplicated = selected[args.key_column][selected[args.key_column].duplicated()].unique().tolist()
    if duplicated:
        raise InferenceError(f"Duplicate sample identifiers in '{args.key_column}': {duplicated[:5]}")

    unknown_labels = sorted(set(selected[args.label_column].dropna()) - set(classes))
    if unknown_labels:
        raise InferenceError(
            f"{len(unknown_labels)} label(s) are outside the checkpoint class list: {unknown_labels[:10]}"
        )
    missing_labels = int(selected[args.label_column].isna().sum())
    if missing_labels:
        raise InferenceError(f"{missing_labels} selected rows have no target label")

    context = {
        "task": args.task,
        "dataset": args.dataset,
        "transform_id": transform["transform_id"],
        "image_size": transform["image_size"],
        "checkpoint_sha256": checkpoint_hash,
        "class_order_sha256": class_digest,
        "membership_source_sha256": source["descriptor_sha256"],
        "code_hashes": code_hashes(),
        "split_file_sha256": sha256_file(args.split_file),
        "split_column": args.split_column,
        "split_value": args.split_value,
        "label_column": args.label_column,
        "key_column": args.key_column,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "selected_count": int(len(selected)),
        "selected_rows_sha256": selected_rows_digest(selected, args.key_column, args.label_column),
    }

    if args.emit_freeze_record is not None:
        prepare_output(args.output)
        target = args.emit_freeze_record
        if not target.is_absolute():
            target = args.output / target
        resolved_target = target.resolve()
        output_root = args.output.resolve()
        # Resolved-path ancestry, not a string prefix: 'out_sibling' starts with
        # the characters of 'out' but is not inside it.
        if not resolved_target.is_relative_to(output_root):
            raise InferenceError(
                f"The freeze-record template must be written inside --output; {resolved_target} is outside "
                f"{output_root}"
            )
        if resolved_target.exists():
            raise InferenceError(
                f"Refusing to overwrite an existing file: {resolved_target}. A reviewed record must never "
                "be replaced by a generated draft."
            )
        draft = {
            "schema": FREEZE_SCHEMA,
            "status": "draft",
            "note": (
                "Generated by the tool with the real context hashes. A human must review it and change "
                "status to 'frozen'; the tool never approves its own record."
            ),
            "frozen_utc": None,
            **context,
        }
        with resolved_target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(draft, indent=2))
        print(json.dumps({"status": "draft_written", "path": str(resolved_target)}, indent=2))
        return 0

    freeze: dict | None = None
    if counts["test"]:
        if not args.allow_official_test:
            raise InferenceError(
                f"{counts['test']} of {len(selected)} selected rows are official test members. "
                "Inference on the official test set is refused by default."
            )
        if args.protocol_frozen is None:
            raise InferenceError("--allow-official-test requires --protocol-frozen <path>")
        freeze = validate_freeze_record(args.protocol_frozen, context)
        log.append(f"official test inference authorised by CLI opt-in and freeze record {freeze['sha256']}")
        log.append(
            "note: the CLI opt-in is an explicit act by the caller; it is not independent user authorisation"
        )

    unreadable = []
    for value in selected["image_path"]:
        try:
            path = resolve_image_path(value, args.data_root)
        except ValueError as error:
            unreadable.append({"image_path": str(value), "reason": str(error)})
            continue
        if not path.is_file():
            unreadable.append({"image_path": str(value), "reason": "missing file"})
    if unreadable:
        raise InferenceError(
            f"{len(unreadable)} selected image(s) are missing or unusable; nothing was inferred. "
            f"First examples: {unreadable[:3]}"
        )
    log.append(f"pre-flight passed: {len(selected)} images resolve and exist")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise InferenceError("--device cuda requested but CUDA is unavailable")

    predictor = AttributePredictor(args.checkpoint, allow_synthetic=args.allow_synthetic)
    predictor.model.to(device)
    predictor.model.eval()
    # Compare device *types*: str(device) is 'cuda:0', not 'cuda'.
    parameter_devices = {parameter.device.type for parameter in predictor.model.parameters()}
    if parameter_devices != {device}:
        raise InferenceError(
            f"Model parameters ended up on {sorted(parameter_devices)} while device is '{device}'"
        )
    log.append(f"model moved to {device}; parameter devices {sorted(parameter_devices)}")

    prepare_output(args.output)
    log.append(f"output directory created: {args.output}")

    dataset = CompCarsMakeDataset(
        selected, args.label_column, config["image_size"], normalise_image, args.data_root, classes
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    torch.set_num_threads(int(config.get("cpu_threads", 2)))

    started = datetime.now(timezone.utc)
    targets, predictions, scores = predict_loader(predictor.model, loader, device)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log.append(f"inferred {len(targets)} rows on {device} in {elapsed:.2f}s")

    predictions_frame = pd.DataFrame(
        {
            "sample_id": selected[args.key_column].tolist(),
            "task": args.task,
            "split": args.split_value,
            "declared_membership": [source["members"][str(key)] for key in selected[args.key_column]],
            "target": [classes[index] for index in targets],
            "prediction": [classes[index] for index in predictions],
            "uncalibrated_score": scores,
            "checkpoint_sha256": checkpoint_hash,
        }
    )
    predictions_path = args.output / "predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False)
    log.append(f"wrote {predictions_path.name}")

    reported, per_class, excluded, matrix, mapping = evaluate_rows(
        predictions_frame,
        classes,
        target_column="target",
        prediction_column="prediction",
        allow_unusable_rows=False,
        id_column="sample_id",
        expected_rows=len(selected),
    )
    payload_out = {
        "schema": "36127.attribute_inference_eval/v3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "dataset": args.dataset,
        "split": args.split_value,
        "scope": "small non-test sample" if args.sample_size else "full declared split",
        "warning": (
            "A sample is a pipeline smoke test. It is not a reproduction of the recorded "
            "386 / 1120 / 782 sample validation results."
            if args.sample_size
            else None
        ),
        "counts": {
            "rows_available_in_split": int(len(frame[frame[args.split_column] == args.split_value])),
            "rows_inferred": int(len(selected)),
            "correct": reported["correct"],
            "declared_membership": counts,
            "unknown_membership": 0,
        },
        "accuracy": reported["accuracy"],
        "macro_f1": reported["macro_f1"],
        "weighted_f1": reported["weighted_f1"],
        "macro_precision": reported["macro_precision"],
        "macro_recall": reported["macro_recall"],
        "classes": classes,
        "per_class": reported["per_class"],
        "confusion_matrix": matrix.tolist(),
        "arithmetic_check": reported["arithmetic_check"],
        "matrix_cross_check": reported["matrix_cross_check"],
        "zero_support_class_count": reported["zero_support_class_count"],
        "membership": {
            "source_path": source["descriptor_path"],
            "source_sha256": source["descriptor_sha256"],
            "dataset": source["descriptor"].get("dataset"),
            "version": source["version"],
            "provenance": source["provenance"],
            "key_format": source["key_format"],
            "verified_lists": source["verified_lists"],
            "join_key": args.key_column,
            "resolved_from": "a hash-pinned official list, never from the split name or a column in the split file",
            "origin_caveat": (
                "the tool pins what was used and cannot by itself establish that a list is the official "
                "distribution; a reviewer must confirm the provenance field against the source"
            ),
        },
        "freeze_record": freeze if freeze else {"used": False, "reason": "no official test member was selected"},
        "evaluation_context": context,
        "transform": transform,
        "inputs": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "class_order_sha256": class_digest,
            "split_file": str(args.split_file),
            "split_file_sha256": context["split_file_sha256"],
            "data_root": str(args.data_root),
            "label_column": args.label_column,
            "sample_size": args.sample_size,
        },
        "preprocessing": {
            "source": "vehicle_id.attributes.dataset.CompCarsMakeDataset, the training loader",
            "transform_id": transform["transform_id"],
            "declared_by_checkpoint": transform["declared"],
            "chain": "cv2 imdecode -> BGR to RGB -> bbox crop (already converted once) -> square resize -> ImageNet mean/std",
            "bbox_handling": "the split file carries zero-based coordinates already; no second conversion is applied",
        },
        "equivalence_claim": (
            "Reproducing a recorded result on the same rows supports equivalence for those rows and that "
            "checkpoint. It is not a claim of equivalence for every image, dataset or model."
        ),
        "environment": environment(),
        "code_hashes": context["code_hashes"],
        "seed": args.seed,
        "device": device,
        "model_parameter_devices": sorted(parameter_devices),
        "elapsed_seconds": elapsed,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    per_class.to_csv(args.output / "per_class_metrics.csv", index=False)
    mapping.to_csv(args.output / "label_mapping.csv", index=False)
    (args.output / "environment.json").write_text(json.dumps(payload_out["environment"], indent=2), encoding="utf-8")
    (args.output / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "task": args.task,
                "split": args.split_value,
                "rows_inferred": len(selected),
                "correct": reported["correct"],
                "accuracy": reported["accuracy"],
                "macro_f1": reported["macro_f1"],
                "weighted_f1": reported["weighted_f1"],
                "device": device,
                "scope": payload_out["scope"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

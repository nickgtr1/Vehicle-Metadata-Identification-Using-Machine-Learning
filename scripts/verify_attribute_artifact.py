"""Intake check for an attribute checkpoint before it is used in an evaluation.

Run this on each incoming checkpoint. It applies the same gates the inference
entry point applies, so an "acceptable" verdict means the runner will accept the
file, and it prints the values a freeze record needs.

It never trains, never downloads and never reads a dataset image.

Exit codes: 0 acceptable, 2 refused, 1 unexpected error.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vehicle_id.attributes.evaluation import EvaluationError, load_class_list, sha256_file  # noqa: E402
from vehicle_id.attributes.modelling import build_model  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "attribute_inference_eval_for_intake", Path(__file__).resolve().parent / "attribute_inference_eval.py"
)
entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entry)


class IntakeRefusal(RuntimeError):
    """Raised when the artifact cannot be used as supplied."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check an attribute checkpoint before use.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", required=True, help="expected task: colour, make or body_type")
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--class-mapping", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="write the full report as JSON")
    parser.add_argument("--skip-state-dict-load", action="store_true",
                        help="skip building the model; useful when the architecture is unavailable offline")
    return parser.parse_args(argv)


def inspect(args: argparse.Namespace) -> dict:
    """Run every intake check, reporting any refusal as IntakeRefusal.

    The structural gates live in the inference entry point, so their errors are
    translated here rather than duplicated.
    """
    try:
        return _inspect(args)
    except IntakeRefusal:
        raise
    except (entry.InferenceError, EvaluationError) as error:
        raise IntakeRefusal(str(error)) from error


def _inspect(args: argparse.Namespace) -> dict:
    if not args.checkpoint.is_file():
        raise IntakeRefusal(f"Checkpoint not found: {args.checkpoint}")
    digest = sha256_file(args.checkpoint)
    if args.expected_sha256 and digest != args.expected_sha256.lower():
        raise IntakeRefusal(
            f"Hash mismatch: supplied {args.expected_sha256[:16]}..., file is {digest[:16]}.... "
            "The file is not the one that was agreed."
        )

    payload = entry.load_checkpoint_payload(args.checkpoint)
    config = payload["config"]
    classes = list(payload["classes"])
    if config["task"] != args.task:
        raise IntakeRefusal(f"Checkpoint task '{config['task']}' does not match the expected task '{args.task}'")
    transform = entry.validate_transform(config)
    if payload["synthetic"]:
        raise IntakeRefusal("The checkpoint is marked synthetic; it is not a usable model")

    mapping_state = "not supplied"
    if args.class_mapping is not None:
        declared = load_class_list(args.class_mapping)
        if declared != classes:
            raise IntakeRefusal(
                "The supplied class mapping does not match the checkpoint's ordered classes; the checkpoint "
                "order is authoritative"
            )
        mapping_state = f"matches the checkpoint ({len(declared)} classes)"

    state_dict_tensors = len(payload["state_dict"])
    load_state = "skipped"
    parameter_count = None
    if not args.skip_state_dict_load:
        model = build_model(config["architecture"], len(classes))
        try:
            model.load_state_dict(payload["state_dict"], strict=True)
        except RuntimeError as error:
            raise IntakeRefusal(
                f"The state dict does not load into a {config['architecture']} head with {len(classes)} "
                f"classes: {error}"
            ) from error
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        load_state = "loaded strictly"

    duplicates = sorted({name for name in classes if classes.count(name) > 1})
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "bytes": args.checkpoint.stat().st_size,
        "sha256": digest,
        "expected_sha256": args.expected_sha256,
        "task": config["task"],
        "architecture": config["architecture"],
        "image_size": config["image_size"],
        "transform_declared": transform["declared"],
        "transform_id": transform["transform_id"],
        "pilot_only": config.get("pilot_only"),
        "synthetic": bool(payload["synthetic"]),
        "class_count": len(classes),
        "duplicate_classes": duplicates,
        "class_mapping": mapping_state,
        "class_order_sha256": entry.class_order_digest(classes),
        "state_dict_tensors": state_dict_tensors,
        "state_dict_load": load_state,
        "parameter_count": parameter_count,
        "freeze_record_values": {
            "task": config["task"],
            "transform_id": transform["transform_id"],
            "image_size": transform["image_size"],
            "checkpoint_sha256": digest,
            "class_order_sha256": entry.class_order_digest(classes),
        },
        "note": (
            "A hash pins the bytes that were used. Confirming that the file is the intended model, and that "
            "the declared preprocessing is the one training used, remains a review step."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = inspect(args)
    except (IntakeRefusal, EvaluationError) as error:
        outcome = {"status": "refused", "error": str(error), "checkpoint": str(args.checkpoint)}
        print(json.dumps(outcome, indent=2))
        print(f"error: {error}", file=sys.stderr)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
        return 2
    except Exception as error:  # noqa: BLE001 - reported verbatim
        outcome = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(outcome, indent=2))
        print(f"error: {error}", file=sys.stderr)
        return 1

    report["status"] = "acceptable"
    report["checked_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

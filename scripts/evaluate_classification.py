"""Command-line entry point for the unified attribute classification evaluation.

Two input modes share one metric specification:

* ``--predictions``: row-level samples (identifier, true label, predicted label).
  This is the primary and preferred mode.
* ``--confusion-matrix``: an already-published aggregate matrix. Use this only
  when raw predictions are unavailable; the reported numbers are then derived
  from somebody else's aggregate and cannot be re-audited row by row.

The script never selects an input path by itself, never downloads anything and
refuses to write into an output directory that already contains files.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vehicle_id.attributes.evaluation import (  # noqa: E402
    LABEL_NORMALISATION_POLICY,
    LABEL_NORMALISATIONS,
    MACRO_POLICY,
    ZERO_DIVISION_POLICY,
    DatasetDescription,
    EvaluationError,
    UnusableRowsError,
    environment_summary,
    evaluate_rows,
    load_class_list,
    make_label_normaliser,
    metrics_from_confusion_matrix,
    read_confusion_matrix,
    resolve_display_classes,
    sha256_file,
    SPLIT_NAMES_REQUIRING_CONFIRMATION,
    split_requires_confirmation,
)

PER_CLASS_COLUMNS = [
    "class",
    "class_index",
    "display_class",
    "precision",
    "recall",
    "f1",
    "support",
    "predicted",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one attribute classification split with the shared metric specification."
    )
    parser.add_argument("--task", required=True, help="Attribute task name, for example colour or body_type")
    parser.add_argument("--split", required=True, help="Split name, for example fit, validation or official_test")
    parser.add_argument("--output", required=True, help="Fresh output directory; must not contain files")
    parser.add_argument("--class-mapping", help="JSON class list or class_to_idx.json mapping")
    parser.add_argument("--predictions", help="CSV of row-level predictions")
    parser.add_argument("--confusion-matrix", help="JSON or CSV confusion matrix for aggregate-only input")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--id-column", default="relative_key")
    parser.add_argument("--score-column", default="uncalibrated_score")
    parser.add_argument("--expected-rows", type=int, default=None, help="Fail when the input row count differs")
    parser.add_argument(
        "--label-normalisation",
        choices=list(LABEL_NORMALISATIONS),
        default="none",
        help="Adapter applied identically to the class list, targets and predictions; default keeps raw labels",
    )
    parser.add_argument("--source", default=None, help="Short description of where the predictions came from")
    parser.add_argument("--notes", default=None, help="Free-form note stored in metrics.json")
    parser.add_argument("--seed", default="not_applicable", help="Seed used to produce the predictions")
    parser.add_argument("--device", default="unused", help="Device used to produce the predictions")
    parser.add_argument(
        "--allow-unknown-labels",
        action="store_true",
        help="Exclude rows that cannot be scored against the class list instead of aborting; every row is recorded",
    )
    parser.add_argument(
        "--allow-official-test",
        action="store_true",
        help=(
            "Confirm that an official test split is intentionally being evaluated. Without it, "
            "the split name is refused when, ignoring case and separators, it is one of: "
            + ", ".join(SPLIT_NAMES_REQUIRING_CONFIRMATION)
        ),
    )
    return parser


def prepare_output(output: Path, log: list[str]) -> None:
    """Take ownership of a fresh output directory, or refuse without touching anything.

    Called immediately before the first write, so a run that is going to be
    rejected never gets near an existing directory. Nothing is read, written or
    removed from a directory this call refuses.
    """
    if output.exists():
        if not output.is_dir():
            raise EvaluationError(f"Output path exists and is not a directory: {output}")
        entries = list(output.iterdir())
        if entries:
            raise EvaluationError(
                f"Output directory is not fresh: {output} already holds {len(entries)} entry/entries. "
                "Nothing in it was read, written or removed. Choose a new directory."
            )
        log.append(f"output directory acquired (existed and was empty): {output}")
        return
    output.mkdir(parents=True, exist_ok=False)
    log.append(f"output directory created: {output}")


def describe_policy() -> dict[str, str]:
    return {
        "class_list": "fixed by the caller and saved in declared order",
        "label_normalisation": LABEL_NORMALISATION_POLICY,
        "zero_division": ZERO_DIVISION_POLICY,
        "macro": MACRO_POLICY,
        "unusable_rows": "abort unless --allow-unknown-labels is given; excluded rows are always written out",
        "official_test": (
            "a split name requires --allow-official-test when, ignoring case and separators, it is one of: "
            + ", ".join(SPLIT_NAMES_REQUIRING_CONFIRMATION)
        ),
        "confusion_counts": (
            "counts must be integers; integral floats such as 1.0 are accepted, anything fractional, "
            "negative, non-finite, boolean, textual or out of int64 range is refused"
        ),
        "class_indices": "class mapping indices must be JSON integers; floats, booleans and strings are refused",
        "expected_rows": "checked against the row count in prediction mode and against the matrix total in aggregate mode",
    }


def run_from_predictions(args: argparse.Namespace, log: list[str]) -> dict:
    if not args.class_mapping:
        raise EvaluationError("--class-mapping is required with --predictions")
    classes = load_class_list(Path(args.class_mapping))
    log.append(f"class list loaded: {len(classes)} classes from {args.class_mapping}")

    predictions_path = Path(args.predictions)
    if not predictions_path.is_file():
        raise EvaluationError(f"Predictions file not found: {predictions_path}")
    frame = pd.read_csv(predictions_path)
    log.append(f"predictions loaded: {len(frame)} rows, columns {list(frame.columns)}")

    reported, per_class_frame, excluded, matrix, mapping_frame = evaluate_rows(
        frame,
        classes,
        target_column=args.target_column,
        prediction_column=args.prediction_column,
        allow_unusable_rows=args.allow_unknown_labels,
        id_column=args.id_column,
        expected_rows=args.expected_rows,
        label_normalisation=args.label_normalisation,
    )
    log.append(f"rows scored: {reported['rows']}, correct: {reported['correct']}")
    if excluded.shape[0]:
        log.append(f"rows excluded from scoring: {excluded.shape[0]}")

    inputs = {
        "predictions_path": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "class_mapping_path": str(Path(args.class_mapping).resolve()),
        "class_mapping_sha256": sha256_file(Path(args.class_mapping)),
        "input_mode": "row_level_predictions",
    }
    if args.score_column in frame.columns:
        inputs["score_column"] = args.score_column
        inputs["score_semantics"] = "uncalibrated model score as supplied; not used by any metric"
    return {
        "reported": reported,
        "per_class_frame": per_class_frame,
        "excluded": excluded,
        "matrix": matrix,
        "mapping_frame": mapping_frame,
        "inputs": inputs,
    }


def run_from_matrix(args: argparse.Namespace, log: list[str]) -> dict:
    matrix_path = Path(args.confusion_matrix)
    classes, matrix = read_confusion_matrix(matrix_path)
    log.append(f"confusion matrix loaded: {matrix.shape[0]}x{matrix.shape[1]} from {matrix_path}")
    total = int(matrix.sum())
    if args.expected_rows is not None and total != args.expected_rows:
        raise EvaluationError(
            f"Row count mismatch: expected {args.expected_rows} rows, the matrix totals {total}. "
            "Nothing was written."
        )
    if args.class_mapping:
        declared = load_class_list(Path(args.class_mapping))
        if declared != classes:
            raise EvaluationError(
                "Declared class list does not match the confusion matrix class order; "
                "an order mismatch would silently permute the results"
            )
        log.append("declared class list matches the matrix order")
    reported = metrics_from_confusion_matrix(matrix, classes)
    normaliser = make_label_normaliser(args.label_normalisation)
    display_classes, mapping = resolve_display_classes(classes, normaliser, args.label_normalisation)
    reported["label_normalisation"] = args.label_normalisation
    reported["display_classes"] = display_classes
    reported["matrix_cross_check"] = {
        "performed": False,
        "status": "not_performed",
        "reason": (
            "aggregate input: there is no second representation to compare against. Recomputing the "
            "supplied matrix with the same formula is not an independent check. The independent check "
            "for this mode is arithmetic_check, which re-runs the arithmetic with exact rationals."
        ),
    }
    per_class_frame = pd.DataFrame(
        [
            {
                "class": name,
                "class_index": index,
                "display_class": display_classes[index],
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1"],
                "support": stats["support"],
                "predicted": stats["predicted"],
            }
            for index, (name, stats) in enumerate(reported["per_class"].items())
        ]
    )
    mapping_frame = pd.DataFrame(mapping, columns=["class_id", "raw_label", "display_label", "changed"])
    excluded = pd.DataFrame(columns=["row_position", "target", "prediction", "reason"])
    inputs = {
        "confusion_matrix_path": str(matrix_path.resolve()),
        "confusion_matrix_sha256": sha256_file(matrix_path),
        "input_mode": "aggregate_confusion_matrix",
    }
    return {
        "reported": reported,
        "per_class_frame": per_class_frame,
        "excluded": excluded,
        "matrix": matrix,
        "mapping_frame": mapping_frame,
        "inputs": inputs,
    }


def write_outputs(
    args: argparse.Namespace,
    output: Path,
    result: dict,
    classes: list[str],
    log: list[str],
) -> dict:
    reported = result["reported"]
    inputs = result["inputs"]
    description = DatasetDescription(task=args.task, split=args.split, source=args.source, notes=args.notes)

    payload = {
        "schema": "36127.unified_classification_evaluation/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task": description.task,
        "split": description.split,
        "source": description.source,
        "notes": description.notes,
        "classes": classes,
        "class_count": len(classes),
        "display_classes": reported["display_classes"],
        "label_normalisation": reported["label_normalisation"],
        "policy": describe_policy(),
        "inputs": inputs,
        "environment": environment_summary(device=args.device, seed=args.seed),
        "counts": {
            "rows_scored": reported["rows"],
            "correct": reported["correct"],
            "rows_excluded": int(result["excluded"].shape[0]),
        },
        "accuracy": reported["accuracy"],
        "macro_f1": reported["macro_f1"],
        "macro_f1_excluding_zero_support": reported["macro_f1_excluding_zero_support"],
        "macro_precision": reported["macro_precision"],
        "macro_recall": reported["macro_recall"],
        "weighted_f1": reported["weighted_f1"],
        "per_class": reported["per_class"],
        "classes_with_zero_support": reported["classes_with_zero_support"],
        "zero_support_class_count": reported["zero_support_class_count"],
        "classes_never_predicted": reported["classes_never_predicted"],
        "confusion_matrix": result["matrix"].tolist(),
        "matrix_cross_check": reported["matrix_cross_check"],
        "arithmetic_check": reported["arithmetic_check"],
        "code_hashes": {
            "evaluation.py": sha256_file(REPO_ROOT / "src/vehicle_id/attributes/evaluation.py"),
            "evaluate_classification.py": sha256_file(Path(__file__).resolve()),
        },
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "limitations": [
            "Metrics describe the supplied split only; they are not official test scores unless the split says so.",
            "Uncalibrated scores are recorded but never used by the metrics.",
            "Macro F1 treats every declared class equally, including classes with tiny support.",
        ],
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    log.append(f"wrote {metrics_path.name}")

    per_class_path = output / "per_class_metrics.csv"
    result["per_class_frame"][PER_CLASS_COLUMNS].to_csv(per_class_path, index=False)
    log.append(f"wrote {per_class_path.name}")

    mapping_path = output / "label_mapping.csv"
    result["mapping_frame"].to_csv(mapping_path, index=False)
    changed = int(result["mapping_frame"]["changed"].sum())
    log.append(f"wrote {mapping_path.name} ({changed} display label(s) differ from the raw label)")

    matrix_frame = pd.DataFrame(result["matrix"], index=classes, columns=classes)
    matrix_frame.index.name = "target"
    matrix_path = output / "confusion_matrix.csv"
    matrix_frame.to_csv(matrix_path)
    log.append(f"wrote {matrix_path.name}")

    if result["excluded"].shape[0]:
        excluded_path = output / "excluded_rows.csv"
        result["excluded"].to_csv(excluded_path, index=False)
        log.append(f"wrote {excluded_path.name} ({result['excluded'].shape[0]} rows)")

    environment_path = output / "environment.json"
    environment_path.write_text(
        json.dumps(payload["environment"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.append(f"wrote {environment_path.name}")

    (output / "README.md").write_text(_readme(args, payload, classes), encoding="utf-8")
    log.append("wrote README.md")
    return payload


def _readme(args: argparse.Namespace, payload: dict, classes: list[str]) -> str:
    return "\n".join(
        [
            f"# Unified classification evaluation - {args.task} / {args.split}",
            "",
            "Generated by `scripts/evaluate_classification.py` in "
            "`13_UNIFIED_CLASSIFICATION_EVAL_2026-09-16`.",
            "",
            "## Metric definitions",
            "",
            f"- Class list: {len(classes)} fixed classes in saved order.",
            f"- Label normalisation: `{args.label_normalisation}`. "
            "Numeric artifacts keep the raw class names; display names appear only in "
            "`per_class_metrics.csv` and `label_mapping.csv`.",
            f"- Zero-division handling: {payload['policy']['zero_division']}.",
            f"- Macro averaging: {payload['policy']['macro']}.",
            f"- Unusable rows: {payload['policy']['unusable_rows']}.",
            "",
            "## This run",
            "",
            f"- Task: {args.task}",
            f"- Split: {args.split}",
            f"- Source: {args.source or 'not recorded'}",
            f"- Rows scored: {payload['counts']['rows_scored']}",
            f"- Rows excluded: {payload['counts']['rows_excluded']}",
            f"- Accuracy: {payload['accuracy']!r}",
            f"- Macro F1: {payload['macro_f1']!r}",
            f"- Macro F1 excluding zero-support classes: {payload['macro_f1_excluding_zero_support']!r}",
            f"- Weighted F1: {payload['weighted_f1']!r}",
            f"- Seed: {payload['environment']['seed']}",
            f"- Device: {payload['environment']['device']}",
            "",
            "## Inputs",
            "",
            *[f"- `{key}`: `{value}`" for key, value in payload["inputs"].items()],
            "",
            "## Checks",
            "",
            f"- `matrix_cross_check`: {payload['matrix_cross_check'].get('status', 'performed')}",
            f"- `arithmetic_check`: {payload['arithmetic_check']['basis']}, "
            f"{payload['arithmetic_check']['compared_values']} values compared, "
            f"max difference {payload['arithmetic_check']['max_abs_difference']:.3e}, "
            f"passed={payload['arithmetic_check']['passed']}",
            "",
            "## Files",
            "",
            "- `metrics.json`: full result, policy, inputs, environment and code hashes.",
            "- `per_class_metrics.csv`: precision, recall, F1, support and predicted count per class.",
            "- `label_mapping.csv`: raw label, display label and class id for every declared class.",
            "- `confusion_matrix.csv`: rows are targets, columns are predictions, same class order.",
            "- `excluded_rows.csv`: only present when rows could not be scored; lists every one.",
            "- `environment.json`: interpreter, platform and package versions.",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in payload["limitations"]],
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log: list[str] = [f"command: {sys.argv}"]
    output = Path(args.output)
    owned_output: Path | None = None
    try:
        if (args.predictions is None) == (args.confusion_matrix is None):
            raise EvaluationError("Provide exactly one of --predictions or --confusion-matrix")
        if split_requires_confirmation(args.split) and not args.allow_official_test:
            raise EvaluationError(
                f"Split '{args.split}' looks like an official test set. "
                "Re-run with --allow-official-test only when that evaluation is intended and approved."
            )
        # Inputs are resolved and validated before any directory is created, so a
        # rejected run leaves nothing behind that could be mistaken for a result.
        result = (
            run_from_predictions(args, log)
            if args.predictions
            else run_from_matrix(args, log)
        )
        prepare_output(output, log)
        owned_output = output
        classes = list(result["reported"]["per_class"].keys())
        payload = write_outputs(args, output, result, classes, log)
    except UnusableRowsError as error:
        log.append(f"FAILED: {error}")
        _report_failure(error, log, owned_output, detail=error.detail)
        return 2
    except EvaluationError as error:
        log.append(f"FAILED: {error}")
        _report_failure(error, log, owned_output)
        return 2

    log.append(
        f"completed: rows={payload['counts']['rows_scored']} accuracy={payload['accuracy']!r} "
        f"macro_f1={payload['macro_f1']!r} weighted_f1={payload['weighted_f1']!r}"
    )
    _write_log(log, owned_output)
    print(
        json.dumps(
            {
                "status": "completed",
                "task": payload["task"],
                "split": payload["split"],
                "rows_scored": payload["counts"]["rows_scored"],
                "rows_excluded": payload["counts"]["rows_excluded"],
                "correct": payload["counts"]["correct"],
                "accuracy": payload["accuracy"],
                "macro_f1": payload["macro_f1"],
                "macro_f1_excluding_zero_support": payload["macro_f1_excluding_zero_support"],
                "weighted_f1": payload["weighted_f1"],
                "matrix_cross_check": payload["matrix_cross_check"],
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )
    return 0


def _write_log(log: list[str], directory: Path | None) -> None:
    """Write the run log, but only into a directory this run owns.

    ``directory`` is None whenever the run never took ownership, which is the
    case for every rejection. That keeps a refused run from touching a
    pre-existing output directory in any way, including its log file.
    """
    if directory is None:
        return
    header = [f"started: {datetime.now(timezone.utc).isoformat()}", f"argv: {sys.argv}", ""]
    (directory / "run_log.txt").write_text("\n".join(header + log) + "\n", encoding="utf-8")


def _report_failure(
    error: Exception,
    log: list[str],
    owned_output: Path | None,
    detail: dict | None = None,
) -> None:
    """Report a failure without ever touching a directory this run does not own.

    The structured message stays on stdout so callers can parse it, and a plain
    line goes to stderr as the human-readable failure record. When the run did
    take ownership before failing, the log is written into that new directory;
    otherwise no file anywhere is created or modified.
    """
    payload: dict = {"status": "failed", "error": str(error)}
    if detail is not None:
        payload["detail"] = detail
    _write_log(log, owned_output)
    print(json.dumps(payload, indent=2))
    print(f"error: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

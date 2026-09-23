"""Unified classification evaluation for the 36127 vehicle attribute tasks.

This module is the reusable evaluation entry point requested for the attribute
workstream. It defines ONE metric specification and applies it to every
attribute task so that colour, make and body-type numbers are comparable:

* a fixed, explicitly supplied class list with a saved order;
* integer-encoded labels, so the class list - not the data - decides which
  classes exist;
* ``zero_division=0`` for precision, recall and F1;
* macro F1 averaged over every declared class, including classes with zero
  validation support. The alternative value that drops zero-support classes is
  reported as well, so the two conventions can never be confused.

The definitions deliberately match ``vehicle_id.attributes.modelling.metrics``
(the implementation that produced the saved colour pilot results) so that a
recomputation reproduces the historical numbers exactly rather than merely
approximately.

Rows that cannot be scored against the declared class list never disappear
silently: without an explicit opt-in the evaluation aborts, and with the opt-in
every excluded row is written to ``excluded_rows.csv`` and counted in the JSON.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ZERO_DIVISION = 0
ZERO_DIVISION_POLICY = "zero_division=0 (undefined precision/recall/F1 become 0.0)"
MACRO_POLICY = (
    "primary macro F1 averages over every declared class, including classes with zero support; "
    "macro_f1_excluding_zero_support is reported separately and is not interchangeable"
)

#: Tokens that describe an assessment state rather than a real class label.
#: They must never be mixed into scored predictions without being declared.
RESERVED_LABEL_TOKENS = (
    "unknown",
    "not_assessed",
    "unassessed",
    "synthetic",
    "synthetic_smoke_only",
)

#: Split names that trigger the official-test confirmation guard. Comparison is
#: case-insensitive and ignores separators, so ``official_test``, ``Official-Test``
#: and ``officialtest`` all match. This is an explicit list rather than a substring
#: search, so unrelated names such as ``latest`` are deliberately not guarded.
SPLIT_NAMES_REQUIRING_CONFIRMATION = (
    "test",
    "tests",
    "testing",
    "officialtest",
    "testset",
    "testdata",
    "finaltest",
    "holdout",
    "holdouts",
    "holdoutset",
)


def normalise_split_name(name: str) -> str:
    """Lower-case a split name and drop separators, for guard comparison."""
    return "".join(character for character in str(name).lower() if character.isalnum())


def split_requires_confirmation(name: str) -> bool:
    """Report whether a split name is on the guarded list."""
    return normalise_split_name(name) in SPLIT_NAMES_REQUIRING_CONFIRMATION

#: Supported label-normalisation adapters. ``none`` is the default and compares
#: labels byte for byte, so no class can be merged by accident.
LABEL_NORMALISATIONS = ("none", "strip")

LABEL_NORMALISATION_POLICY = (
    "raw labels are compared unchanged unless an adapter is requested explicitly; "
    "the same adapter is applied to the class list, the target column and the prediction "
    "column, and any many-to-one collapse aborts the run instead of merging classes"
)


class EvaluationError(ValueError):
    """Raised when the evaluation input is unusable or ambiguous."""


class UnusableRowsError(EvaluationError):
    """Raised when rows cannot be scored and the caller did not opt in."""

    def __init__(self, message: str, detail: dict[str, Any]):
        super().__init__(message)
        self.detail = detail


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_class_list(path: Path) -> list[str]:
    """Load a class list from JSON, preserving the declared order.

    Accepts either a JSON list of class names or the project's
    ``class_to_idx.json`` mapping form. The order is preserved exactly.
    """
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"Class mapping file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationError(f"Class mapping is not valid JSON: {path} ({error})") from error

    if isinstance(payload, dict):
        if not payload:
            raise EvaluationError("Class mapping is empty")
        indices: dict[str, int] = {}
        for name, index in payload.items():
            if isinstance(index, (bool, np.bool_)):
                raise EvaluationError(f"Class index for '{name}' is a boolean; indices must be JSON integers")
            if not isinstance(index, (int, np.integer)):
                raise EvaluationError(
                    f"Class index for '{name}' is {type(index).__name__} ({index!r}); indices must be "
                    "JSON integers. Fractional or textual indices would be converted silently, so they "
                    "are rejected instead."
                )
            indices[name] = int(index)
        if sorted(indices.values()) != list(range(len(indices))):
            raise EvaluationError("Class mapping indices must be exactly 0..n-1 without gaps or repeats")
        classes = [name for name, _ in sorted(indices.items(), key=lambda item: item[1])]
    elif isinstance(payload, list):
        classes = list(payload)
    else:
        raise EvaluationError("Class mapping must be a JSON list or an object mapping names to indices")

    if not classes:
        raise EvaluationError("Class mapping is empty")
    if not all(isinstance(name, str) and name.strip() for name in classes):
        raise EvaluationError("Class names must be non-empty strings")
    if len(set(classes)) != len(classes):
        duplicates = sorted({name for name in classes if classes.count(name) > 1})
        raise EvaluationError(f"Class mapping contains duplicate names: {duplicates}")
    return classes


INT64_MAX = int(np.iinfo(np.int64).max)


def _validated_count(element: Any, label: str, position: tuple[int, int]) -> int:
    """Return one count as an int, refusing anything that would need truncation."""
    if isinstance(element, (bool, np.bool_)):
        raise EvaluationError(f"{label} holds a boolean ({element!r}) at {position}; counts must be integers")
    if isinstance(element, (int, np.integer)):
        value = int(element)
    elif isinstance(element, (float, np.floating)):
        if not np.isfinite(element):
            raise EvaluationError(f"{label} holds a non-finite value ({element!r}) at {position}")
        if float(element) != int(element):
            raise EvaluationError(
                f"{label} holds the fractional count {element!r} at {position}. "
                "Fractional counts are rejected rather than truncated."
            )
        value = int(element)
    else:
        raise EvaluationError(
            f"{label} holds the non-numeric value {element!r} at {position}; counts must be integers"
        )
    if value < 0:
        raise EvaluationError(f"{label} holds the negative count {value} at {position}")
    if value > INT64_MAX:
        raise EvaluationError(f"{label} holds the count {value} at {position}, which overflows int64")
    return value


def validate_confusion_counts(raw: Any, label: str = "confusion matrix") -> np.ndarray:
    """Return an ``int64`` count matrix, rejecting every lossy conversion.

    Integral floats such as ``1.0`` are accepted and recorded here: they are
    exact, and CSV readers routinely hand back a float column for an all-integer
    table. Anything that would be changed by the conversion - fractional values,
    negatives, non-finite values, booleans, text and out-of-range numbers - is
    refused instead of being rounded, truncated or sign-coerced.

    The raw elements are inspected before any numeric dtype is chosen. Building
    an integer array first would turn a stray ``True`` in a mixed list into the
    count ``1`` and hide the mistake; ``np.asarray(..., dtype=object)`` keeps the
    original Python and numpy scalar types so each one can be judged on its own.
    """
    try:
        array = np.asarray(raw, dtype=object)
    except (ValueError, TypeError) as error:
        raise EvaluationError(f"{label} is not a rectangular numeric table: {error}") from error

    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise EvaluationError(f"{label} must be a non-empty two-dimensional table, got shape {array.shape}")

    counts = np.empty(array.shape, dtype=np.int64)
    for position, element in np.ndenumerate(array):
        counts[position] = _validated_count(element, label, position)
    return counts


def prepare_confusion_matrix(raw: Any, classes: Sequence[str], label: str = "confusion matrix") -> np.ndarray:
    """Validate counts, shape and class count together.

    Every public entry point goes through here, so calling
    :func:`metrics_from_confusion_matrix` directly cannot bypass validation.
    """
    matrix = validate_confusion_counts(raw, label)
    if matrix.shape[0] != matrix.shape[1]:
        raise EvaluationError(f"{label} must be square, got shape {matrix.shape}")
    if matrix.shape[0] != len(classes):
        raise EvaluationError(
            f"{label} is {matrix.shape[0]}x{matrix.shape[1]} but {len(classes)} classes were declared"
        )
    if int(matrix.sum()) == 0:
        raise EvaluationError(f"{label} is empty (all counts are zero)")
    return matrix


def read_confusion_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a published confusion matrix in the project's JSON or CSV layout."""
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"Confusion matrix file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "confusion_matrix" not in payload or "classes" not in payload:
            raise EvaluationError("JSON confusion matrix input needs 'classes' and 'confusion_matrix' keys")
        classes = [str(name) for name in payload["classes"]]
        raw = payload["confusion_matrix"]
    else:
        frame = pd.read_csv(path, index_col=0)
        classes = [str(name) for name in frame.index]
        if [str(name) for name in frame.columns] != classes:
            raise EvaluationError("CSV confusion matrix must have identical row and column class order")
        raw = frame.to_numpy()

    return classes, prepare_confusion_matrix(raw, classes)


def exact_metrics_from_confusion_matrix(matrix: np.ndarray) -> dict[str, Any]:
    """Recompute the metrics with exact rational arithmetic.

    This is a deliberately separate implementation: it uses
    :class:`fractions.Fraction` and Python integers only, so it shares no code
    with the numpy path and acts as an independent arithmetic check.
    """
    size = matrix.shape[0]
    counts = [[int(matrix[row][column]) for column in range(size)] for row in range(size)]
    support = [sum(row) for row in counts]
    predicted = [sum(counts[row][column] for row in range(size)) for column in range(size)]
    true_positive = [counts[index][index] for index in range(size)]
    total = sum(support)

    def ratio(numerator: int, denominator: int) -> Fraction:
        return Fraction(numerator, denominator) if denominator else Fraction(0, 1)

    precision = [ratio(true_positive[index], predicted[index]) for index in range(size)]
    recall = [ratio(true_positive[index], support[index]) for index in range(size)]
    f1 = [
        ratio(2 * precision[index] * recall[index], precision[index] + recall[index])
        if (precision[index] + recall[index])
        else Fraction(0, 1)
        for index in range(size)
    ]
    weighted = sum((f1[index] * support[index] for index in range(size)), Fraction(0, 1))
    return {
        "accuracy": ratio(sum(true_positive), total),
        "macro_f1": sum(f1, Fraction(0, 1)) / size,
        "macro_precision": sum(precision, Fraction(0, 1)) / size,
        "macro_recall": sum(recall, Fraction(0, 1)) / size,
        "weighted_f1": weighted / total if total else Fraction(0, 1),
        "per_class_f1": f1,
        "per_class_precision": precision,
        "per_class_recall": recall,
    }


def metrics_from_confusion_matrix(matrix: Any, classes: Sequence[str]) -> dict[str, Any]:
    """Compute the shared metric specification from a confusion matrix.

    A confusion matrix is a sufficient statistic for these metrics, so this path
    returns exactly the values the row-level path returns. The counts are
    validated here as well, so this public entry point cannot be used to bypass
    the checks that :func:`read_confusion_matrix` performs.
    """
    matrix = prepare_confusion_matrix(matrix, classes).astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(true_positive), where=denominator > 0)

    total = float(matrix.sum())
    accuracy = float(true_positive.sum() / total)
    weights = support / total
    zero_support = support == 0
    macro_f1_excluding = float(f1[~zero_support].mean()) if (~zero_support).any() else None
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "predicted": int(predicted[index]),
        }
        for index, name in enumerate(classes)
    }
    zero_support = [name for name, stats in per_class.items() if stats["support"] == 0]
    reported = {
        "rows": int(total),
        "correct": int(true_positive.sum()),
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "macro_f1_excluding_zero_support": macro_f1_excluding,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "weighted_f1": float((f1 * weights).sum()),
        "per_class": per_class,
        "classes_with_zero_support": zero_support,
        "zero_support_class_count": len(zero_support),
        "classes_never_predicted": [
            name for name, stats in per_class.items() if stats["predicted"] == 0
        ],
    }
    reported["arithmetic_check"] = _arithmetic_check(reported, matrix, classes)
    return reported


def _arithmetic_check(reported: dict[str, Any], matrix: np.ndarray, classes: Sequence[str]) -> dict[str, Any]:
    """Compare the float results against an exact rational recomputation."""
    exact = exact_metrics_from_confusion_matrix(matrix)
    pairs = [
        ("accuracy", reported["accuracy"], exact["accuracy"]),
        ("macro_f1", reported["macro_f1"], exact["macro_f1"]),
        ("macro_precision", reported["macro_precision"], exact["macro_precision"]),
        ("macro_recall", reported["macro_recall"], exact["macro_recall"]),
        ("weighted_f1", reported["weighted_f1"], exact["weighted_f1"]),
    ]
    for index, name in enumerate(classes):
        pairs.append((f"per_class.{name}.precision", reported["per_class"][name]["precision"], exact["per_class_precision"][index]))
        pairs.append((f"per_class.{name}.recall", reported["per_class"][name]["recall"], exact["per_class_recall"][index]))
        pairs.append((f"per_class.{name}.f1", reported["per_class"][name]["f1"], exact["per_class_f1"][index]))
    differences = {name: abs(float(value) - float(exact_value)) for name, value, exact_value in pairs}
    worst = max(differences.values()) if differences else 0.0
    return {
        "basis": "exact rational arithmetic (fractions.Fraction), independent of the numpy path",
        "compared_values": len(pairs),
        "max_abs_difference": worst,
        "tolerance": 1e-12,
        "passed": worst <= 1e-12,
        "worst_key": max(differences, key=differences.get) if differences else None,
    }


@dataclass
class DatasetDescription:
    """Descriptive metadata recorded alongside the metrics."""

    task: str
    split: str
    source: str | None = None
    notes: str | None = None


def _raw_label(value: Any) -> str | None:
    """Return the label exactly as stored, or None when the value is missing.

    Deliberately does not strip whitespace. Stripping here would silently turn
    ``"Hyundai "`` into ``"Hyundai"`` and merge or split classes without anyone
    asking for it; normalisation is an explicit, announced adapter instead.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value)
    if text == "" or text.strip().lower() in {"nan", "<na>", "null", "na", "n/a"}:
        return None
    return text


def make_label_normaliser(name: str):
    """Return the requested label adapter, or raise for an unknown name."""
    if name == "none":
        return lambda value: value
    if name == "strip":
        return lambda value: value.strip()
    raise EvaluationError(
        f"Unsupported label normalisation '{name}'; supported values are {list(LABEL_NORMALISATIONS)}"
    )


def resolve_display_classes(
    classes: Sequence[str],
    normaliser,
    mode: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Map raw class names to display names, refusing any many-to-one collapse.

    Returns the display names in class order plus one mapping record per class.
    The class index never changes, so downstream metrics stay on the original
    mapping.
    """
    display = [normaliser(name) for name in classes]
    buckets: dict[str, list[int]] = {}
    for index, name in enumerate(display):
        buckets.setdefault(name, []).append(index)
    collisions = {name: indices for name, indices in buckets.items() if len(indices) > 1}
    if collisions:
        detail = {name: [classes[index] for index in indices] for name, indices in collisions.items()}
        raise EvaluationError(
            f"Label normalisation '{mode}' would merge distinct classes: {detail}. "
            "Resolve the class list instead of merging, dropping rows or changing the class count."
        )
    if len(set(display)) != len(classes):
        raise EvaluationError(f"Label normalisation '{mode}' changed the class count")
    mapping = [
        {
            "class_id": index,
            "raw_label": classes[index],
            "display_label": display[index],
            "changed": display[index] != classes[index],
        }
        for index in range(len(classes))
    ]
    return display, mapping


def check_observed_label_collisions(
    frame: pd.DataFrame,
    classes: Sequence[str],
    columns: Sequence[str],
    normaliser,
    mode: str,
) -> None:
    """Refuse a data-side many-to-one collapse.

    Two different raw spellings that normalise to the same class must be a
    human decision, not a silent merge, even when the class list itself is
    collision free.
    """
    if mode == "none":
        return
    display = {normaliser(name) for name in classes}
    for column in columns:
        grouped: dict[str, set[str]] = {}
        for value in frame[column].tolist():
            raw = _raw_label(value)
            if raw is None:
                continue
            grouped.setdefault(normaliser(raw), set()).add(raw)
        ambiguous = {
            target: sorted(raws)
            for target, raws in grouped.items()
            if len(raws) > 1 and target in display
        }
        if ambiguous:
            raise EvaluationError(
                f"Column '{column}' contains several raw spellings that normalise to the same "
                f"declared class under '{mode}': {ambiguous}. Fix the labels instead of letting "
                "them merge silently."
            )


def classify_rows(
    frame: pd.DataFrame,
    classes: Sequence[str],
    *,
    target_column: str,
    prediction_column: str,
    normaliser=None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Split rows into scorable rows and rows that cannot be scored.

    Returns integer label arrays for the scorable rows plus a frame describing
    every unusable row. It never drops a row without describing it.
    """
    normaliser = normaliser or (lambda value: value)
    class_index = {normaliser(name): index for index, name in enumerate(classes)}
    reserved = {token.lower() for token in RESERVED_LABEL_TOKENS}

    target_index = np.full(len(frame), -1, dtype=np.int64)
    prediction_index = np.full(len(frame), -1, dtype=np.int64)
    problems: list[dict[str, Any]] = []

    for position, (target, prediction) in enumerate(
        zip(frame[target_column].tolist(), frame[prediction_column].tolist())
    ):
        target_text = _raw_label(target)
        prediction_text = _raw_label(prediction)
        target_key = normaliser(target_text) if target_text is not None else None
        prediction_key = normaliser(prediction_text) if prediction_text is not None else None
        reasons: list[str] = []

        if target_text is None:
            reasons.append("missing target label")
        elif target_text.lower() in reserved:
            reasons.append(f"reserved target token '{target_text}'")
        elif target_key not in class_index:
            reasons.append(f"target label '{target_text}' outside the declared class list")
        else:
            target_index[position] = class_index[target_key]

        if prediction_text is None:
            reasons.append("missing prediction label")
        elif prediction_text.lower() in reserved:
            reasons.append(f"reserved prediction token '{prediction_text}'")
        elif prediction_key not in class_index:
            reasons.append(f"prediction label '{prediction_text}' outside the declared class list")
        else:
            prediction_index[position] = class_index[prediction_key]

        if reasons:
            problems.append(
                {
                    "row_position": position,
                    "target": target_text,
                    "prediction": prediction_text,
                    "reason": "; ".join(reasons),
                }
            )

    problem_index = {entry["row_position"] for entry in problems}
    keep = np.array([position not in problem_index for position in range(len(frame))], dtype=bool)
    excluded = pd.DataFrame(problems, columns=["row_position", "target", "prediction", "reason"])
    return target_index[keep], prediction_index[keep], excluded


def evaluate_rows(
    frame: pd.DataFrame,
    classes: Sequence[str],
    *,
    target_column: str,
    prediction_column: str,
    allow_unusable_rows: bool,
    id_column: str | None = None,
    expected_rows: int | None = None,
    label_normalisation: str = "none",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Validate a prediction table and compute the shared metric specification.

    Returns the metrics, the per-class table, the excluded-row table, the
    confusion matrix over the declared class order, and the raw-to-display
    label mapping. Metrics always use the original class mapping; the display
    names only affect presentation.
    """
    if not classes:
        raise EvaluationError("The declared class list is empty")
    if frame.empty:
        raise EvaluationError("The prediction table is empty; nothing can be evaluated")
    for column in (target_column, prediction_column):
        if column not in frame.columns:
            raise EvaluationError(
                f"Column '{column}' is missing. Available columns: {list(frame.columns)}"
            )
    if id_column is not None:
        if id_column not in frame.columns:
            raise EvaluationError(
                f"Identifier column '{id_column}' is missing. Available columns: {list(frame.columns)}"
            )
        duplicated = frame[id_column][frame[id_column].duplicated()].unique().tolist()
        if duplicated:
            shown = duplicated[:10]
            raise EvaluationError(
                f"Duplicate sample identifiers in '{id_column}': {len(duplicated)} value(s), e.g. {shown}. "
                "Row counts would not be trustworthy; fix the input instead of dropping rows."
            )
    if expected_rows is not None and len(frame) != expected_rows:
        raise EvaluationError(
            f"Row count mismatch: expected {expected_rows} rows, found {len(frame)}"
        )

    normaliser = make_label_normaliser(label_normalisation)
    display_classes, label_mapping = resolve_display_classes(classes, normaliser, label_normalisation)
    check_observed_label_collisions(
        frame, classes, (target_column, prediction_column), normaliser, label_normalisation
    )

    target_index, prediction_index, excluded = classify_rows(
        frame,
        classes,
        target_column=target_column,
        prediction_column=prediction_column,
        normaliser=normaliser,
    )
    if excluded.shape[0]:
        if not allow_unusable_rows:
            reasons = excluded["reason"].value_counts().to_dict()
            raise UnusableRowsError(
                f"{excluded.shape[0]} of {len(frame)} rows cannot be scored against the declared class list "
                f"({reasons}). Re-run with --allow-unknown-labels to exclude and record them explicitly.",
                {"excluded_rows": int(excluded.shape[0]), "reasons": reasons},
            )
    if target_index.size == 0:
        raise EvaluationError("Every row was excluded; there is nothing left to evaluate")

    labels = list(range(len(classes)))
    matrix = confusion_matrix(target_index, prediction_index, labels=labels)
    per_class_f1 = f1_score(
        target_index, prediction_index, labels=labels, average=None, zero_division=ZERO_DIVISION
    )
    per_class_precision = precision_score(
        target_index, prediction_index, labels=labels, average=None, zero_division=ZERO_DIVISION
    )
    per_class_recall = recall_score(
        target_index, prediction_index, labels=labels, average=None, zero_division=ZERO_DIVISION
    )
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    reported: dict[str, Any] = {
        "rows": int(target_index.size),
        "correct": int((target_index == prediction_index).sum()),
        "accuracy": float(accuracy_score(target_index, prediction_index)),
        "macro_f1": float(
            f1_score(target_index, prediction_index, labels=labels, average="macro", zero_division=ZERO_DIVISION)
        ),
        "macro_precision": float(
            precision_score(target_index, prediction_index, labels=labels, average="macro", zero_division=ZERO_DIVISION)
        ),
        "macro_recall": float(
            recall_score(target_index, prediction_index, labels=labels, average="macro", zero_division=ZERO_DIVISION)
        ),
        "weighted_f1": float(
            f1_score(target_index, prediction_index, labels=labels, average="weighted", zero_division=ZERO_DIVISION)
        ),
        "per_class": {
            name: {
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
                "predicted": int(predicted[index]),
            }
            for index, name in enumerate(classes)
        },
    }

    # Independent cross-check: a confusion matrix is a sufficient statistic for
    # these metrics, so the closed-form reconstruction must agree with sklearn.
    from_matrix = metrics_from_confusion_matrix(matrix, classes)
    shared = ("accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1")
    differences = [abs(from_matrix[key] - reported[key]) for key in shared]
    differences += [
        abs(from_matrix["per_class"][name][key] - reported["per_class"][name][key])
        for name in classes
        for key in ("precision", "recall", "f1")
    ]
    reported["matrix_cross_check"] = {
        "performed": True,
        "basis": (
            "two representations of the same rows: sklearn scores computed from the row-level "
            "labels versus the closed-form reconstruction from the aggregated confusion matrix"
        ),
        "max_abs_difference": max(differences),
        "tolerance": 1e-12,
        "passed": max(differences) <= 1e-12,
    }
    reported["arithmetic_check"] = from_matrix["arithmetic_check"]
    reported["macro_f1_excluding_zero_support"] = from_matrix["macro_f1_excluding_zero_support"]
    reported["classes_with_zero_support"] = [
        name for name, stats in reported["per_class"].items() if stats["support"] == 0
    ]
    reported["classes_never_predicted"] = [
        name for name, stats in reported["per_class"].items() if stats["predicted"] == 0
    ]
    reported["zero_support_class_count"] = len(reported["classes_with_zero_support"])
    reported["label_normalisation"] = label_normalisation
    reported["display_classes"] = display_classes

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
    mapping_frame = pd.DataFrame(label_mapping, columns=["class_id", "raw_label", "display_label", "changed"])
    return reported, per_class_frame, excluded, matrix, mapping_frame


def environment_summary(*, device: str, seed: Any) -> dict[str, Any]:
    """Record interpreter, platform and package versions for the run."""
    import importlib.metadata

    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": device,
        "cuda_available": _cuda_available(),
        "seed": seed,
        "packages": packages,
    }


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())

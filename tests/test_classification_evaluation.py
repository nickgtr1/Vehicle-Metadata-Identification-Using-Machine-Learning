"""Tests for the unified classification evaluation entry point.

Hand-worked cases come first so that the metric convention is pinned down by
arithmetic rather than by the saved experiment files. The colour reproduction
test then checks the tool against the historical pilot result.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vehicle_id.attributes.evaluation import (  # noqa: E402
    EvaluationError,
    UnusableRowsError,
    evaluate_rows,
    exact_metrics_from_confusion_matrix,
    load_class_list,
    make_label_normaliser,
    metrics_from_confusion_matrix,
    prepare_confusion_matrix,
    read_confusion_matrix,
    resolve_display_classes,
    split_requires_confirmation,
    validate_confusion_counts,
)

PROJECT_ROOT = REPO.parent
COLOUR_DIR = PROJECT_ROOT / "outputs" / "2026-09-11_colour_transfer"
R1_ROOT = PROJECT_ROOT / "outputs" / "2026-09-16_unified_classification_f1_eval_r1"
PR5_AGGREGATE = R1_ROOT / "inputs" / "pr5_aggregate"


def frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"sample_id": f"s{index}", "target": target, "prediction": prediction} for index, (target, prediction) in enumerate(rows)]
    )


def measure(rows: list[tuple[str, str]], classes: list[str], **kwargs):
    return evaluate_rows(
        frame(rows),
        classes,
        target_column="target",
        prediction_column="prediction",
        allow_unusable_rows=kwargs.pop("allow_unusable_rows", False),
        id_column=kwargs.pop("id_column", "sample_id"),
        **kwargs,
    )


class HandWorkedMetricTests(unittest.TestCase):
    def test_all_correct(self):
        reported, per_class, excluded, matrix, mapping = measure([("a", "a"), ("b", "b")], ["a", "b"])
        self.assertEqual(list(mapping["display_label"]), ["a", "b"])
        self.assertFalse(bool(mapping["changed"].any()))
        self.assertEqual(reported["accuracy"], 1.0)
        self.assertEqual(reported["macro_f1"], 1.0)
        self.assertEqual(reported["weighted_f1"], 1.0)
        self.assertEqual(reported["rows"], 2)
        self.assertEqual(excluded.shape[0], 0)
        self.assertEqual(matrix.tolist(), [[1, 0], [0, 1]])
        self.assertTrue(per_class[["precision", "recall", "f1"]].eq(1.0).all().all())

    def test_all_wrong(self):
        reported, _, _, matrix, _ = measure([("a", "b"), ("b", "a")], ["a", "b"])
        self.assertEqual(reported["accuracy"], 0.0)
        self.assertEqual(reported["macro_f1"], 0.0)
        self.assertEqual(matrix.tolist(), [[0, 1], [1, 0]])

    def test_class_never_predicted_keeps_zero_precision(self):
        reported, per_class, _, _, _ = measure(
            [("a", "a"), ("a", "a"), ("b", "b"), ("c", "a")], ["a", "b", "c"]
        )
        stats = reported["per_class"]
        self.assertAlmostEqual(stats["a"]["precision"], 2 / 3)
        self.assertEqual(stats["a"]["recall"], 1.0)
        self.assertAlmostEqual(stats["a"]["f1"], 0.8)
        self.assertEqual(stats["b"]["f1"], 1.0)
        self.assertEqual(stats["c"]["precision"], 0.0)
        self.assertEqual(stats["c"]["recall"], 0.0)
        self.assertEqual(stats["c"]["f1"], 0.0)
        self.assertAlmostEqual(reported["macro_f1"], (0.8 + 1.0 + 0.0) / 3)
        self.assertAlmostEqual(reported["accuracy"], 0.75)
        self.assertEqual(list(per_class["class"]), ["a", "b", "c"])

    def test_missing_support_class_still_counts_in_macro(self):
        reported, _, _, _, _ = measure([("a", "a"), ("a", "b"), ("b", "b")], ["a", "b", "c"])
        expected_pair = 2 * (1.0 * 0.5) / 1.5
        self.assertAlmostEqual(reported["per_class"]["a"]["f1"], expected_pair)
        self.assertAlmostEqual(reported["per_class"]["b"]["f1"], expected_pair)
        self.assertEqual(reported["per_class"]["c"]["support"], 0)
        self.assertEqual(reported["per_class"]["c"]["f1"], 0.0)
        self.assertAlmostEqual(reported["macro_f1"], (expected_pair + expected_pair + 0.0) / 3)
        self.assertAlmostEqual(reported["macro_f1_excluding_zero_support"], expected_pair)
        self.assertEqual(reported["classes_with_zero_support"], ["c"])

    def test_support_is_the_row_sum_of_the_matrix(self):
        reported, _, _, matrix, _ = measure([("a", "a"), ("a", "b"), ("b", "b")], ["a", "b", "c"])
        supports = [reported["per_class"][name]["support"] for name in ("a", "b", "c")]
        self.assertEqual(supports, matrix.sum(axis=1).tolist())
        self.assertEqual(sum(supports), reported["rows"])

    def test_single_class_never_predicted_gives_zero_f1(self):
        reported, _, _, _, _ = measure([("a", "a"), ("b", "a")], ["a", "b"])
        # a: precision 1/2, recall 1/1, f1 2*(0.5*1)/1.5 = 2/3
        # b: never predicted, so precision 0 by zero_division=0 and f1 0
        self.assertAlmostEqual(reported["per_class"]["a"]["precision"], 0.5)
        self.assertAlmostEqual(reported["per_class"]["a"]["f1"], 2 / 3)
        self.assertEqual(reported["per_class"]["b"]["predicted"], 0)
        self.assertEqual(reported["per_class"]["b"]["f1"], 0.0)
        self.assertAlmostEqual(reported["macro_f1"], (2 / 3 + 0.0) / 2)
        self.assertEqual(reported["classes_never_predicted"], ["b"])


class InvalidInputTests(unittest.TestCase):
    def test_empty_input_is_rejected(self):
        empty = pd.DataFrame(columns=["sample_id", "target", "prediction"])
        with self.assertRaises(EvaluationError):
            evaluate_rows(
                empty,
                ["a", "b"],
                target_column="target",
                prediction_column="prediction",
                allow_unusable_rows=True,
            )

    def test_unknown_label_aborts_by_default(self):
        with self.assertRaises(UnusableRowsError) as context:
            measure([("a", "a"), ("z", "a")], ["a", "b"])
        self.assertEqual(context.exception.detail["excluded_rows"], 1)

    def test_unknown_label_is_recorded_when_allowed(self):
        reported, _, excluded, _, _ = measure(
            [("a", "a"), ("z", "a"), ("b", "b")], ["a", "b"], allow_unusable_rows=True
        )
        self.assertEqual(reported["rows"], 2)
        self.assertEqual(excluded.shape[0], 1)
        self.assertEqual(excluded.loc[0, "target"], "z")
        self.assertIn("outside the declared class list", excluded.loc[0, "reason"])

    def test_reserved_tokens_are_not_silently_scored(self):
        with self.assertRaises(UnusableRowsError):
            measure([("a", "a"), ("a", "not_assessed")], ["a", "b"])
        reported, _, excluded, _, _ = measure(
            [("a", "a"), ("a", "not_assessed")], ["a", "b"], allow_unusable_rows=True
        )
        self.assertEqual(reported["rows"], 1)
        self.assertIn("reserved prediction token", excluded.loc[0, "reason"])

    def test_missing_value_aborts_by_default(self):
        broken = pd.DataFrame({"sample_id": ["s0", "s1"], "target": ["a", None], "prediction": ["a", "a"]})
        with self.assertRaises(UnusableRowsError):
            evaluate_rows(
                broken,
                ["a", "b"],
                target_column="target",
                prediction_column="prediction",
                allow_unusable_rows=False,
            )

    def test_duplicate_identifiers_are_rejected(self):
        duplicated = pd.DataFrame(
            {
                "sample_id": ["s0", "s0"],
                "target": ["a", "b"],
                "prediction": ["a", "b"],
            }
        )
        with self.assertRaises(EvaluationError) as context:
            evaluate_rows(
                duplicated,
                ["a", "b"],
                target_column="target",
                prediction_column="prediction",
                allow_unusable_rows=False,
                id_column="sample_id",
            )
        self.assertIn("Duplicate sample identifiers", str(context.exception))

    def test_row_count_mismatch_is_rejected(self):
        with self.assertRaises(EvaluationError) as context:
            measure([("a", "a"), ("b", "b")], ["a", "b"], expected_rows=3)
        self.assertIn("Row count mismatch", str(context.exception))

    def test_missing_column_is_rejected(self):
        broken = pd.DataFrame({"sample_id": ["s0"], "target": ["a"]})
        with self.assertRaises(EvaluationError):
            evaluate_rows(
                broken,
                ["a", "b"],
                target_column="target",
                prediction_column="prediction",
                allow_unusable_rows=False,
            )


class ClassMappingTests(unittest.TestCase):
    def test_mapping_form_must_not_have_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "class_to_idx.json"
            path.write_text(json.dumps({"a": 0, "b": 2}), encoding="utf-8")
            with self.assertRaises(EvaluationError):
                load_class_list(path)

    def test_mapping_form_preserves_index_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "classes.json"
            path.write_text(json.dumps({"z": 1, "a": 0}), encoding="utf-8")
            self.assertEqual(load_class_list(path), ["a", "z"])

    def test_list_form_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "classes.json"
            path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
            self.assertEqual(load_class_list(path), ["a", "b"])

    def test_duplicate_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "classes.json"
            path.write_text(json.dumps(["a", "a"]), encoding="utf-8")
            with self.assertRaises(EvaluationError):
                load_class_list(path)


class LabelNormalisationTests(unittest.TestCase):
    """The raw class mapping is authoritative; display trimming never changes metrics."""

    CLASSES = ["Hyundai ", "Lamorghini ", "Honda"]
    ROWS = [("Hyundai ", "Hyundai "), ("Lamorghini ", "Hyundai "), ("Honda", "Honda")]

    def test_default_mode_keeps_raw_labels(self):
        reported, _, _, _, mapping = measure(self.ROWS, self.CLASSES)
        self.assertEqual(list(mapping["raw_label"]), self.CLASSES)
        self.assertEqual(list(mapping["display_label"]), self.CLASSES)
        self.assertFalse(bool(mapping["changed"].any()))
        self.assertEqual(reported["label_normalisation"], "none")

    def test_strip_adapter_reports_display_names_without_changing_metrics(self):
        raw, _, _, _, raw_mapping = measure(self.ROWS, self.CLASSES)
        stripped, _, _, _, mapping = measure(self.ROWS, self.CLASSES, label_normalisation="strip")
        for key in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"):
            self.assertAlmostEqual(stripped[key], raw[key], places=15)
        for name in self.CLASSES:
            self.assertAlmostEqual(
                stripped["per_class"][name]["f1"], raw["per_class"][name]["f1"], places=15
            )
        self.assertEqual(list(mapping["display_label"]), ["Hyundai", "Lamorghini", "Honda"])
        self.assertEqual(list(mapping["raw_label"]), self.CLASSES)
        self.assertEqual(int(mapping["changed"].sum()), 2)
        self.assertEqual(stripped["display_classes"], ["Hyundai", "Lamorghini", "Honda"])
        self.assertEqual(stripped["classes_with_zero_support"], [])
        self.assertEqual(stripped["zero_support_class_count"], 0)
        self.assertFalse(bool(raw_mapping["changed"].any()))

    def test_one_sided_rename_is_rejected_by_default(self):
        with self.assertRaises(UnusableRowsError) as context:
            measure([("Hyundai", "Hyundai")], self.CLASSES)
        self.assertEqual(context.exception.detail["excluded_rows"], 1)

    def test_class_list_collision_is_rejected(self):
        with self.assertRaises(EvaluationError) as context:
            measure([("Hyundai ", "Hyundai ")], ["Hyundai ", "Hyundai"], label_normalisation="strip")
        self.assertIn("merge distinct classes", str(context.exception))

    def test_observed_label_collision_is_rejected(self):
        rows = [("Hyundai ", "Hyundai "), ("Hyundai", "Hyundai ")]
        with self.assertRaises(EvaluationError) as context:
            measure(rows, ["Hyundai "], label_normalisation="strip")
        self.assertIn("several raw spellings", str(context.exception))

    def test_unknown_adapter_name_is_rejected(self):
        with self.assertRaises(EvaluationError):
            measure(self.ROWS, self.CLASSES, label_normalisation="lowercase")

    def test_display_mapping_records_the_class_id(self):
        display, mapping = resolve_display_classes(self.CLASSES, make_label_normaliser("strip"), "strip")
        self.assertEqual(display, ["Hyundai", "Lamorghini", "Honda"])
        self.assertEqual([record["class_id"] for record in mapping], [0, 1, 2])


class ZeroSupportReportingTests(unittest.TestCase):
    def test_zero_support_count_is_explicit(self):
        reported, _, _, _, _ = measure([("a", "a"), ("b", "b")], ["a", "b", "c"])
        self.assertEqual(reported["classes_with_zero_support"], ["c"])
        self.assertEqual(reported["zero_support_class_count"], 1)
        self.assertAlmostEqual(reported["macro_f1"], (1.0 + 1.0 + 0.0) / 3)
        self.assertAlmostEqual(reported["macro_f1_excluding_zero_support"], 1.0)

    def test_no_zero_support_classes_is_reported_as_zero(self):
        reported, _, _, _, _ = measure([("a", "a"), ("b", "b")], ["a", "b"])
        self.assertEqual(reported["zero_support_class_count"], 0)
        self.assertEqual(reported["classes_with_zero_support"], [])


class MakeLabelAdapterIntegrationTests(unittest.TestCase):
    """Run the real PR #5 make class list through the display adapter."""

    @classmethod
    def setUpClass(cls):
        matrix_path = PR5_AGGREGATE / "make_validation_metrics.json"
        mapping_path = PR5_AGGREGATE / "make_class_to_idx.json"
        if not matrix_path.is_file() or not mapping_path.is_file():
            raise unittest.SkipTest(f"PR #5 aggregate inputs not available at {PR5_AGGREGATE}")
        cls.classes, cls.matrix = read_confusion_matrix(matrix_path)
        declared = load_class_list(mapping_path)
        if declared != cls.classes:
            raise AssertionError("aggregate class list and matrix order disagree")

    def test_strip_is_bijective_over_the_real_class_list(self):
        display, mapping = resolve_display_classes(self.classes, make_label_normaliser("strip"), "strip")
        self.assertEqual(len(set(display)), len(self.classes))
        changed = [record for record in mapping if record["changed"]]
        self.assertEqual(
            [(record["class_id"], record["raw_label"], record["display_label"]) for record in changed],
            [(31, "Hyundai ", "Hyundai"), (41, "Lamorghini ", "Lamorghini")],
        )

    def test_display_trim_does_not_change_any_metric(self):
        baseline = metrics_from_confusion_matrix(self.matrix, self.classes)
        # A collision-free rename cannot touch counts, so only the reported values matter.
        self.assertAlmostEqual(baseline["macro_f1"], 0.5890462077274896, places=15)
        self.assertEqual(baseline["classes_with_zero_support"], [])
        self.assertEqual(baseline["rows"], 1120)
        support_below_ten = sum(1 for stats in baseline["per_class"].values() if stats["support"] < 10)
        self.assertEqual(support_below_ten, 19)


class ConfusionMatrixInputTests(unittest.TestCase):
    def test_matrix_metrics_match_row_level_metrics(self):
        rows = [("a", "a"), ("a", "a"), ("b", "b"), ("c", "a"), ("c", "c")]
        classes = ["a", "b", "c"]
        reported, _, _, matrix, _ = measure(rows, classes)
        from_matrix = metrics_from_confusion_matrix(matrix, classes)
        self.assertAlmostEqual(from_matrix["macro_f1"], reported["macro_f1"], places=12)
        self.assertAlmostEqual(from_matrix["accuracy"], reported["accuracy"], places=12)
        for name in classes:
            self.assertAlmostEqual(from_matrix["per_class"][name]["f1"], reported["per_class"][name]["f1"], places=12)

    def test_reads_project_json_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.json"
            path.write_text(
                json.dumps({"classes": ["a", "b"], "confusion_matrix": [[2, 1], [0, 3]]}), encoding="utf-8"
            )
            classes, matrix = read_confusion_matrix(path)
            self.assertEqual(classes, ["a", "b"])
            self.assertEqual(matrix.tolist(), [[2, 1], [0, 3]])

    def test_non_square_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.json"
            path.write_text(json.dumps({"classes": ["a", "b"], "confusion_matrix": [[1, 2]]}), encoding="utf-8")
            with self.assertRaises(EvaluationError):
                read_confusion_matrix(path)

    def test_csv_requires_identical_row_and_column_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.csv"
            path.write_text(",a,b\nb,1,0\na,0,1\n", encoding="utf-8")
            with self.assertRaises(EvaluationError):
                read_confusion_matrix(path)


class ColourReproductionTests(unittest.TestCase):
    """Reproduce the saved colour pilot result with the shared specification."""

    @classmethod
    def setUpClass(cls):
        if not COLOUR_DIR.is_dir():
            raise unittest.SkipTest(f"Colour pilot outputs not available at {COLOUR_DIR}")
        cls.frame = pd.read_csv(COLOUR_DIR / "validation_predictions.csv")
        cls.class_to_idx = json.loads((COLOUR_DIR / "class_to_idx.json").read_text(encoding="utf-8"))
        cls.reported = json.loads((COLOUR_DIR / "validation_metrics.json").read_text(encoding="utf-8"))

    def test_acceptance_numbers_are_reproduced(self):
        classes = list(self.class_to_idx)
        reported, _, excluded, matrix, _ = evaluate_rows(
            self.frame,
            classes,
            target_column="target",
            prediction_column="prediction",
            allow_unusable_rows=False,
            id_column="relative_key",
            expected_rows=386,
        )
        self.assertEqual(excluded.shape[0], 0)
        self.assertEqual(reported["rows"], 386)
        self.assertEqual(reported["correct"], 280)
        self.assertLessEqual(abs(reported["accuracy"] - 0.7253886010362695), 1e-10)
        self.assertLessEqual(abs(reported["macro_f1"] - 0.7299082471403191), 1e-10)
        self.assertEqual(matrix.tolist(), self.reported["confusion_matrix"])
        self.assertEqual(sum(reported["per_class"][name]["support"] for name in classes), 386)
        for name, value in self.reported["per_class_recall"].items():
            self.assertLessEqual(abs(reported["per_class"][name]["recall"] - value), 1e-10)

    def test_matrix_input_mode_agrees_with_row_level_mode(self):
        classes = list(self.class_to_idx)
        _, _, _, matrix, _ = evaluate_rows(
            self.frame,
            classes,
            target_column="target",
            prediction_column="prediction",
            allow_unusable_rows=False,
            id_column="relative_key",
        )
        from_matrix = metrics_from_confusion_matrix(matrix, classes)
        self.assertLessEqual(abs(from_matrix["macro_f1"] - self.reported["macro_f1"]), 1e-10)
        self.assertLessEqual(abs(from_matrix["accuracy"] - self.reported["accuracy"]), 1e-10)

    def test_weighted_f1_is_reported_and_differs_from_macro(self):
        classes = list(self.class_to_idx)
        reported, _, _, _, _ = evaluate_rows(
            self.frame,
            classes,
            target_column="target",
            prediction_column="prediction",
            allow_unusable_rows=False,
        )
        self.assertAlmostEqual(reported["weighted_f1"], 0.7264826819937054, places=12)
        self.assertNotAlmostEqual(reported["weighted_f1"], reported["macro_f1"], places=6)


class ConfusionCountValidationTests(unittest.TestCase):
    """R3-1: no count may be silently truncated, sign-coerced or coerced from text."""

    CLASSES = ["a", "b"]

    def write_json(self, directory: str, matrix) -> Path:
        path = Path(directory) / "matrix.json"
        path.write_text(json.dumps({"classes": self.CLASSES, "confusion_matrix": matrix}), encoding="utf-8")
        return path

    def test_fractional_counts_are_rejected_from_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_json(temporary, [[1.9, 0], [0, 1]])
            with self.assertRaises(EvaluationError) as context:
                read_confusion_matrix(path)
            self.assertIn("fractional count", str(context.exception))

    def test_negative_fractional_counts_are_rejected_not_coerced_to_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_json(temporary, [[-0.5, 1], [0, 1]])
            with self.assertRaises(EvaluationError):
                read_confusion_matrix(path)

    def test_negative_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_json(temporary, [[-1, 0], [0, 1]])
            with self.assertRaises(EvaluationError) as context:
                read_confusion_matrix(path)
            self.assertIn("negative count", str(context.exception))

    def test_boolean_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_json(temporary, [[True, False], [False, True]])
            with self.assertRaises(EvaluationError) as context:
                read_confusion_matrix(path)
            self.assertIn("boolean", str(context.exception))

    def test_text_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_json(temporary, [["1", "0"], ["0", "1"]])
            with self.assertRaises(EvaluationError) as context:
                read_confusion_matrix(path)
            self.assertIn("non-numeric", str(context.exception))

    def test_non_finite_counts_are_rejected(self):
        for token in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(token=token):
                with self.assertRaises(EvaluationError) as context:
                    validate_confusion_counts([[token, 0], [0, 1]])
                self.assertIn("non-finite", str(context.exception))

    def test_int64_overflow_is_rejected(self):
        with self.assertRaises(EvaluationError) as context:
            validate_confusion_counts([[2**70, 0], [0, 1]])
        self.assertIn("overflows int64", str(context.exception))

    def test_ragged_table_is_rejected(self):
        with self.assertRaises(EvaluationError):
            validate_confusion_counts([[1, 0], [0]])

    def test_integral_floats_are_accepted_and_documented(self):
        counts = validate_confusion_counts([[1.0, 0.0], [0.0, 2.0]])
        self.assertEqual(counts.dtype, np.int64)
        self.assertEqual(counts.tolist(), [[1, 0], [0, 2]])

    def test_csv_fractional_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.csv"
            path.write_text(",a,b\na,1.9,0\nb,0,1\n", encoding="utf-8")
            with self.assertRaises(EvaluationError):
                read_confusion_matrix(path)

    def test_csv_integral_float_column_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.csv"
            path.write_text(",a,b\na,1.0,0.0\nb,0.0,1.0\n", encoding="utf-8")
            classes, matrix = read_confusion_matrix(path)
            self.assertEqual(classes, ["a", "b"])
            self.assertEqual(matrix.tolist(), [[1, 0], [0, 1]])

    def test_direct_public_call_cannot_bypass_validation(self):
        for bad in ([[1.9, 0], [0, 1]], [[-0.5, 1], [0, 1]], [[-1, 0], [0, 1]]):
            with self.subTest(matrix=bad):
                with self.assertRaises(EvaluationError):
                    metrics_from_confusion_matrix(bad, self.CLASSES)

    def test_direct_public_call_still_checks_shape_and_class_count(self):
        with self.assertRaises(EvaluationError):
            metrics_from_confusion_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], self.CLASSES)
        with self.assertRaises(EvaluationError):
            metrics_from_confusion_matrix([[1, 0]], self.CLASSES)
        with self.assertRaises(EvaluationError):
            metrics_from_confusion_matrix([[0, 0], [0, 0]], self.CLASSES)

    def test_integral_float_counts_keep_the_metrics_unchanged(self):
        integer = metrics_from_confusion_matrix([[2, 1], [0, 3]], self.CLASSES)
        floaty = metrics_from_confusion_matrix([[2.0, 1.0], [0.0, 3.0]], self.CLASSES)
        self.assertAlmostEqual(integer["macro_f1"], floaty["macro_f1"], places=15)
        self.assertEqual(integer["rows"], floaty["rows"])


class ExactArithmeticCheckTests(unittest.TestCase):
    """R3-1 follow-up: aggregate mode needs a real independent check."""

    def test_exact_rational_recomputation_matches_the_float_path(self):
        matrix = np.array([[2, 1, 0], [0, 3, 1], [1, 0, 4]], dtype=np.int64)
        classes = ["a", "b", "c"]
        reported = metrics_from_confusion_matrix(matrix, classes)
        exact = exact_metrics_from_confusion_matrix(matrix)
        self.assertAlmostEqual(reported["macro_f1"], float(exact["macro_f1"]), places=15)
        check = reported["arithmetic_check"]
        self.assertTrue(check["passed"])
        self.assertLessEqual(check["max_abs_difference"], 1e-12)
        self.assertEqual(check["compared_values"], 5 + 3 * 3)
        self.assertIn("Fraction", check["basis"])

    def test_aggregate_cli_reports_the_cross_check_as_not_performed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(
                json.dumps({"classes": ["a", "b"], "confusion_matrix": [[2, 1], [0, 3]]}), encoding="utf-8"
            )
            output = root / "out"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts" / "evaluate_classification.py"),
                    "--task",
                    "unit",
                    "--split",
                    "validation",
                    "--confusion-matrix",
                    str(matrix_path),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["matrix_cross_check"]["performed"])
            self.assertEqual(payload["matrix_cross_check"]["status"], "not_performed")
            self.assertTrue(payload["arithmetic_check"]["passed"])


class ClassIndexValidationTests(unittest.TestCase):
    """R3-2: class indices must not be converted from fractional, boolean or textual values."""

    def load(self, payload) -> list[str]:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "class_to_idx.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_class_list(path)

    def test_fractional_indices_are_rejected(self):
        with self.assertRaises(EvaluationError) as context:
            self.load({"a": 0.9, "b": 1.9})
        self.assertIn("must be JSON integers", str(context.exception))

    def test_integral_float_indices_are_rejected(self):
        with self.assertRaises(EvaluationError):
            self.load({"a": 0.0, "b": 1.0})

    def test_boolean_indices_are_rejected(self):
        with self.assertRaises(EvaluationError) as context:
            self.load({"a": False, "b": True})
        self.assertIn("boolean", str(context.exception))

    def test_textual_indices_are_rejected(self):
        with self.assertRaises(EvaluationError):
            self.load({"a": "0", "b": "1"})

    def test_null_index_is_rejected(self):
        with self.assertRaises(EvaluationError):
            self.load({"a": None, "b": 1})

    def test_genuine_integers_are_accepted(self):
        self.assertEqual(self.load({"b": 1, "a": 0}), ["a", "b"])

    def test_gaps_and_duplicates_are_still_rejected(self):
        with self.assertRaises(EvaluationError):
            self.load({"a": 0, "b": 2})
        with self.assertRaises(EvaluationError):
            self.load({"a": 0, "b": 0})


class AggregateExpectedRowsTests(unittest.TestCase):
    """R3-3: aggregate mode must honour --expected-rows."""

    def run_cli(self, root: Path, expected: int) -> subprocess.CompletedProcess:
        matrix_path = root / "matrix.json"
        matrix_path.write_text(
            json.dumps({"classes": ["a", "b"], "confusion_matrix": [[1, 0], [0, 1]]}), encoding="utf-8"
        )
        return subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "evaluate_classification.py"),
                "--task",
                "unit",
                "--split",
                "validation",
                "--confusion-matrix",
                str(matrix_path),
                "--expected-rows",
                str(expected),
                "--output",
                str(root / "out"),
            ],
            capture_output=True,
            text=True,
        )

    def test_matching_row_count_succeeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = self.run_cli(root, 2)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue((root / "out" / "metrics.json").is_file())

    def test_mismatched_row_count_fails_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = self.run_cli(root, 999)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("Row count mismatch", completed.stdout)
            self.assertFalse((root / "out" / "metrics.json").exists())
            self.assertFalse((root / "out").exists())

    def test_fractional_matrix_fails_through_the_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(
                json.dumps({"classes": ["a", "b"], "confusion_matrix": [[1.9, 0], [0, 1]]}), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts" / "evaluate_classification.py"),
                    "--task",
                    "unit",
                    "--split",
                    "validation",
                    "--confusion-matrix",
                    str(matrix_path),
                    "--output",
                    str(root / "out"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertFalse((root / "out").exists())


class SplitConfirmationTests(unittest.TestCase):
    """The guard message and the guard behaviour must describe the same rule."""

    def test_guarded_names(self):
        for name in ("test", "Test", "official_test", "Official-Test", "officialtest", "holdout", "holdout_set"):
            with self.subTest(name=name):
                self.assertTrue(split_requires_confirmation(name))

    def test_unguarded_names(self):
        for name in ("validation", "fit", "train", "latest", "smoke", ""):
            with self.subTest(name=name):
                self.assertFalse(split_requires_confirmation(name))

    def test_cli_refuses_a_guarded_split_without_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(
                json.dumps({"classes": ["a", "b"], "confusion_matrix": [[1, 0], [0, 1]]}), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts" / "evaluate_classification.py"),
                    "--task",
                    "unit",
                    "--split",
                    "official_test",
                    "--confusion-matrix",
                    str(matrix_path),
                    "--output",
                    str(root / "out"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("official test", completed.stdout)


def snapshot(directory: Path) -> dict[str, str]:
    """Map every path under a directory to its content hash."""
    if not directory.exists():
        return {}
    state: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        state[relative] = "dir" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
    return state


def run_cli(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "evaluate_classification.py"), *arguments],
        capture_output=True,
        text=True,
    )


class ExistingOutputProtectionTests(unittest.TestCase):
    """R4-1: a refused run must not create, modify or delete anything on disk."""

    MARKER = "ORIGINAL LOG DO NOT CHANGE\n"

    def seed_directory(self, root: Path) -> Path:
        output = root / "existing_output"
        output.mkdir()
        (output / "run_log.txt").write_text(self.MARKER, encoding="utf-8")
        (output / "metrics.json").write_text('{"previous_run": true}\n', encoding="utf-8")
        (output / "notes.txt").write_text("earlier evidence\n", encoding="utf-8")
        return output

    def valid_matrix(self, root: Path) -> Path:
        path = root / "valid.json"
        path.write_text(json.dumps({"classes": ["a", "b"], "confusion_matrix": [[1, 0], [0, 1]]}), encoding="utf-8")
        return path

    def test_rejected_non_empty_directory_is_byte_for_byte_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.seed_directory(root)
            before = snapshot(output)
            completed = run_cli(
                "--task", "unit", "--split", "validation",
                "--confusion-matrix", str(self.valid_matrix(root)),
                "--output", str(output),
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("not fresh", completed.stdout)
            self.assertEqual(snapshot(output), before)
            self.assertEqual((output / "run_log.txt").read_text(encoding="utf-8"), self.MARKER)

    def test_rejected_directory_is_unchanged_even_when_the_input_is_bad(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.seed_directory(root)
            before = snapshot(output)
            bad = root / "bad.json"
            bad.write_text(json.dumps({"classes": ["a", "b"], "confusion_matrix": [[1.9, 0], [0, 1]]}), encoding="utf-8")
            completed = run_cli(
                "--task", "unit", "--split", "validation",
                "--confusion-matrix", str(bad),
                "--output", str(output),
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(snapshot(output), before)

    def test_rejected_directory_holding_only_a_log_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "only_log"
            output.mkdir()
            (output / "run_log.txt").write_text(self.MARKER, encoding="utf-8")
            before = snapshot(output)
            completed = run_cli(
                "--task", "unit", "--split", "validation",
                "--confusion-matrix", str(self.valid_matrix(root)),
                "--output", str(output),
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(snapshot(output), before)
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["run_log.txt"])

    def test_failure_is_reported_on_stderr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.seed_directory(root)
            completed = run_cli(
                "--task", "unit", "--split", "validation",
                "--confusion-matrix", str(self.valid_matrix(root)),
                "--output", str(output),
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("error:", completed.stderr)
            self.assertIn("not fresh", completed.stderr)

    def test_fresh_directory_still_writes_metrics_and_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "fresh"
            completed = run_cli(
                "--task", "unit", "--split", "validation",
                "--confusion-matrix", str(self.valid_matrix(root)),
                "--output", str(output),
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "run_log.txt").is_file())
            self.assertIn("completed:", (output / "run_log.txt").read_text(encoding="utf-8"))

    def test_empty_existing_directory_is_acquired_and_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "empty_dir"
            output.mkdir()
            completed = run_cli(
                "--task", "unit", "--split", "validation",
                "--confusion-matrix", str(self.valid_matrix(root)),
                "--output", str(output),
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertIn("acquired (existed and was empty)", (output / "run_log.txt").read_text(encoding="utf-8"))

    def test_failure_before_ownership_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "never_created"
            bad = root / "bad.json"
            bad.write_text(json.dumps({"classes": ["a", "b"], "confusion_matrix": [[-1, 0], [0, 1]]}), encoding="utf-8")
            completed = run_cli(
                "--task", "unit", "--split", "validation",
                "--confusion-matrix", str(bad),
                "--output", str(output),
            )
            self.assertEqual(completed.returncode, 2)
            self.assertFalse(output.exists())


class MixedTypeConfusionCountTests(unittest.TestCase):
    """R4-2: element types are judged before any dtype normalisation."""

    CLASSES = ["a", "b"]

    def test_mixed_boolean_and_integer_list_is_rejected(self):
        with self.assertRaises(EvaluationError) as context:
            validate_confusion_counts([[True, 0], [0, 1]])
        self.assertIn("boolean", str(context.exception))

    def test_mixed_boolean_and_float_list_is_rejected(self):
        with self.assertRaises(EvaluationError):
            validate_confusion_counts([[True, 0.0], [0, 1]])

    def test_boolean_in_any_position_is_rejected(self):
        for matrix in ([[1, 0], [0, True]], [[1, False], [0, 1]], [[False, 0], [0, 1]]):
            with self.subTest(matrix=matrix):
                with self.assertRaises(EvaluationError):
                    validate_confusion_counts(matrix)

    def test_mixed_boolean_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.json"
            path.write_text(json.dumps({"classes": self.CLASSES, "confusion_matrix": [[True, 0], [0, 1]]}), encoding="utf-8")
            with self.assertRaises(EvaluationError):
                read_confusion_matrix(path)

    def test_public_metrics_entry_point_rejects_mixed_booleans(self):
        with self.assertRaises(EvaluationError):
            metrics_from_confusion_matrix([[True, 0], [0, 1]], self.CLASSES)

    def test_plain_integers_are_still_accepted(self):
        self.assertEqual(validate_confusion_counts([[1, 0], [0, 2]]).tolist(), [[1, 0], [0, 2]])

    def test_integral_floats_are_still_accepted(self):
        counts = validate_confusion_counts([[1.0, 0.0], [0.0, 2.0]])
        self.assertEqual(counts.dtype, np.int64)
        self.assertEqual(counts.tolist(), [[1, 0], [0, 2]])

    def test_numpy_integer_array_is_still_accepted(self):
        # An array a caller already converted is accepted as-is: the original
        # element types no longer exist, so the tool cannot recover them.
        counts = validate_confusion_counts(np.array([[1, 0], [0, 2]], dtype=np.int64))
        self.assertEqual(counts.tolist(), [[1, 0], [0, 2]])

    def test_cli_rejects_a_mixed_boolean_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = root / "mixed.json"
            matrix.write_text(json.dumps({"classes": ["a", "b"], "confusion_matrix": [[True, 0], [0, 1]]}), encoding="utf-8")
            output = root / "out"
            completed = run_cli(
                "--task", "unit", "--split", "validation",
                "--confusion-matrix", str(matrix),
                "--output", str(output),
            )
            self.assertEqual(completed.returncode, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

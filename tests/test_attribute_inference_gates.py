"""Regression tests for the batch inference entry point's safety gates.

Every test mocks the checkpoint, the split table and image resolution, so no
weight file, no dataset image and no official test sample is touched. The device
tests use synthetic tensors and a synthetic tiny checkpoint.

The gate failures these tests lock down were demonstrated against earlier
revisions by the pretest audits on 2026-09-29.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "attribute_inference_eval.py"

spec = importlib.util.spec_from_file_location("pretest_entry_under_test", SCRIPT)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class StoppedBeforeImageAccess(Exception):
    """Raised by the mocked resolver so gates can be tested without images."""


def valid_payload() -> dict:
    return {
        "state_dict": {},
        "classes": ["red", "blue"],
        "config": {
            "task": "colour",
            "architecture": "resnet18",
            "image_size": 224,
            "preprocessing": "OpenCV whole-crop resize 224x224, RGB, ImageNet mean/std",
        },
        "synthetic": False,
    }


class GateTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.checkpoint = self.root / "checkpoint.bin"
        self.checkpoint.write_bytes(b"not a real checkpoint")
        self.split_file = self.root / "split.csv"
        self.split_file.write_text("k\n", encoding="utf-8")
        self.data_root = self.root / "data"
        self.data_root.mkdir()
        self.output = self.root / "out"
        self._seq = 0

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_membership(self, *, train=None, test=None, dataset="unit_dataset", **overrides) -> Path:
        self._seq += 1
        suffix = f"_{self._seq}"
        lists = []
        for label, keys in (("train", train or []), ("test", test or [])):
            path = self.root / f"{label}{suffix}.txt"
            path.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")
            lists.append({"split": label, "path": str(path), "sha256": entry.sha256_file(path)})
        descriptor = {
            "schema": entry.MEMBERSHIP_SCHEMA,
            "dataset": dataset,
            "version": "unit-version",
            "key_format": "key",
            "provenance": "unit fixture, not an official list",
            "lists": lists,
        }
        descriptor.update(overrides)
        target = self.root / f"membership{suffix}.json"
        target.write_text(json.dumps(descriptor), encoding="utf-8")
        return target

    def build_arguments(self, rows, **options) -> tuple[list[str], dict, object]:
        setting = {
            "membership": None,
            "dataset": "unit_dataset",
            "split_value": "validation",
            "split_column": "experiment_split",
            "key_column": "relative_key",
            "label_column": "label",
            "task": "colour",
            "sample_size": None,
            "payload": None,
            "emit": None,
        }
        setting.update(options)
        rows = [{**row, setting["split_column"]: setting["split_value"]} for row in rows]
        if setting["membership"] is None:
            setting["membership"] = self.write_membership(
                train=[str(row.get(setting["key_column"], "")) for row in rows]
            )
        import pandas as pd

        argv = [
            "--checkpoint", str(self.checkpoint),
            "--data-root", str(self.data_root),
            "--split-file", str(setting.get("split_file") or self.split_file),
            "--split-column", setting["split_column"],
            "--split-value", setting["split_value"],
            "--key-column", setting["key_column"],
            "--membership-source", str(setting["membership"]),
            "--dataset", setting["dataset"],
            "--label-column", setting["label_column"],
            "--task", setting["task"],
            "--output", str(self.output),
        ]
        if setting["sample_size"] is not None:
            argv += ["--sample-size", str(setting["sample_size"])]
        if setting.get("seed") is not None:
            argv += ["--seed", str(setting["seed"])]
        if setting.get("freeze") is not None:
            argv += ["--protocol-frozen", str(setting["freeze"])]
        if setting.get("optin"):
            argv += ["--allow-official-test"]
        if setting["emit"] is not None:
            argv += ["--emit-freeze-record", str(setting["emit"])]
        return argv, setting, pd.DataFrame(rows)

    def invoke(self, rows, **options):
        """Drive _run with the checkpoint, table and resolver mocked."""
        argv, setting, frame = self.build_arguments(rows, **options)
        args = entry.parse_args(argv)
        fake_model = SimpleNamespace(to=lambda *_: None, eval=lambda: None, parameters=lambda: [])
        fake = SimpleNamespace(classes=["red", "blue"], config=valid_payload()["config"],
                               synthetic=False, model=fake_model)
        with patch.object(entry.torch, "load", return_value=setting.get("payload") or valid_payload()), \
             patch.object(entry.pd, "read_csv", return_value=frame), \
             patch.object(entry, "resolve_image_path", side_effect=StoppedBeforeImageAccess), \
             patch.object(entry, "AttributePredictor", return_value=fake):
            with contextlib.redirect_stdout(io.StringIO()):
                return entry._run(args, [])

    def emit_draft(self, rows, **options) -> Path:
        options.setdefault("emit", "draft.json")
        self.invoke(rows, **options)
        return self.output / "draft.json"

    def frozen_record(self, rows, **options) -> Path:
        """Produce a frozen record the same way an operator would: emit, then sign."""
        draft = self.emit_draft(rows, **options)
        record = json.loads(draft.read_text(encoding="utf-8"))
        record["status"] = "frozen"
        target = self.root / f"frozen_{self._seq}.json"
        target.write_text(json.dumps(record), encoding="utf-8")
        return target


class MembershipDescriptorTests(GateTestBase):
    def row(self, key: str = "k1") -> dict:
        return {"experiment_split": "validation", "relative_key": key, "label": "red",
                "image_path": "never_opened.jpg"}

    def test_non_canonical_split_label_with_whitespace_is_refused(self):
        membership = self.write_membership(train=[], test=["k1"])
        descriptor = json.loads(membership.read_text(encoding="utf-8"))
        descriptor["lists"][1]["split"] = "test "
        membership.write_text(json.dumps(descriptor), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership)
        self.assertIn("not canonical", str(context.exception))

    def test_unknown_split_label_is_refused(self):
        membership = self.write_membership(train=[], test=["k1"])
        descriptor = json.loads(membership.read_text(encoding="utf-8"))
        descriptor["lists"][1]["split"] = "unrecognised"
        membership.write_text(json.dumps(descriptor), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership)
        self.assertIn("not canonical", str(context.exception))

    def test_differently_cased_split_label_is_refused(self):
        membership = self.write_membership(train=[], test=["k1"])
        descriptor = json.loads(membership.read_text(encoding="utf-8"))
        descriptor["lists"][1]["split"] = "Test"
        membership.write_text(json.dumps(descriptor), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership)
        self.assertIn("not canonical", str(context.exception))

    def test_duplicate_split_label_is_refused(self):
        membership = self.write_membership(train=["k1"], test=[])
        descriptor = json.loads(membership.read_text(encoding="utf-8"))
        descriptor["lists"][1]["split"] = "train"
        membership.write_text(json.dumps(descriptor), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership)
        self.assertIn("more than once", str(context.exception))

    def test_empty_version_is_refused(self):
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=self.write_membership(train=["k1"], version=""))
        self.assertIn("version", str(context.exception))

    def test_missing_provenance_is_refused(self):
        membership = self.write_membership(train=["k1"])
        descriptor = json.loads(membership.read_text(encoding="utf-8"))
        del descriptor["provenance"]
        membership.write_text(json.dumps(descriptor), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership)
        self.assertIn("missing", str(context.exception))

    def test_malformed_hash_is_refused(self):
        membership = self.write_membership(train=["k1"])
        descriptor = json.loads(membership.read_text(encoding="utf-8"))
        descriptor["lists"][0]["sha256"] = "not-a-hash"
        membership.write_text(json.dumps(descriptor), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership)
        self.assertIn("sha256", str(context.exception))


class MembershipGateTests(GateTestBase):
    def row(self, key: str = "k1", label: str = "red") -> dict:
        return {"experiment_split": "validation", "relative_key": key, "label": label,
                "image_path": "never_opened.jpg"}

    def test_membership_source_is_a_required_argument(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                entry.parse_args([
                    "--checkpoint", str(self.checkpoint), "--data-root", str(self.data_root),
                    "--split-file", str(self.split_file), "--split-column", "experiment_split",
                    "--split-value", "validation", "--label-column", "label", "--task", "colour",
                    "--output", str(self.output),
                ])

    def test_removed_bypass_argument_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                entry.parse_args([
                    "--checkpoint", str(self.checkpoint), "--data-root", str(self.data_root),
                    "--split-file", str(self.split_file), "--split-column", "experiment_split",
                    "--split-value", "validation", "--label-column", "label", "--task", "colour",
                    "--output", str(self.output), "--official-split-column", "official",
                ])

    def test_missing_key_column_is_refused(self):
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([{"experiment_split": "validation", "label": "red", "image_path": "x.jpg"}],
                        key_column="not_a_column")
        self.assertIn("canonical key", str(context.exception))

    def test_unknown_keys_are_refused(self):
        membership = self.write_membership(train=["something_else"])
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership)
        self.assertIn("no entry in the membership source", str(context.exception))

    def test_relabelled_train_split_carrying_test_members_is_refused(self):
        membership = self.write_membership(train=[], test=["k1"])
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership, split_value="validation")
        self.assertIn("train-like", str(context.exception))

    def test_relabelled_test_split_carrying_train_members_is_refused(self):
        membership = self.write_membership(train=["k1"], test=[])
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership, split_value="test")
        self.assertIn("test-like", str(context.exception))

    def test_test_split_with_test_members_needs_the_opt_in(self):
        membership = self.write_membership(train=[], test=["k1"])
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership, split_value="test")
        self.assertIn("refused by default", str(context.exception))

    def test_whitespace_in_the_split_value_does_not_bypass_the_rules(self):
        membership = self.write_membership(train=["k1"], test=[])
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership, split_value="test ")
        self.assertIn("test-like", str(context.exception))

    def test_unrecognised_split_name_is_refused(self):
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], split_value="holdout_part_a")
        self.assertIn("neither train-like nor test-like", str(context.exception))

    def test_membership_descriptor_hash_mismatch_is_refused(self):
        membership = self.write_membership(train=["k1"])
        Path(json.loads(membership.read_text(encoding="utf-8"))["lists"][0]["path"]).write_text(
            "tampered\n", encoding="utf-8"
        )
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership)
        self.assertIn("hash mismatch", str(context.exception))

    def test_dataset_mismatch_is_refused(self):
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], dataset="another_dataset")
        self.assertIn("must not be conflated", str(context.exception))

    def test_conflicting_membership_lists_are_refused(self):
        membership = self.write_membership(train=["k1"], test=["k1"])
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], membership=membership)
        self.assertIn("conflict", str(context.exception))

    def test_valid_train_control_passes_the_membership_gate(self):
        with self.assertRaises(StoppedBeforeImageAccess):
            self.invoke([self.row("k1")], membership=self.write_membership(train=["k1"]))

    def test_sampling_is_refused_for_test_members(self):
        membership = self.write_membership(train=[], test=["k1", "k2"])
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke(
                [self.row("k1"), self.row("k2")],
                membership=membership, split_value="test", optin=True,
                freeze=self.frozen_record([self.row("k1"), self.row("k2")],
                                          membership=membership, split_value="test", optin=True),
                sample_size=1,
            )
        self.assertIn("Sampling is refused", str(context.exception))


class FreezeRecordTests(GateTestBase):
    def row(self, key: str = "k1") -> dict:
        return {"experiment_split": "test", "relative_key": key, "label": "red",
                "image_path": "never_opened.jpg"}

    def scenario(self, **overrides) -> tuple[Path, Path, dict]:
        """A frozen record plus the options that produced it."""
        membership = self.write_membership(train=[], test=["k1"])
        options = {"membership": membership, "split_value": "test", "optin": True}
        options.update(overrides)
        return membership, self.frozen_record([self.row("k1")], **options), options

    def test_arbitrary_text_is_not_a_freeze_record(self):
        membership = self.write_membership(train=[], test=["k1"])
        plain = self.root / "notes.txt"
        plain.write_text("draft protocol, not a record\n", encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], membership=membership, split_value="test", optin=True, freeze=plain)
        self.assertIn("not machine-readable JSON", str(context.exception))

    def test_a_source_file_is_not_a_freeze_record(self):
        membership = self.write_membership(train=[], test=["k1"])
        with self.assertRaises(entry.InferenceError):
            self.invoke([self.row()], membership=membership, split_value="test", optin=True, freeze=SCRIPT)

    def test_draft_status_is_refused(self):
        membership = self.write_membership(train=[], test=["k1"])
        draft = self.emit_draft([self.row()], membership=membership, split_value="test", optin=True)
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], membership=membership, split_value="test", optin=True, freeze=draft)
        self.assertIn("only 'frozen' is accepted", str(context.exception))

    def test_incomplete_record_is_refused(self):
        membership, freeze, options = self.scenario()
        record = json.loads(freeze.read_text(encoding="utf-8"))
        del record["selected_rows_sha256"]
        freeze.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], freeze=freeze, **options)
        self.assertIn("incomplete", str(context.exception))

    def test_missing_file_is_refused(self):
        membership = self.write_membership(train=[], test=["k1"])
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], membership=membership, split_value="test", optin=True,
                        freeze=self.root / "absent.json")
        self.assertIn("Freeze record not found", str(context.exception))

    def test_empty_code_hash_block_is_refused(self):
        membership, freeze, options = self.scenario()
        record = json.loads(freeze.read_text(encoding="utf-8"))
        record["code_hashes"] = {}
        freeze.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], freeze=freeze, **options)
        self.assertIn("incomplete", str(context.exception))

    def test_partial_code_hash_block_is_refused(self):
        membership, freeze, options = self.scenario()
        record = json.loads(freeze.read_text(encoding="utf-8"))
        record["code_hashes"] = dict(list(record["code_hashes"].items())[:1])
        freeze.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], freeze=freeze, **options)
        self.assertIn("incomplete", str(context.exception))

    def test_code_hash_block_of_the_wrong_type_is_refused(self):
        membership, freeze, options = self.scenario()
        record = json.loads(freeze.read_text(encoding="utf-8"))
        record["code_hashes"] = ["scripts/attribute_inference_eval.py"]
        freeze.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], freeze=freeze, **options)
        self.assertIn("JSON object", str(context.exception))

    def test_unknown_file_in_the_code_hash_block_is_refused(self):
        membership, freeze, options = self.scenario()
        record = json.loads(freeze.read_text(encoding="utf-8"))
        record["code_hashes"]["scripts/some_other.py"] = "0" * 64
        freeze.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], freeze=freeze, **options)
        self.assertIn("unknown file", str(context.exception))

    def test_changed_code_hash_is_refused(self):
        membership, freeze, options = self.scenario()
        record = json.loads(freeze.read_text(encoding="utf-8"))
        first = sorted(record["code_hashes"])[0]
        record["code_hashes"][first] = "f" * 64
        freeze.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], freeze=freeze, **options)
        self.assertIn("does not match this run", str(context.exception))

    def test_checkpoint_hash_mismatch_is_refused(self):
        membership, freeze, options = self.scenario()
        record = json.loads(freeze.read_text(encoding="utf-8"))
        record["checkpoint_sha256"] = "1" * 64
        freeze.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], freeze=freeze, **options)
        self.assertIn("checkpoint_sha256", str(context.exception))

    def test_valid_record_reaches_the_image_pre_flight(self):
        membership, freeze, options = self.scenario()
        with self.assertRaises(StoppedBeforeImageAccess):
            self.invoke([self.row()], freeze=freeze, **options)


class SelectionBindingTests(GateTestBase):
    """The freeze must bind the evaluation subset, not just the membership source."""

    def row(self, key: str = "k1", label: str = "red", bbox=None) -> dict:
        record = {"experiment_split": "test", "relative_key": key, "label": label,
                  "image_path": f"{key}.jpg"}
        if bbox is not None:
            record["bbox"] = bbox
        return record

    def scenario(self, **overrides):
        rows = [self.row("k1"), self.row("k2")]
        membership = self.write_membership(train=[], test=["k1", "k2"])
        options = {"membership": membership, "split_value": "test", "optin": True}
        options.update(overrides)
        return rows, membership, self.frozen_record(rows, **options), options

    def test_a_different_selected_row_set_is_refused(self):
        rows, membership, freeze, options = self.scenario()
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row("k1")], freeze=freeze, **options)
        self.assertIn("selected_rows_sha256", str(context.exception))

    def test_a_changed_target_label_is_refused(self):
        rows, membership, freeze, options = self.scenario()
        changed = [self.row("k1"), self.row("k2", label="blue")]
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke(changed, freeze=freeze, **options)
        self.assertIn("selected_rows_sha256", str(context.exception))

    def test_a_changed_bounding_box_is_refused(self):
        rows = [self.row("k1", bbox="(1, 2, 30, 40)"), self.row("k2", bbox="(5, 6, 70, 80)")]
        membership = self.write_membership(train=[], test=["k1", "k2"])
        options = {"membership": membership, "split_value": "test", "optin": True}
        freeze = self.frozen_record(rows, **options)
        changed = [self.row("k1", bbox="(1, 2, 300, 400)"), self.row("k2", bbox="(5, 6, 70, 80)")]
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke(changed, freeze=freeze, **options)
        self.assertIn("selected_rows_sha256", str(context.exception))

    def test_a_different_split_value_is_refused(self):
        rows, membership, freeze, options = self.scenario()
        record = json.loads(freeze.read_text(encoding="utf-8"))
        record["split_value"] = "validation"
        freeze.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke(rows, freeze=freeze, **options)
        self.assertIn("split_value", str(context.exception))

    def test_a_changed_split_file_is_refused(self):
        rows, membership, freeze, options = self.scenario()
        self.split_file.write_text("k\nk2\n", encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke(rows, freeze=freeze, **options)
        self.assertIn("split_file_sha256", str(context.exception))

    def test_a_different_label_column_is_refused(self):
        rows, membership, freeze, options = self.scenario()
        record = json.loads(freeze.read_text(encoding="utf-8"))
        record["label_column"] = "other_label"
        freeze.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke(rows, freeze=freeze, **options)
        self.assertIn("label_column", str(context.exception))


class EmitFreezeRecordSafetyTests(GateTestBase):
    def row(self, key: str = "k1") -> dict:
        return {"experiment_split": "validation", "relative_key": key, "label": "red",
                "image_path": "never_opened.jpg"}

    def test_existing_target_is_not_overwritten(self):
        membership = self.write_membership(train=["k1"])
        self.output.mkdir()
        target = self.output / "draft.json"
        target.write_text("REVIEWED CONTENT DO NOT CHANGE\n", encoding="utf-8")
        # The fresh-directory rule already blocks this situation in normal use, so
        # it is stubbed here to reach the second line of defence: exclusive creation.
        with patch.object(entry, "prepare_output", lambda *_: None):
            with self.assertRaises(entry.InferenceError) as context:
                self.invoke([self.row()], membership=membership, emit="draft.json")
        self.assertIn("Refusing to overwrite", str(context.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), "REVIEWED CONTENT DO NOT CHANGE\n")

    def test_a_populated_output_directory_is_refused_before_anything_is_written(self):
        membership = self.write_membership(train=["k1"])
        self.output.mkdir()
        (self.output / "earlier_evidence.txt").write_text("keep me\n", encoding="utf-8")
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], membership=membership, emit="draft.json")
        self.assertIn("not fresh", str(context.exception))
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), ["earlier_evidence.txt"])

    def test_target_outside_the_output_directory_is_refused(self):
        membership = self.write_membership(train=["k1"])
        outside = self.root / "elsewhere.json"
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], membership=membership, emit=outside)
        self.assertIn("must be written inside --output", str(context.exception))
        self.assertFalse(outside.exists())

    def test_sibling_directory_sharing_a_name_prefix_is_refused(self):
        """'out_sibling' begins with the characters of 'out' but is not inside it.

        A string-prefix containment check accepted this; resolved-path ancestry
        does not.
        """
        membership = self.write_membership(train=["k1"])
        sibling = self.root / "out_sibling"
        sibling.mkdir()
        target = sibling / "draft.json"
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], membership=membership, emit=target)
        self.assertIn("must be written inside --output", str(context.exception))
        self.assertFalse(target.is_file())

    def test_target_inside_the_output_directory_is_accepted(self):
        """Positive control for the containment check."""
        membership = self.write_membership(train=["k1"])
        draft = self.emit_draft([self.row()], membership=membership)
        self.assertTrue(draft.is_file())
        self.assertTrue(draft.resolve().is_relative_to(self.output.resolve()))

    def test_draft_carries_status_draft(self):
        membership = self.write_membership(train=["k1"])
        draft = self.emit_draft([self.row()], membership=membership)
        record = json.loads(draft.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "draft")
        self.assertEqual(record["selected_count"], 1)


class CheckpointValidationTests(GateTestBase):
    def row(self) -> dict:
        return {"experiment_split": "validation", "relative_key": "k1", "label": "red",
                "image_path": "never_opened.jpg"}

    def test_missing_checkpoint_keys_are_refused(self):
        broken = valid_payload()
        del broken["classes"]
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], payload=broken)
        self.assertIn("missing required key", str(context.exception))

    def test_missing_config_keys_are_refused(self):
        broken = valid_payload()
        del broken["config"]["preprocessing"]
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], payload=broken)
        self.assertIn("config is missing", str(context.exception))

    def test_unsupported_transform_is_refused(self):
        broken = valid_payload()
        broken["config"]["preprocessing"] = "some other pipeline"
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], payload=broken)
        self.assertIn("unsupported preprocessing contract", str(context.exception))

    def test_image_size_disagreeing_with_the_transform_is_refused(self):
        broken = valid_payload()
        broken["config"]["image_size"] = 256
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], payload=broken)
        self.assertIn("disagrees with the transform contract", str(context.exception))

    def test_task_mismatch_is_refused(self):
        broken = valid_payload()
        broken["config"]["task"] = "make"
        with self.assertRaises(entry.InferenceError) as context:
            self.invoke([self.row()], payload=broken)
        self.assertIn("does not match --task", str(context.exception))

    def test_duplicate_classes_are_refused(self):
        duplicate = valid_payload()
        duplicate["classes"] = ["red", "red"]
        with self.assertRaises(entry.InferenceError):
            self.invoke([self.row()], payload=duplicate)


class DevicePolicyTests(unittest.TestCase):
    def test_entry_point_moves_the_model_to_the_selected_device(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("predictor.model.to(device)", source)
        self.assertIn("predictor.model.eval()", source)
        self.assertIn("parameter_devices", source)


class DevicePolicyFunctionalTests(unittest.TestCase):
    """Exercise the policy with synthetic weights and a synthetic image."""

    def setUp(self) -> None:
        import torch

        from vehicle_id.attributes.modelling import build_model

        self.torch = torch
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        config = {
            "task": "colour",
            "architecture": "tiny_cnn",
            "image_size": 224,
            "preprocessing": "OpenCV whole-crop resize 224x224, RGB, ImageNet mean/std",
        }
        model = build_model("tiny_cnn", 2)
        self.checkpoint = self.root / "tiny.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": ["red", "blue"],
                "config": config,
                "synthetic": False,
                "best_epoch": 1,
                "manifest_sha256": "0" * 64,
            },
            self.checkpoint,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def synthetic_image(self):
        import numpy as np

        return (np.random.default_rng(0).random((224, 224, 3)) * 255).astype("uint8")

    def test_parameters_and_outputs_follow_the_selected_device(self):
        from vehicle_id.attributes.modelling import AttributePredictor, normalise_image

        devices = ["cpu"] + (["cuda"] if self.torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                predictor = AttributePredictor(self.checkpoint)
                predictor.model.to(device)
                predictor.model.eval()
                # device.type is 'cuda' while str(device) is 'cuda:0'
                self.assertEqual({p.device.type for p in predictor.model.parameters()}, {device})
                tensor = normalise_image(self.synthetic_image()).unsqueeze(0).to(device)
                with self.torch.inference_mode():
                    output = predictor.model(tensor)
                self.assertEqual(str(output.device).split(":")[0], device)

    def test_moving_only_the_input_is_what_the_old_path_did(self):
        if not self.torch.cuda.is_available():
            self.skipTest("CUDA is not available on this machine")
        from vehicle_id.attributes.modelling import AttributePredictor, normalise_image

        predictor = AttributePredictor(self.checkpoint)  # stays on CPU, as before
        tensor = normalise_image(self.synthetic_image()).unsqueeze(0).to("cuda")
        with self.assertRaises(RuntimeError):
            with self.torch.inference_mode():
                predictor.model(tensor)


if __name__ == "__main__":
    unittest.main()

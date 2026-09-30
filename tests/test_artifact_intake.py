"""Tests for the checkpoint intake check.

Everything is synthetic: a tiny model is built and saved in a temporary
directory. No dataset image, no pretrained download and no real checkpoint is
involved.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "verify_attribute_artifact.py"

spec = importlib.util.spec_from_file_location("intake_under_test", SCRIPT)
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)


def valid_config() -> dict:
    return {
        "task": "colour",
        "architecture": "tiny_cnn",
        "image_size": 224,
        "preprocessing": "OpenCV whole-crop resize 224x224, RGB, ImageNet mean/std",
    }


class IntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        from vehicle_id.attributes.modelling import build_model

        self.torch = torch
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.checkpoint = self.root / "tiny.pt"
        self.payload = {
            "state_dict": build_model("tiny_cnn", 2).state_dict(),
            "classes": ["red", "blue"],
            "config": valid_config(),
            "synthetic": False,
            "best_epoch": 1,
            "manifest_sha256": "0" * 64,
        }
        torch.save(self.payload, self.checkpoint)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def save(self, payload: dict) -> Path:
        path = self.root / f"variant_{len(list(self.root.glob('variant_*.pt')))}.pt"
        self.torch.save(payload, path)
        return path

    def run_check(self, checkpoint: Path | None = None, **overrides):
        argv = ["--checkpoint", str(checkpoint or self.checkpoint), "--task", overrides.pop("task", "colour")]
        if "expected_sha256" in overrides:
            argv += ["--expected-sha256", overrides.pop("expected_sha256")]
        if "class_mapping" in overrides:
            argv += ["--class-mapping", str(overrides.pop("class_mapping"))]
        return intake.parse_args(argv), overrides

    def inspect(self, checkpoint: Path | None = None, **overrides) -> dict:
        args, _ = self.run_check(checkpoint, **overrides)
        return intake.inspect(args)

    def test_a_valid_checkpoint_is_acceptable(self):
        report = self.inspect()
        self.assertEqual(report["task"], "colour")
        self.assertEqual(report["transform_id"], "whole_crop_square_resize_imagenet_v1")
        self.assertEqual(report["class_count"], 2)
        self.assertEqual(report["state_dict_load"], "loaded strictly")
        self.assertIsNotNone(report["parameter_count"])
        self.assertEqual(report["duplicate_classes"], [])

    def test_the_report_carries_the_values_a_freeze_record_needs(self):
        values = self.inspect()["freeze_record_values"]
        self.assertEqual(
            sorted(values),
            ["checkpoint_sha256", "class_order_sha256", "image_size", "task", "transform_id"],
        )

    def test_a_hash_mismatch_is_refused(self):
        with self.assertRaises(intake.IntakeRefusal) as context:
            self.inspect(expected_sha256="0" * 64)
        self.assertIn("Hash mismatch", str(context.exception))

    def test_the_matching_hash_is_accepted(self):
        report = self.inspect(expected_sha256=intake.sha256_file(self.checkpoint))
        self.assertEqual(report["sha256"], intake.sha256_file(self.checkpoint))

    def test_a_wrong_task_is_refused(self):
        with self.assertRaises(intake.IntakeRefusal) as context:
            self.inspect(task="make")
        self.assertIn("does not match the expected task", str(context.exception))

    def test_an_unsupported_transform_is_refused(self):
        payload = dict(self.payload)
        payload["config"] = {**valid_config(), "preprocessing": "something else"}
        with self.assertRaises(intake.IntakeRefusal):
            self.inspect(self.save(payload))

    def test_a_missing_key_is_refused(self):
        payload = dict(self.payload)
        payload.pop("classes")
        with self.assertRaises(intake.IntakeRefusal):
            self.inspect(self.save(payload))

    def test_a_synthetic_checkpoint_is_refused(self):
        payload = {**self.payload, "synthetic": True}
        with self.assertRaises(intake.IntakeRefusal) as context:
            self.inspect(self.save(payload))
        self.assertIn("synthetic", str(context.exception))

    def test_an_incompatible_state_dict_is_refused(self):
        from vehicle_id.attributes.modelling import build_model

        payload = {**self.payload, "state_dict": build_model("tiny_cnn", 5).state_dict()}
        with self.assertRaises(intake.IntakeRefusal) as context:
            self.inspect(self.save(payload))
        self.assertIn("does not load", str(context.exception))

    def test_a_mismatched_class_mapping_is_refused(self):
        mapping = self.root / "classes.json"
        mapping.write_text(json.dumps(["red", "green"]), encoding="utf-8")
        with self.assertRaises(intake.IntakeRefusal) as context:
            self.inspect(class_mapping=mapping)
        self.assertIn("class mapping", str(context.exception))

    def test_a_matching_class_mapping_is_recorded(self):
        mapping = self.root / "classes_ok.json"
        mapping.write_text(json.dumps(["red", "blue"]), encoding="utf-8")
        report = self.inspect(class_mapping=mapping)
        self.assertIn("matches the checkpoint", report["class_mapping"])

    def test_a_missing_file_is_refused(self):
        with self.assertRaises(intake.IntakeRefusal) as context:
            self.inspect(self.root / "absent.pt")
        self.assertIn("not found", str(context.exception))

    def test_duplicate_classes_are_reported_and_refused(self):
        payload = {**self.payload, "classes": ["red", "red"]}
        with self.assertRaises(intake.IntakeRefusal):
            self.inspect(self.save(payload))


if __name__ == "__main__":
    unittest.main()

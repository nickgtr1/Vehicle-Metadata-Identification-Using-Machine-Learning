"""Command-line runner for the Stage 2 deterministic baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vehicle_id.pipeline import run_stage2_baseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--localisation-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage2_baseline.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    localisation = json.loads(args.localisation_json.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    outputs = run_stage2_baseline(
        image_path=args.image,
        localisation_payload=localisation,
        config=config,
        output_dir=args.output_dir,
        command=command,
    )
    print("Stage 2 baseline: PASS")
    print(f"Vehicles processed: {outputs['vehicle_count']}")
    print(f"JSON: {outputs['json']}")
    print(f"CSV: {outputs['csv']}")
    print(f"Annotated image: {outputs['annotated_image']}")
    print("Evidence boundary: colour is an unvalidated baseline; other attributes were not assessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

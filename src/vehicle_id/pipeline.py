"""Stage 1 localisation to Stage 2 attribute-baseline integration."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

from vehicle_id.attributes.colour import estimate_colour
from vehicle_id.schema import VehicleProfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _bounded_bbox(values: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError("bbox_xyxy must contain exactly four numbers")
    x1, y1, x2, y2 = (int(round(float(value))) for value in values)
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid or empty bounded box: {[x1, y1, x2, y2]}")
    return x1, y1, x2, y2


def run_stage2_baseline(
    image_path: Path,
    localisation_payload: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    command: str,
) -> dict[str, Path | int]:
    """Consume Stage 1 boxes and emit auditable Stage 2 baseline artefacts."""

    image_path = image_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        label_font = ImageFont.truetype("arial.ttf", max(18, width // 90))
    except OSError:
        label_font = ImageFont.load_default(size=max(18, width // 90))
    records: list[dict[str, Any]] = []
    reasons = config.get("unavailable_models", {})

    for index, detection in enumerate(localisation_payload.get("detections", []), start=1):
        crop_id = str(detection.get("vehicle_crop_id") or f"vehicle_{index:03d}")
        x1, y1, x2, y2 = _bounded_bbox(detection["bbox_xyxy"], width, height)
        crop = image.crop((x1, y1, x2, y2))
        crop_path = crops_dir / f"{crop_id}.jpg"
        crop.save(crop_path, quality=95)

        colour = estimate_colour(crop, config["colour_baseline"])
        profile = VehicleProfile(
            colour=colour.label,
            confidence={"colour": colour.confidence},
            status={
                "make": "not_assessed",
                "model": "not_assessed",
                "body_type": "not_assessed",
                "colour": colour.status,
                "damage": "not_assessed",
                "accessories": "not_assessed",
            },
        )
        record = {
            "vehicle_crop_id": crop_id,
            "bbox_xyxy": [x1, y1, x2, y2],
            "detection_confidence": float(detection["confidence"]),
            "crop_path": str(crop_path),
            "vehicle_profile": asdict(profile),
            "colour_evidence": asdict(colour),
            "not_assessed_reasons": reasons,
        }
        records.append(record)

        label = f"V{index:02d}: {colour.label or 'unknown'}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 215, 0), width=4)
        text_box = draw.textbbox((0, 0), label, font=label_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = min(x1 + 3, max(0, width - text_width - 8))
        text_y = y1 + 3
        draw.rectangle(
            (text_x - 3, text_y - 3, text_x + text_width + 3, text_y + text_height + 5),
            fill=(0, 0, 0),
        )
        draw.text((text_x, text_y), label, fill=(255, 215, 0), font=label_font)

    annotated_path = output_dir / "annotated_stage2_baseline.jpg"
    annotated.save(annotated_path, quality=95)

    colour_labels = [record["vehicle_profile"]["colour"] for record in records]
    warnings: list[str] = []
    if len(colour_labels) >= 3:
        most_common_count = max(colour_labels.count(label) for label in set(colour_labels))
        if most_common_count / len(colour_labels) >= 0.8:
            warnings.append(
                "At least 80% of vehicles received the same colour label; inspect for scene colour cast, "
                "background contamination or baseline bias before interpreting the labels."
            )

    result = {
        "schema_version": "0.2.0-smoke-test",
        "project_code": "36127",
        "source_image_id": localisation_payload.get("source_image_id", image_path.stem),
        "source_image_path": str(image_path),
        "localisation_model": localisation_payload.get("localisation_model", {}),
        "stage2_component": {
            "name": "deterministic-colour-baseline",
            "version": "0.1.1",
            "confidence_semantics": "Dominant classified-pixel share, not predictive accuracy.",
            "evidence_notice": config.get("evidence_notice"),
        },
        "vehicle_count": len(records),
        "vehicles": records,
        "warnings": warnings,
        "run_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "random_seed": config.get("random_seed"),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "pillow": PIL.__version__,
            },
            "source_image_sha256": _sha256(image_path),
            "client_data_used": False,
            "model_weights_used": False,
        },
        "what_this_proves": [
            "The supplied Stage 1 bounding boxes can be consumed and cropped by Stage 2.",
            "The deterministic colour baseline generates JSON, CSV, crop and annotated-image outputs.",
        ],
        "what_this_does_not_prove": [
            "It does not validate colour accuracy or any operational metric.",
            "It does not train or evaluate make, model, body type, damage or accessory models.",
            "It does not establish permission to use NSW Police data.",
        ],
    }

    json_path = output_dir / "stage2_results.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = output_dir / "stage2_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_image_id", "vehicle_crop_id", "bbox_x1", "bbox_y1",
                "bbox_x2", "bbox_y2", "detection_confidence", "colour",
                "colour_confidence", "colour_status", "make_status", "model_status",
                "body_type_status", "damage_status", "accessories_status",
            ],
        )
        writer.writeheader()
        for record in records:
            profile = record["vehicle_profile"]
            x1, y1, x2, y2 = record["bbox_xyxy"]
            writer.writerow({
                "source_image_id": result["source_image_id"],
                "vehicle_crop_id": record["vehicle_crop_id"],
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "detection_confidence": record["detection_confidence"],
                "colour": profile["colour"],
                "colour_confidence": profile["confidence"]["colour"],
                "colour_status": profile["status"]["colour"],
                "make_status": profile["status"]["make"],
                "model_status": profile["status"]["model"],
                "body_type_status": profile["status"]["body_type"],
                "damage_status": profile["status"]["damage"],
                "accessories_status": profile["status"]["accessories"],
            })

    return {"json": json_path, "csv": csv_path, "annotated_image": annotated_path, "vehicle_count": len(records)}

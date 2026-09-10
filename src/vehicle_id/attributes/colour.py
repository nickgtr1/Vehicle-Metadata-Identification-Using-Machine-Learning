"""Deterministic colour baseline for vehicle crops.

This module intentionally uses no trained weights. It estimates the dominant
colour in a configurable central region and reports the dominant pixel share as
an internal confidence indicator, not as classification accuracy.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ColourEstimate:
    label: str | None
    confidence: float | None
    method: str
    status: str
    evaluated_pixels: int
    class_distribution: dict[str, float]


def _validate_fraction(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def estimate_colour(image: Image.Image, config: dict[str, Any]) -> ColourEstimate:
    """Estimate a coarse colour label from the configured central crop region."""

    method = str(config.get("method", "central-roi-hsv-v1"))
    roi = config["central_roi"]
    x_start = _validate_fraction(roi["x_start"], "x_start")
    x_end = _validate_fraction(roi["x_end"], "x_end")
    y_start = _validate_fraction(roi["y_start"], "y_start")
    y_end = _validate_fraction(roi["y_end"], "y_end")
    if x_start >= x_end or y_start >= y_end:
        raise ValueError("central_roi start values must be smaller than end values")

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    x1, x2 = int(width * x_start), max(int(width * x_end), 1)
    y1, y2 = int(height * y_start), max(int(height * y_end), 1)
    central = rgb[y1:y2, x1:x2]
    pixel_count = int(central.shape[0] * central.shape[1])
    minimum_pixels = int(config.get("minimum_pixels", 100))
    if pixel_count < minimum_pixels:
        return ColourEstimate(
            label=None,
            confidence=None,
            method=method,
            status="insufficient_pixels",
            evaluated_pixels=pixel_count,
            class_distribution={},
        )

    hsv = cv2.cvtColor(central, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    achromatic = config["achromatic"]
    black = value <= int(achromatic["black_value_max"])
    white = (
        (value >= int(achromatic["white_value_min"]))
        & (saturation <= int(achromatic["white_saturation_max"]))
    )
    grey = (
        (saturation <= int(achromatic["grey_saturation_max"]))
        & ~black
        & ~white
    )
    chromatic = ~(black | white | grey)

    labels = np.full(hue.shape, "red", dtype="<U7")
    labels[black] = "black"
    labels[white] = "white"
    labels[grey] = "grey"

    hue_ranges = config["hue_ranges"]
    for label in ("orange", "yellow", "green", "blue", "purple", "pink"):
        low, high = (int(v) for v in hue_ranges[label])
        labels[chromatic & (hue >= low) & (hue < high)] = label

    unique, counts = np.unique(labels, return_counts=True)
    distribution = {
        str(label): round(int(count) / pixel_count, 6)
        for label, count in zip(unique, counts, strict=True)
    }
    dominant_label = max(distribution, key=distribution.get)
    return ColourEstimate(
        label=dominant_label,
        confidence=distribution[dominant_label],
        method=method,
        status="baseline_estimate",
        evaluated_pixels=pixel_count,
        class_distribution=distribution,
    )

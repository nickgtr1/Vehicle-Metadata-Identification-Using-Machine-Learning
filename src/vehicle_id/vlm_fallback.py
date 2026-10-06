"""VLM-based fallback for make / model / body_type, mirroring the existing
HSV colour-baseline block in streamlit_app.py. Not routed through pipeline.py's
`predictors` mechanism: `model` isn't a supported predictor key there, and
predict_profile_tuned() already returns make+model+body_type in a single call,
so going through the one-task-at-a-time predictor interface would mean 2-3x
redundant API calls for no benefit.
"""
import tempfile
from pathlib import Path

from PIL import Image


def _pad_bbox(bbox_xyxy, width, height, padding):
    """Expand a tight detection box by `padding` (fraction of box size) on each
    side, clamped to image bounds. CompCars training photos aren't cropped as
    tightly as a YOLO box, so a little context around the vehicle better
    matches what the fine-tuned model actually learned on."""
    x1, y1, x2, y2 = bbox_xyxy
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )


def apply_vlm_fallback(image_path, vehicles, predictors, *, enabled, mode="tuned", padding=0.15):
    """Mutates each vehicle's vehicle_profile dict in place for any of
    make / body_type / model not already covered by a trained predictor
    ('model' is never covered by predictors -- it's always VLM-only).

    mode="tuned": predict_profile_tuned() -- best accuracy, needs Vertex/gcloud auth.
    mode="zero-shot": predict_profile() with the constrained prompt -- lower
        accuracy, only needs GEMINI_API_KEY, portable to any teammate's machine.
    padding: fraction of the detection box size to expand by on each side
        before cropping (default 0.15 = 15%), so the crop looks more like a
        training-style photo and less like a tight, zoomed-in YOLO box.

    Returns (vehicles, error_message_or_None). Continues past a single
    vehicle's failure rather than aborting the whole batch; the last error
    seen (if any) is returned so the caller can surface one warning.
    """
    if not enabled or not vehicles:
        return vehicles, None

    fields_needed = [a for a in ("make", "body_type") if a not in predictors] + ["model"]
    if not fields_needed:
        return vehicles, None

    from vehicle_id.vlm.client import predict_profile, predict_profile_tuned  # lazy: needs google-genai/API key

    call = (lambda p: predict_profile_tuned(p)) if mode == "tuned" else (lambda p: predict_profile(p, prompt="constrained"))

    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
    width, height = rgb.size

    error = None
    for rec in vehicles:
        prof = rec["vehicle_profile"]
        box = _pad_bbox(rec["bbox_xyxy"], width, height, padding)
        crop = rgb.crop(box)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            crop.save(tmp_path, quality=95)
            vlm_profile, _usage = call(tmp_path)
        except Exception as exc:  # noqa: BLE001 -- surface as a warning, keep processing other vehicles
            error = f"{type(exc).__name__}: {exc}"
            continue
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        for field in fields_needed:
            value = getattr(vlm_profile, field, None)
            if value is None:
                continue
            prof[field] = value
            prof["confidence"][field] = None  # VLM has no calibrated softmax score
            prof["status"][field] = f"vlm ({mode}, gemini-3.1-flash-lite, padded crop)"

    return vehicles, error

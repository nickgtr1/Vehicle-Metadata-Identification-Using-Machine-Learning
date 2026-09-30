"""Streamlit demo for the 36127 vehicle metadata pipeline.

Run from the repo root (app lives in app/):
    streamlit run app/streamlit_app.py
Set VEHICLE_ID_REPO to the repo root if the file is run from elsewhere.

Nothing is downloaded. YOLO weights and attribute checkpoints must already exist
locally. Without a detector you can paste Stage 1 boxes as JSON. Without a colour
checkpoint the deterministic HSV colour baseline is used.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from PIL import Image

REPO_ROOT = Path(os.environ.get("VEHICLE_ID_REPO") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(REPO_ROOT / "src"))

from vehicle_id.attributes.colour import estimate_colour  # noqa: E402
from vehicle_id.pipeline import analyse_vehicle_image  # noqa: E402

BASELINE_CONFIG = REPO_ROOT / "configs" / "stage2_baseline.yaml"
ATTRIBUTES = ("make", "body_type", "colour")
SWATCH = {
    "black": "#111111", "white": "#f5f5f5", "grey": "#8c8c8c", "gray": "#8c8c8c", "silver": "#c0c0c0",
    "red": "#d62828", "orange": "#f77f00", "yellow": "#fcbf49", "green": "#2a9d4b", "blue": "#2563eb",
    "purple": "#7c3aed", "pink": "#ec4899", "brown": "#7a4b2a",
}

st.set_page_config(page_title="Vehicle Metadata Identification", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.hero {background: linear-gradient(120deg,#0f172a 0%,#1e3a8a 100%); color:#fff; padding:1.6rem 2rem;
       border-radius:16px; margin-bottom:1.2rem;}
.hero h1 {margin:0; font-size:2rem; color:#fff;}
.hero p {margin:.3rem 0 0; opacity:.8;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:14px; padding:.9rem 1.1rem; margin-bottom:.8rem;
       background: rgba(128,128,128,.06);}
.card h4 {margin:0 0 .5rem; font-size:1.05rem;}
.row {display:flex; justify-content:space-between; align-items:center; padding:.2rem 0; font-size:.92rem;}
.row .k {opacity:.65;}
.swatch {display:inline-block; width:14px; height:14px; border-radius:50%; margin-right:6px;
         border:1px solid rgba(128,128,128,.5); vertical-align:middle;}
.badge {font-size:.72rem; padding:.1rem .5rem; border-radius:999px; background:rgba(128,128,128,.2);}
.badge.warn {background:#f59e0b33; color:#b45309;}
.muted {opacity:.55; font-style:italic;}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>Vehicle Metadata Identification</h1>'
    "<p>Project #6 · 36127 · make, body type and colour from a single image</p></div>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_predictor(checkpoint: str, allow_synthetic: bool):
    from vehicle_id.attributes.modelling import AttributePredictor  # lazy: needs torch

    return AttributePredictor(checkpoint, allow_synthetic=allow_synthetic)


@st.cache_data(show_spinner=False)
def load_baseline_config() -> dict:
    return yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))


def parse_detections(text: str) -> list[dict]:
    payload = json.loads(text)
    detections = payload["detections"] if isinstance(payload, dict) else payload
    if not detections:
        raise ValueError("No detections found in JSON")
    return detections


def value_html(value, score=None) -> str:
    if not value:
        return '<span class="muted">not assessed</span>'
    dot = f'<span class="swatch" style="background:{SWATCH[value.lower()]}"></span>' if value.lower() in SWATCH else ""
    tail = f' <span class="badge">{score:.0%}</span>' if isinstance(score, float) else ""
    return f"{dot}<b>{value}</b>{tail}"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    with st.expander("Detector", expanded=True):
        detector_weights = st.text_input("YOLO weights path", value=str(REPO_ROOT / "models" / "yolo26n.pt"))
    with st.expander("Attribute models", expanded=False):
        checkpoints = {a: st.text_input(f"{a} checkpoint (.pt)", placeholder="optional") for a in ATTRIBUTES}
        allow_synthetic = st.checkbox("Allow synthetic smoke-test weights", value=False)
    st.caption("Scores are uncalibrated softmax values. Pilot or synthetic outputs are not validated accuracy.")

# ---------------------------------------------------------------- input
in_col, prev_col = st.columns([2, 3], gap="large")
sample_dir = REPO_ROOT / "data" / "samples"
samples = sorted(p.name for p in sample_dir.glob("*.jpg")) if sample_dir.is_dir() else []

with in_col:
    st.subheader("1 · Choose an image")
    uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    sample_choice = st.selectbox("or use a sample", ["(none)"] + samples)
    with st.expander("Use Stage 1 boxes instead of the detector"):
        boxes_text = st.text_area(
            "Detections JSON",
            height=110,
            placeholder='{"detections": [{"bbox_xyxy": [x1, y1, x2, y2], "confidence": 0.9}]}',
            label_visibility="collapsed",
        )
    run = st.button("Analyse", type="primary", width="stretch")

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".jpg") as tmp:
        tmp.write(uploaded.getvalue())
    image_path = Path(tmp.name)
elif sample_choice != "(none)":
    image_path = sample_dir / sample_choice
else:
    image_path = None

with prev_col:
    st.subheader("2 · Preview" if not run else "2 · Result")
    if image_path is None:
        st.info("Upload an image or pick a sample to begin.")
        st.stop()
    if not run:
        st.image(Image.open(image_path), width="stretch")
        st.stop()

# ---------------------------------------------------------------- analyse
try:
    predictors = {a: load_predictor(p.strip(), allow_synthetic) for a, p in checkpoints.items() if p.strip()}
    detections = parse_detections(boxes_text) if boxes_text.strip() else None
    with st.spinner("Running pipeline..."):
        result = analyse_vehicle_image(
            image_path, predictors, detections=detections,
            detector_weights=None if detections else detector_weights,
        )
except Exception as exc:  # pipeline raises explicit errors (missing weights, bad boxes, ...)
    st.error(f"{type(exc).__name__}: {exc}")
    st.stop()

vehicles = result["vehicles"]
colour_is_baseline = "colour" not in predictors
if colour_is_baseline and vehicles:
    cfg = load_baseline_config()["colour_baseline"]
    with Image.open(image_path) as src:
        rgb = src.convert("RGB")
    for rec in vehicles:
        est = estimate_colour(rgb.crop(tuple(rec["bbox_xyxy"])), cfg)
        prof = rec["vehicle_profile"]
        prof["colour"], prof["confidence"]["colour"] = est.label, est.confidence
        prof["status"]["colour"] = f"{est.status} (HSV baseline, unvalidated)"

with prev_col:
    st.image(result["annotated_image"], width="stretch")

# ---------------------------------------------------------------- results
st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Vehicles found", len(vehicles))
m2.metric("Avg detection conf.", f"{sum(v['detection_confidence'] for v in vehicles) / len(vehicles):.0%}" if vehicles else "–")
m3.metric("Attribute models", len(predictors) or "none (colour baseline)")

if not vehicles:
    st.warning("No vehicles detected.")
    st.stop()

tab_cards, tab_table, tab_json = st.tabs(["Vehicles", "Table", "Raw JSON"])

with tab_cards:
    cols = st.columns(3)
    for i, rec in enumerate(vehicles):
        prof, conf = rec["vehicle_profile"], rec["vehicle_profile"]["confidence"]
        with cols[i % 3]:
            st.markdown(
                f'<div class="card"><h4>{rec["vehicle_crop_id"].replace("_", " ").title()} '
                f'<span class="badge">det {rec["detection_confidence"]:.0%}</span></h4>'
                f'<div class="row"><span class="k">Make</span><span>{value_html(prof["make"], conf.get("make"))}</span></div>'
                f'<div class="row"><span class="k">Body type</span><span>{value_html(prof["body_type"], conf.get("body_type"))}</span></div>'
                f'<div class="row"><span class="k">Colour</span><span>{value_html(prof["colour"], conf.get("colour"))}'
                f'{" <span class=badge warn>baseline</span>" if colour_is_baseline else ""}</span></div>'
                f'<div class="row"><span class="k">Model / damage / accessories</span><span class="muted">not assessed</span></div>'
                "</div>",
                unsafe_allow_html=True,
            )

rows = [
    {
        "id": r["vehicle_crop_id"],
        "det_conf": round(r["detection_confidence"], 3),
        "make": r["vehicle_profile"]["make"],
        "body_type": r["vehicle_profile"]["body_type"],
        "colour": r["vehicle_profile"]["colour"],
        "make_score": r["vehicle_profile"]["confidence"].get("make"),
        "body_score": r["vehicle_profile"]["confidence"].get("body_type"),
        "colour_score": r["vehicle_profile"]["confidence"].get("colour"),
        "bbox_xyxy": r["bbox_xyxy"],
    }
    for r in vehicles
]
df = pd.DataFrame(rows)
with tab_table:
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "vehicle_results.csv", "text/csv")
with tab_json:
    payload = json.dumps({"vehicles": vehicles}, indent=2)
    st.download_button("Download JSON", payload, "vehicle_results.json", "application/json")
    st.code(payload, language="json")

st.caption(result["confidence_semantics"] + ". Colour baseline (when shown) is an unvalidated HSV heuristic.")

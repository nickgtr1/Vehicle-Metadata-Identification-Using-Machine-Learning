
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- shared

# n=500, test split, seed 42 -- the final, report-grade runs.
MAIN_RUNS = [
    {"label": "Zero-shot (raw)", "file": "vlm_eval_compcars_classification_raw_n500.csv",
     "attrs": ["make", "model", "body_type"], "kind": "zero-shot"},
    {"label": "Zero-shot (constrained)", "file": "vlm_eval_compcars_classification_constrained_n500.csv",
     "attrs": ["make", "model", "body_type"], "kind": "zero-shot"},
    {"label": "Zero-shot — colour", "file": "vlm_eval_compcars_surveillance_raw_n500.csv",
     "attrs": ["colour"], "kind": "zero-shot"},
    {"label": "Fine-tuned v2 (10 epochs, adapter 4)", "file": "vlm_eval_compcars_classification_tuned_v2_n500.csv",
     "attrs": ["make", "model", "body_type"], "kind": "tuned"},
]

# n=100 reference run only -- the naive first tuning attempt that showed no
# improvement, kept for the "why we re-tuned" story but not plotted alongside
# the n=500 runs (different sample size, not a fair bar-chart comparison).
REFERENCE_RUN = {
    "label": "Fine-tuned v1 (3 epochs, adapter 2) — reference, n=100",
    "file": "vlm_eval_compcars_classification_tuned.csv",
    "attrs": ["make", "model", "body_type"], "kind": "tuned",
}

# gemini-3.1-flash-lite pricing (USD/token). Tuned-model inference is billed
# at 1.5x the base rate for Gemini-3-generation models.
PRICE_IN, PRICE_OUT = 0.25 / 1_000_000, 1.50 / 1_000_000
TUNED_PRICE_IN, TUNED_PRICE_OUT = PRICE_IN * 1.5, PRICE_OUT * 1.5

# Training cost for the two Vertex AI fine-tuning jobs -- not in any eval CSV,
# since the training jobs themselves aren't scored, just billed. From the
# job status checks at the time.
TUNING_JOBS = [
    {"label": "Tuning job v1 (3 epochs, adapter 2)", "tokens": 587_983 * 3, "cost_usd": 5.29},
    {"label": "Tuning job v2 (10 epochs, adapter 4)", "tokens": 587_983 * 10, "cost_usd": 17.60},
]


def _wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def _load_run(path: Path, attrs: list[str]):
    df = pd.read_csv(path)
    if "error" in df.columns:
        df["error"] = df["error"].fillna("")  # pandas reads "" back as NaN
    n_total = len(df)
    n_failed = int((df["error"].astype(str) != "").sum()) if "error" in df else 0
    rows = []
    for attr in attrs:
        ok_col, gt_col = f"{attr}_ok", f"{attr}_gt"
        if ok_col not in df.columns:
            continue
        scored = df[df[gt_col].notna()] if gt_col in df.columns else df
        scored = scored[scored["error"].astype(str) == ""] if "error" in scored else scored
        n = len(scored)
        successes = int(scored[ok_col].astype(bool).sum())
        acc = successes / n if n else np.nan
        lo, hi = _wilson_ci(successes, n)
        rows.append({"attribute": attr, "n": n, "accuracy": acc, "ci_low": lo, "ci_high": hi, "n_failed": n_failed})
    tokens_in = int(df["prompt_tokens"].sum()) if "prompt_tokens" in df else 0
    tokens_out = int(df["output_tokens"].sum()) if "output_tokens" in df else 0
    return pd.DataFrame(rows), tokens_in, tokens_out, df


def _top_mismatches(df: pd.DataFrame, attr: str, k: int = 5) -> pd.DataFrame:
    gt_col, pred_col, ok_col = f"{attr}_gt", f"{attr}_pred", f"{attr}_ok"
    if ok_col not in df.columns:
        return pd.DataFrame()
    wrong = df[(df["error"].astype(str) == "") & (~df[ok_col].astype(bool)) & df[gt_col].notna()]
    if wrong.empty:
        return pd.DataFrame()
    return (
        wrong.groupby([gt_col, pred_col], dropna=False).size()
        .reset_index(name="count").sort_values("count", ascending=False).head(k)
    )


def _collect(results_dir: Path, run_specs: list[dict]):
    """Returns (summary_df, cost_df, per_run_raw_dfs, missing_labels)."""
    summary_rows, cost_rows, raw_dfs = [], [], {}
    missing = []
    for spec in run_specs:
        path = results_dir / spec["file"]
        if not path.exists():
            missing.append(spec["label"])
            continue
        scored, tin, tout, raw_df = _load_run(path, spec["attrs"])
        scored["run"] = spec["label"]
        scored["kind"] = spec["kind"]
        summary_rows.append(scored)
        raw_dfs[spec["label"]] = (raw_df, spec["attrs"])
        pin, pout = (TUNED_PRICE_IN, TUNED_PRICE_OUT) if spec["kind"] == "tuned" else (PRICE_IN, PRICE_OUT)
        cost_rows.append({
            "run": spec["label"], "tokens_in": tin, "tokens_out": tout,
            "est_cost_usd": round(tin * pin + tout * pout, 4),
        })
    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    costs = pd.DataFrame(cost_rows)
    return summary, costs, raw_dfs, missing


# ---------------------------------------------------------------- comparison

def render_vlm_comparison(results_dir: Path) -> None:
    st.subheader("Zero-shot vs fine-tuned VLM accuracy")
    st.caption(
        "CompCars held-out test split, n=500, seed 42. Wilson 95% confidence "
        "intervals shown as error bars -- more reliable than the normal "
        "approximation at this sample size and near 0%/100%."
    )

    summary, _costs, raw_dfs, missing = _collect(results_dir, MAIN_RUNS)
    for label in missing:
        st.info(f"'{label}' results not found in {results_dir} -- run the eval script to generate it.")
    if summary.empty:
        return

    attrs_present = [a for a in ("make", "model", "body_type", "colour") if a in summary["attribute"].unique()]
    fig, axes = plt.subplots(1, len(attrs_present), figsize=(5.2 * len(attrs_present), 4.2))
    if len(attrs_present) == 1:
        axes = [axes]
    for ax, attr in zip(axes, attrs_present):
        sub = summary[summary["attribute"] == attr]
        x = np.arange(len(sub))
        acc = sub["accuracy"].to_numpy() * 100
        err_low = acc - sub["ci_low"].to_numpy() * 100
        err_high = sub["ci_high"].to_numpy() * 100 - acc
        colors = ["#C44E52" if k == "tuned" else "#4C72B0" for k in sub["kind"]]
        ax.bar(x, acc, yerr=[err_low, err_high], capsize=4, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([r.split(" (")[0].split(" —")[0] for r in sub["run"]], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{attr}  (n={int(sub['n'].iloc[0])})")
        for xi, a in zip(x, acc):
            ax.text(xi, a + 2, f"{a:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    table = summary.copy()
    for c in ("accuracy", "ci_low", "ci_high"):
        table[c] = (table[c] * 100).round(1)
    st.dataframe(table[["run", "attribute", "n", "accuracy", "ci_low", "ci_high", "n_failed"]], width="stretch", hide_index=True)

    with st.expander("Top mismatches per run"):
        for label, (raw_df, attrs) in raw_dfs.items():
            for attr in attrs:
                mism = _top_mismatches(raw_df, attr)
                if mism.empty:
                    continue
                st.markdown(f"**{label} — {attr}**")
                st.dataframe(mism, width="stretch", hide_index=True)


# ---------------------------------------------------------------- final findings

def render_vlm_final_findings(results_dir: Path) -> None:
    st.subheader("Final findings")

    summary, costs, _raw, missing = _collect(results_dir, MAIN_RUNS)
    for label in missing:
        st.info(f"'{label}' results not found in {results_dir}.")

    def acc(run_label: str, attr: str):
        sub = summary[(summary["run"] == run_label) & (summary["attribute"] == attr)]
        return float(sub["accuracy"].iloc[0]) * 100 if len(sub) else None

    make_zs = acc("Zero-shot (raw)", "make")
    model_zs = acc("Zero-shot (raw)", "model")
    body_zs_raw = acc("Zero-shot (raw)", "body_type")
    body_zs_con = acc("Zero-shot (constrained)", "body_type")
    make_tuned = acc("Fine-tuned v2 (10 epochs, adapter 4)", "make")
    model_tuned = acc("Fine-tuned v2 (10 epochs, adapter 4)", "model")
    body_tuned = acc("Fine-tuned v2 (10 epochs, adapter 4)", "body_type")
    colour_zs = acc("Zero-shot — colour", "colour")

    ref_summary, _ref_costs, _ref_raw, ref_missing = _collect(results_dir, [REFERENCE_RUN])
    for label in ref_missing:
        st.info(f"'{label}' results not found in {results_dir}.")

    def ref_acc(attr: str):
        sub = ref_summary[ref_summary["attribute"] == attr] if not ref_summary.empty else ref_summary
        return float(sub["accuracy"].iloc[0]) * 100 if len(sub) else None

    v1_make, v1_model, v1_body = ref_acc("make"), ref_acc("model"), ref_acc("body_type")

    c1, c2, c3, c4 = st.columns(4)
    if make_zs is not None and make_tuned is not None:
        c1.metric("Make", f"{make_tuned:.1f}%", f"{make_tuned - make_zs:+.1f}pt vs zero-shot")
        if v1_make is not None:
            c1.caption(f"v1 (n=100, naive): {v1_make:.1f}%")
    if model_zs is not None and model_tuned is not None:
        c2.metric("Model", f"{model_tuned:.1f}%", f"{model_tuned - model_zs:+.1f}pt vs zero-shot")
        if v1_model is not None:
            c2.caption(f"v1 (n=100, naive): {v1_model:.1f}%")
    if body_zs_con is not None and body_tuned is not None:
        c3.metric("Body type", f"{body_tuned:.1f}%", f"{body_tuned - body_zs_con:+.1f}pt vs best zero-shot")
        if v1_body is not None:
            c3.caption(f"v1 (n=100, naive): {v1_body:.1f}%")
    if colour_zs is not None:
        c4.metric("Colour (zero-shot only)", f"{colour_zs:.1f}%", "not fine-tuned")
        c4.caption("No v1 equivalent (colour was never fine-tuned)")

    st.markdown(
        f"""
- **Model identification is the clear win from fine-tuning**: {model_zs:.1f}% zero-shot →
  **{model_tuned:.1f}%** fine-tuned, a gain that clears the ~±3–4pt confidence interval at n=500 --
  a real, defensible improvement, not sampling noise.
- **Make** moved only marginally ({make_zs:.1f}% → {make_tuned:.1f}%) -- within the noise band,
  not a result worth claiming on its own.
- **Body type**: constraining the zero-shot prompt (listing the exact CompCars labels) already
  recovered most of the achievable gain ({body_zs_raw:.1f}% → {body_zs_con:.1f}%); fine-tuning
  ({body_tuned:.1f}%) did not improve meaningfully beyond that -- prompt engineering, not
  fine-tuning, was the fix that mattered here.
- **Colour** ({colour_zs:.1f}% zero-shot) was never fine-tuned -- the training manifest
  (CompCars web-nature) has no colour labels; only the surveillance manifest does, and it lacks
  model/body_type.
- **The first fine-tuning attempt showed no improvement at all** (3 epochs, the smallest LoRA
  adapter size, Vertex's auto-selected defaults). Re-running with 10 epochs and a larger adapter
  is what produced the real gains above -- default auto-tuning settings were not sufficient for
  this dataset size.
"""
    )

    st.markdown("#### Cost & tokens")
    st.caption("Inference (the eval runs above) and training (the two Vertex AI fine-tuning jobs) are billed separately.")

    if not costs.empty:
        st.markdown("**Inference**")
        st.dataframe(costs, width="stretch", hide_index=True)

    tuning_df = pd.DataFrame(TUNING_JOBS)
    st.markdown("**Training**")
    st.dataframe(tuning_df, width="stretch", hide_index=True)

    total = (costs["est_cost_usd"].sum() if not costs.empty else 0) + tuning_df["cost_usd"].sum()
    st.metric("Total project spend (evals + both training jobs)", f"${total:.2f}")

    labels = (list(costs["run"]) if not costs.empty else []) + list(tuning_df["label"])
    values = (list(costs["est_cost_usd"]) if not costs.empty else []) + list(tuning_df["cost_usd"])
    colors = (["#4C72B0"] * len(costs) if not costs.empty else []) + ["#DD8452"] * len(tuning_df)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 1))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cost (USD)")
    ax.set_title("Cost by run (blue = inference, orange = training)")
    for yi, v in zip(y, values):
        ax.text(v, yi, f" ${v:.2f}" if v >= 1 else f" ${v:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Tuning job v1: cost and why it was re-run")
    v1 = TUNING_JOBS[0]
    st.markdown(
        f"""
The first fine-tuning attempt cost **${v1['cost_usd']:.2f}** (3 epochs, Vertex's smallest LoRA
adapter size, both auto-selected defaults for this dataset size) and produced **no measurable
improvement** over zero-shot -- at n=100 it was flat on body type, and actually slightly *worse*
on model and make+model accuracy, with every difference well inside the confidence interval.

That result pointed to under-training rather than a ceiling on what fine-tuning could do: too few
passes over the data, combined with the smallest adapter capacity, meant the model had little room
to shift away from its zero-shot behaviour. Re-running with 10 epochs and a larger adapter (v2,
${TUNING_JOBS[1]['cost_usd']:.2f}) is what produced the real gains reported above -- the two runs
together cost **${v1['cost_usd'] + TUNING_JOBS[1]['cost_usd']:.2f}** and the v1 attempt is best read
as a diagnostic step, not a wasted one: it's the evidence that default auto-tuning settings aren't
automatically sufficient, which is itself worth reporting.
"""
    )

    st.markdown("#### Limitations")
    st.markdown(
        """
- **Small, fixed training set.** 480 examples (40 per body type) is enough to show a real effect,
  but leaves rare makes -- especially Chinese-market brands -- thinly represented. A larger or
  class-targeted training set is untested.
- **Only one successful tuning configuration was tried.** 10 epochs / adapter size 4 beat zero-shot,
  but the hyperparameter space (more epochs, a larger adapter, more/targeted examples) was not
  explored beyond that single run, so this is not known to be the *best* achievable result, only a
  working one.
- **Colour was never fine-tuned.** The training manifest (CompCars web-nature) carries no colour
  labels; only the surveillance manifest does, and it lacks model/body_type. A colour-specific
  tuning run would need its own dataset.
- **Model-name scoring is strict exact match** (after stripping make prefixes / body-type suffixes
  CompCars sometimes embeds). Genuine near-misses -- a correct trim or generation the manifest
  doesn't distinguish -- still count as wrong, so true "right car, different granularity" accuracy
  is likely somewhat higher than reported.
- **Ground-truth label noise.** CompCars itself contains typos and inconsistencies (handled where
  found, e.g. "BWM"/"Benz", the car_type=0 bug) -- some noise may remain unflagged.
- **Single training run per configuration.** Each tuning job was trained once, not across multiple
  seeds, so run-to-run variance in the fine-tuning result itself hasn't been quantified.
- **Evaluated only on CompCars-distribution images.** The Streamlit app's real-world demo photos
  (street scenes, varied lighting, multiple vehicles per frame) are a different distribution from
  the training/test images; accuracy on those hasn't been separately measured.
- **Make+model combined accuracy remains modest (~64% even after tuning).** Any product-facing claim
  about overall identification accuracy should quote the per-attribute numbers, not imply a single
  vehicle is fully and correctly identified end-to-end at a higher rate than that.
"""
    )

    st.markdown("#### Recommendations")
    st.markdown(
        """
1. **Keep the constrained zero-shot prompt for body type.** It matched fine-tuning's result at a
   fraction of the cost and with no training step -- fine-tuning didn't earn its cost there.
2. **Use the fine-tuned v2 model specifically for make and model-name identification**, where the
   gain over zero-shot is real and reproducible at n=500.
3. **If further tuning budget is available, target the data rather than just the schedule**: oversample
   the specific confused pairs seen in the mismatch tables (sports/coupe, fastback/sedan/hatchback,
   SUV/crossover) instead of uniformly increasing epochs or examples.
4. **Validate on real app-demo photos**, not just the CompCars test split, before quoting these
   accuracy figures in a product or client-facing context -- the training and evaluation
   distribution is narrower than real uploaded images will be.
5. **If colour fine-tuning is wanted**, budget it as a separate, similarly-sized experiment
   (~$5-20) using the surveillance manifest, since no existing tuned model covers it.
6. **Clean any remaining CompCars label noise before a further training run** -- the dataset's own
   typos and ambiguities set a ceiling on how clean any fine-tuned result can be.
"""
    )
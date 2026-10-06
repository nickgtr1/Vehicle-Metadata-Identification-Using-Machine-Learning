import argparse
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

from vehicle_id.vlm.client import predict_profile, predict_profile_tuned

# gemini-3.1-flash-lite, USD per token (check the pricing page if this changes)
PRICE_IN = 0.25 / 1_000_000
PRICE_OUT = 1.50 / 1_000_000

# Map VLM wording onto the CompCars label vocabulary (normalised: lowercase, letters/digits only).
# Only unambiguous wording differences are mapped; real confusions (e.g. sedan vs fastback,
# convertible vs hardtop convertible) should stay as errors.
MAKE_ALIASES = {
    "vw": "volkswagen",
    "mercedes": "mercedesbenz",
    "chevy": "chevrolet",
    # CompCars label typos / abbreviations (ground-truth side)
    "bwm": "bmw",
    "benz": "mercedesbenz",
    "buck": "buick",
    "chrey": "chery",
    "lamorghini": "lamborghini",
    # Chinese-market brands: CompCars uses pinyin names, a VLM will use English/brand names
    "yiqi": "faw",
    "riich": "ruiqi",
    "jac": "jianghuai",
    "ssangyong": "shuanglong",
    "zotye": "zoyte",
    "maxus": "shangqidatong",
    "saicmaxus": "shangqidatong",
    "trumpchi": "guangqichuanqi",
    "gactrumpchi": "guangqichuanqi",
    "bestune": "besturn",
    "fawbesturn": "besturn",
    "dongfengvenucia": "venucia",
    "greatwallmotors": "greatwall",
    # Other naming variants
    "minicooper": "mini",
    "dsautomobiles": "ds",
}
BODY_ALIASES = {
    "wagon": "estate",
    "stationwagon": "estate",
    "hatch": "hatchback",
    "pickuptruck": "pickup",
    "ute": "pickup",
    "minivan": "mpv",
    "sportscar": "sports",
    "cabriolet": "convertible",
    "roadster": "convertible",
}
# CompCars colours: black, silver, white, red, blue, brown, yellow, champagne, green, purple
# (there is no 'grey' or 'gold' label, so those map to silver / champagne).
COLOUR_ALIASES = {
    "gray": "silver",
    "grey": "silver",
    "gold": "champagne",
    "beige": "champagne",
    "tan": "champagne",
    "cream": "champagne",
    "maroon": "red",
    "burgundy": "red",
    "navy": "blue",
}


def _norm(value) -> str | None:
    """Lowercase and strip everything except letters/digits. None for missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]", "", text.lower())
    return text or None


def match_make(gt, pred) -> bool:
    g, p = _norm(gt), _norm(pred)
    return p is not None and MAKE_ALIASES.get(g, g) == MAKE_ALIASES.get(p, p)


def match_body(gt, pred) -> bool:
    g, p = _norm(gt), _norm(pred)
    return p is not None and BODY_ALIASES.get(g, g) == BODY_ALIASES.get(p, p)


def match_colour(gt, pred) -> bool:
    """Match if the ground-truth colour appears as a word in the prediction
    (so 'blue and white' still counts for 'blue')."""
    if _norm(gt) is None or pred is None:
        return False
    words = [COLOUR_ALIASES.get(w, w) for w in re.findall(r"[a-z]+", str(pred).lower())]
    g = COLOUR_ALIASES.get(str(gt).lower().strip(), str(gt).lower().strip())
    return g in words


_BODY_SUFFIXES = {
    "hatchback", "sedan", "suv", "coupe", "estate", "convertible", "mpv",
    "pickup", "fastback", "sports", "crossover", "minibus", "wagon",
}


def match_model(gt, pred, make_gt=None) -> bool:
    """Exact match on normalised model name, after stripping a make prefix and/or
    body-type suffix CompCars sometimes embeds in model_name (e.g. 'Audi Q5' -> 'Q5',
    'AVEO hatchback' -> 'Aveo'). No fuzzy matching beyond that: with ~1,700 distinct
    CompCars models, trim/generation naming still varies too much to hand-map."""
    g, p = _norm(gt), _norm(pred)
    if p is None:
        return False
    if g == p:
        return True
    if g and make_gt:
        mk = _norm(make_gt)
        if mk and g.startswith(mk):
            g = g[len(mk):]
    if g:
        for suffix in _BODY_SUFFIXES:
            if g.endswith(suffix) and len(g) > len(suffix):
                g = g[: -len(suffix)]
                break
    return g == p


def call_with_retry(call_fn, path: str, retries: int = 5):
    """Retry rate-limit / transient errors with exponential backoff (5s, 10s, 20s, 40s).
    call_fn is a zero-extra-arg wrapper around predict_profile / predict_profile_tuned,
    so this stays agnostic to which model is being called."""
    for attempt in range(retries):
        try:
            return call_fn(path)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            transient = any(k in msg for k in ("429", "quota", "resource", "503", "unavailable"))
            if transient and attempt < retries - 1:
                time.sleep(min(60, 5 * 2**attempt))
                continue
            raise


def resolve_image(image_root: Path | None, rel: str) -> Path:
    """Manifest paths may be relative to the notebook dir (../../data/...).
    Try as-is, then with leading '..' stripped (repo-root relative), then under image_root."""
    p = Path(str(rel))
    if p.exists():
        return p
    parts = list(p.parts)
    while parts and parts[0] == "..":
        parts.pop(0)
    stripped = Path(*parts) if parts else p
    if stripped.exists():
        return stripped
    return image_root / p if image_root else stripped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", default=None, help="optional; only needed if manifest paths don't resolve on their own")
    ap.add_argument("--path-col", required=True, help="column holding the image path")
    ap.add_argument("--make-col")
    ap.add_argument("--model-col")
    ap.add_argument("--body-col")
    ap.add_argument("--colour-col")
    ap.add_argument("--split-col")
    ap.add_argument("--split-value")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls (helps on free-tier rate limits)")
    ap.add_argument("--model", default="zero-shot", choices=["zero-shot", "tuned"], help="which client.py function to call")
    ap.add_argument("--prompt", default="raw", choices=["raw", "constrained"], help="prompt variant (zero-shot only; ignored for --model tuned, which always uses its trained prompt)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.model == "tuned":
        call_fn = lambda path: predict_profile_tuned(path)  # noqa: E731
    else:
        call_fn = lambda path: predict_profile(path, prompt=args.prompt)  # noqa: E731

    df = pd.read_csv(args.manifest)
    wanted = {
        "make": args.make_col,
        "model": args.model_col,
        "body_type": args.body_col,
        "colour": args.colour_col,
    }
    wanted = {attr: col for attr, col in wanted.items() if col}
    if not wanted:
        raise SystemExit("Pass at least one of --make-col / --body-col / --colour-col.")

    if args.model == "tuned" and "colour" in wanted:
        print("  [note] --model tuned never returns a colour (training data had no colour labels) — dropping colour from scoring.")
        del wanted["colour"]
        if not wanted:
            raise SystemExit("Nothing left to score: --model tuned only supports make/model/body_type.")

    needed = [args.path_col, *wanted.values()] + ([args.split_col] if args.split_col else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns not found in manifest: {missing}\nAvailable columns: {list(df.columns)}")

    if args.split_col:
        df = df[df[args.split_col].astype(str) == str(args.split_value)]
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    image_root = Path(args.image_root) if args.image_root else None
    first = resolve_image(image_root, df.iloc[0][args.path_col])
    if not first.exists():
        raise SystemExit(
            f"Image not found: {first}\n"
            "Run from the repo root, or pass --image-root pointing at the folder the manifest paths are relative to."
        )

    matchers = {"make": match_make, "model": match_model, "body_type": match_body, "colour": match_colour}
    rows = []
    for i, row in df.iterrows():
        img = resolve_image(image_root, row[args.path_col])
        rec = {"image": str(img)}
        t0 = time.time()
        try:
            profile, usage = call_with_retry(call_fn, str(img))
            rec["error"] = ""
            rec["prompt_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
            # thinking tokens (if any) are billed as output
            rec["output_tokens"] = (getattr(usage, "candidates_token_count", 0) or 0) + (
                getattr(usage, "thoughts_token_count", 0) or 0
            )
            for attr, col in wanted.items():
                pred = getattr(profile, attr, None)
                gt = row[col]
                rec[f"{attr}_gt"] = gt
                rec[f"{attr}_pred"] = pred
                if attr == "model" and "make" in wanted:
                    rec[f"{attr}_ok"] = matchers[attr](gt, pred, make_gt=row[wanted["make"]])
                else:
                    rec[f"{attr}_ok"] = matchers[attr](gt, pred)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
            rec["prompt_tokens"] = rec["output_tokens"] = 0
            for attr, col in wanted.items():
                rec[f"{attr}_gt"] = row[col]
        rec["seconds"] = round(time.time() - t0, 2)
        rows.append(rec)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(df)} done")
        if args.sleep:
            time.sleep(args.sleep)

    res = pd.DataFrame(rows)
    variant = "tuned" if args.model == "tuned" else args.prompt
    out = args.out or f"vlm_eval_{Path(args.manifest).stem}_{variant}.csv"
    res.to_csv(out, index=False)

    ok = res[res["error"] == ""]
    label = "Tuned model" if args.model == "tuned" else "Zero-shot baseline"
    print(f"\n=== {label}: {len(res)} images, {len(res) - len(ok)} failed (API/JSON errors) ===")
    for attr, col in wanted.items():
        scored = ok[ok[f"{attr}_gt"].notna()]
        if len(scored):
            acc = scored[f"{attr}_ok"].mean()
            null_rate = scored[f"{attr}_pred"].isna().mean()
            print(f"{attr:10s} accuracy {acc:6.1%}  (null predictions {null_rate:5.1%}, n={len(scored)})")
            wrong = scored[~scored[f"{attr}_ok"].astype(bool)]
            if len(wrong):
                pairs = (
                    wrong.groupby([f"{attr}_gt", f"{attr}_pred"], dropna=False)
                    .size()
                    .sort_values(ascending=False)
                    .head(5)
                )
                print("    top mismatches (truth -> predicted): " + "; ".join(f"{g} -> {p} x{n}" for (g, p), n in pairs.items()))

    if "make" in wanted and "model" in wanted:
        both = ok[ok["make_gt"].notna() & ok["model_gt"].notna()]
        if len(both):
            combined_ok = both["make_ok"].astype(bool) & both["model_ok"].astype(bool)
            print(f"{'make+model':10s} accuracy {combined_ok.mean():6.1%}  (both correct, n={len(both)})")

    cost = res["prompt_tokens"].sum() * PRICE_IN + res["output_tokens"].sum() * PRICE_OUT
    print(f"tokens: {int(res['prompt_tokens'].sum())} in / {int(res['output_tokens'].sum())} out  ~ ${cost:.4f}")
    print(f"per-image results: {out}")


if __name__ == "__main__":
    main()
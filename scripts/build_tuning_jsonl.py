import argparse
import json
import mimetypes
import shutil
from pathlib import Path

import pandas as pd

PROMPT = """Look at this vehicle image and identify:
- make (manufacturer)
- model
- body type: must be exactly one of sedan, SUV, hatchback, sports, MPV, fastback, estate, hardtop convertible, minibus, convertible, pickup, crossover

Respond ONLY in this exact JSON format:
{"make": "...", "model": "...", "body_type": "..."}
"""


MAKE_FIXES = {
    "BWM": "BMW",
    "Benz": "Mercedes-Benz",
    "Buck": "Buick",
    "Chrey": "Chery",
    "Lamorghini": "Lamborghini",
}

# CompCars sometimes appends the body type to model_name (e.g. "AVEO hatchback").
_BODY_WORDS = {
    "hatchback", "sedan", "suv", "coupe", "couple", "estate", "convertible",
    "mpv", "pickup", "fastback", "sports", "crossover", "minibus", "wagon",
}

TOKENS_PER_EXAMPLE = 1200  # measured ~1,160 input tokens per image call + short answer


def clean_make(make_raw) -> str:
    make = str(make_raw).strip()
    return MAKE_FIXES.get(make, make)


def clean_model(model_raw, make_raw) -> str:
    """Drop a leading make prefix and a trailing body-type word from model_name.
    'Audi Q5' -> 'Q5', 'AVEO hatchback' -> 'AVEO', 'BWM 5 Series GT' -> '5 Series GT'."""
    model = str(model_raw).strip()
    make = str(make_raw).strip()
    if make and model.lower().startswith(make.lower()):
        rest = model[len(make):].strip(" -")
        if rest:
            model = rest
    words = model.split()
    if len(words) > 1 and words[-1].lower() in _BODY_WORDS:
        model = " ".join(words[:-1])
    return model


def resolve_image(raw: str) -> Path:
    """Manifest paths look like ..\\..\\data\\raw\\... (relative to notebooks/vlm/).
    Try as-is, then with leading '..' stripped (repo-root relative)."""
    p = Path(str(raw).replace("\\", "/"))
    if p.exists():
        return p
    parts = list(p.parts)
    while parts and parts[0] == "..":
        parts.pop(0)
    return Path(*parts) if parts else p


def sample_per_class(df: pd.DataFrame, col: str, n: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(n=min(n, len(g)), random_state=seed) for _, g in df.groupby(col)]
    return pd.concat(parts) if parts else df.iloc[0:0]


def build_examples(df, prefix, out_images: Path, bucket_uri: str):
    lines, staged = [], 0
    for i, (_, row) in enumerate(df.iterrows()):
        src = resolve_image(row["image_path"])
        if not src.exists():
            print(f"  [skip] image not found: {src}")
            continue
        ext = src.suffix.lower() or ".jpg"
        name = f"{prefix}_{i:04d}{ext}"
        shutil.copy2(src, out_images / name)
        staged += 1
        mime = mimetypes.guess_type(name)[0] or "image/jpeg"
        make = clean_make(row["make_name"])
        target = {
            "make": make,
            "model": clean_model(row["model_name"], row["make_name"]),
            "body_type": str(row["car_type_name"]).strip(),
        }
        example = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"fileData": {"mimeType": mime, "fileUri": f"{bucket_uri}/images/{name}"}},
                        {"text": PROMPT},
                    ],
                },
                {"role": "model", "parts": [{"text": json.dumps(target, ensure_ascii=False)}]},
            ]
        }
        lines.append(json.dumps(example, ensure_ascii=False))
    return lines, staged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bucket-uri", required=True, help="e.g. gs://my-bucket/vehicle-id (no trailing slash needed)")
    ap.add_argument("--out-dir", default="data/tuning")
    ap.add_argument("--per-body", type=int, default=40, help="max training examples per body type")
    ap.add_argument("--val-per-body", type=int, default=5, help="max validation examples per body type")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--train-value", default="train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bucket_uri = args.bucket_uri.rstrip("/")
    if not bucket_uri.startswith("gs://"):
        raise SystemExit("--bucket-uri must start with gs://")

    df = pd.read_csv(args.manifest)
    needed = ["image_path", "make_name", "model_name", "car_type_name", args.split_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns not found: {missing}\nAvailable: {list(df.columns)}")

    train_pool = df[df[args.split_col].astype(str) == args.train_value]
    train_pool = train_pool.dropna(subset=["make_name", "model_name", "car_type_name"])
    print(f"train-split rows with complete labels: {len(train_pool)}")

    train_df = sample_per_class(train_pool, "car_type_name", args.per_body, args.seed)
    val_pool = train_pool.drop(train_df.index)
    val_df = sample_per_class(val_pool, "car_type_name", args.val_per_body, args.seed)

    out_dir = Path(args.out_dir)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    train_lines, n_train = build_examples(train_df.sample(frac=1, random_state=args.seed), "train", out_images, bucket_uri)
    val_lines, n_val = build_examples(val_df.sample(frac=1, random_state=args.seed), "val", out_images, bucket_uri)

    (out_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (out_dir / "val.jsonl").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"\ntrain examples: {len(train_lines)}   val examples: {len(val_lines)}")
    print("train examples per body type:")
    print(train_df["car_type_name"].value_counts().to_string())
    est = len(train_lines) * TOKENS_PER_EXAMPLE
    print(f"\nrough training size: ~{est:,} tokens per epoch (x number of epochs = billed training tokens)")
    print("\nNext steps:")
    print(f"  gcloud storage cp -r {out_dir / 'images'} {bucket_uri}/")
    print(f"  gcloud storage cp {out_dir / 'train.jsonl'} {out_dir / 'val.jsonl'} {bucket_uri}/")


if __name__ == "__main__":
    main()

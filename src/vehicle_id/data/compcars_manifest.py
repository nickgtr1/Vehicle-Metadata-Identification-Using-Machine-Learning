import os
import pandas as pd

def load_bbox(label_path: str):
    with open(label_path) as f:
        lines = f.read().splitlines()
    viewpoint = int(lines[0])
    car_type = int(lines[1])
    x1, y1, x2, y2 = map(int, lines[2].split())
    return (x1, y1, x2, y2), viewpoint, car_type

def build_manifest(
    image_root: str = "data/raw/compcars/image",
    label_root: str = "data/raw/compcars/label",
) -> pd.DataFrame:
    records = []
    for make_id in os.listdir(image_root):
        for model_id in os.listdir(os.path.join(image_root, make_id)):
            for year in os.listdir(os.path.join(image_root, make_id, model_id)):
                year_dir = os.path.join(image_root, make_id, model_id, year)
                for fname in os.listdir(year_dir):
                    image_path = os.path.join(year_dir, fname)
                    label_path = os.path.join(
                        label_root, make_id, model_id, year,
                        fname.replace(".jpg", ".txt")
                    )
                    bbox, viewpoint, car_type = (None, None, None)
                    if os.path.exists(label_path):
                        bbox, viewpoint, car_type = load_bbox(label_path)
                    records.append({
                        "image_path": image_path,
                        "make_id": make_id,
                        "model_id": model_id,
                        "year": year,
                        "bbox": bbox,
                        "viewpoint": viewpoint,
                        "car_type": car_type,  
                    })
    df = pd.DataFrame(records)

  
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    attrs_path = os.path.join(repo_root, "data", "raw", "compcars", "misc", "attributes.txt")
    attrs = pd.read_csv(attrs_path, sep=r"\s+", header=0)

    df["model_id"] = df["model_id"].astype(int)
    attrs["model_id"] = attrs["model_id"].astype(int)
    df = df.merge(attrs[["model_id", "type"]], on="model_id", how="left")

    manifest_dir = os.path.join(repo_root, "data", "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    df.to_csv(os.path.join(manifest_dir, "compcars_manifest.csv"), index=False)

    return df
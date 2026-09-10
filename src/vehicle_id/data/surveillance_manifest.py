import os
import pandas as pd
from scipy.io import loadmat

def build_surveillance_manifest(
    sv_root: str = "data/raw/compcars/sv_data"
) -> pd.DataFrame:
    image_root = os.path.join(sv_root, "image")

    color_data = loadmat(os.path.join(sv_root, "color_list.mat"))["color_list"]
    name_data = loadmat(os.path.join(sv_root, "sv_make_model_name.mat"))["sv_make_model_name"]

    COLOR_MAP = {
        -1: "unrecognized", 0: "black", 1: "white", 2: "red", 3: "yellow",
        4: "blue", 5: "green", 6: "purple", 7: "brown", 8: "champagne", 9: "silver",
    }

    
    color_lookup = {}
    for row in color_data:
        path = str(row[0][0])
        color_id = int(row[1][0][0])
        color_lookup[path] = color_id

   
    def resolve_make_model(surveillance_model_id: int):
        idx = surveillance_model_id - 1
        make = str(name_data[idx][0][0])
        model = str(name_data[idx][1][0])
        web_model_id = int(name_data[idx][2][0][0])
        return make, model, web_model_id

   
    def load_split_set(path):
        with open(path) as f:
            return set(line.strip() for line in f if line.strip())

    train_set = load_split_set(os.path.join(sv_root, "train_surveillance.txt"))
    test_set = load_split_set(os.path.join(sv_root, "test_surveillance.txt"))

   
    records = []
    for surveillance_model_id_str in os.listdir(image_root):
        surveillance_model_id = int(surveillance_model_id_str)
        model_dir = os.path.join(image_root, surveillance_model_id_str)
        make, model, web_model_id = resolve_make_model(surveillance_model_id)

        for fname in os.listdir(model_dir):
            image_path = os.path.join(model_dir, fname)
            relative_key = f"{surveillance_model_id_str}/{fname}"

            color_id = color_lookup.get(relative_key, None)
            color_name = COLOR_MAP.get(color_id, None) if color_id is not None else None

            split = "train" if relative_key in train_set else ("test" if relative_key in test_set else None)

            records.append({
                "image_path": image_path,
                "surveillance_model_id": surveillance_model_id,
                "make_name": make,
                "model_name": model,
                "web_nature_model_id": web_model_id,
                "color_id": color_id,
                "color_name": color_name,
                "split": split,
            })

    df = pd.DataFrame(records)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    manifest_dir = os.path.join(repo_root, "data", "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    df.to_csv(os.path.join(manifest_dir, "compcars_surveillance_manifest.csv"), index=False)

    return df
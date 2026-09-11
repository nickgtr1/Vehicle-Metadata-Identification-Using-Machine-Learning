import ast
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def resolve_image_path(value, root):
    text = str(value).replace('\\', '/')
    if text.startswith('../../data/'):
        text = text[6:]
    path = Path(text)
    if path.is_absolute() or ':' in text:
        raise ValueError('Manifest paths must be repository-relative')
    root = Path(root).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Image path escapes repository root')
    return resolved


def parse_bbox(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    box = ast.literal_eval(value) if isinstance(value, str) else value
    a = np.asarray(box, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError('bbox must contain four finite coordinates')
    x1, y1, x2, y2 = a
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError('Invalid bbox extent')
    return tuple(int(round(v)) for v in a)


class CompCarsMakeDataset(Dataset):
    

    def __init__(self, df, label_col="make_name", image_size=224, transform=None, root=None, classes=None):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.image_size = image_size
        self.transform = transform

        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        if self.df[label_col].isna().any():
            raise ValueError('Missing labels')
        self.classes = list(classes) if classes is not None else sorted(self.df[label_col].unique())
        if len(set(self.classes)) != len(self.classes):
            raise ValueError('Duplicate classes')
        if set(self.df[label_col]) - set(self.classes):
            raise ValueError('Labels outside training class mapping')
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(row['image_path'], self.root)

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox = parse_bbox(row.get('bbox'))
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            image = image[min(y1,h):min(y2,h), min(x1,w):min(x2,w)]
            if image.size == 0:
                raise ValueError(f'Empty crop: {image_path}')

        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = self.class_to_idx[row[self.label_col]]
        return image, label

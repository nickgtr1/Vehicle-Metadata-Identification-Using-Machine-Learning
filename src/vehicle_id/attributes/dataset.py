import ast
import cv2
import torch
from torch.utils.data import Dataset


class CompCarsMakeDataset(Dataset):
    

    def __init__(self, df, label_col="make_name", image_size=224, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.image_size = image_size
        self.transform = transform

        self.classes = sorted(self.df[label_col].dropna().unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"].replace("\\", "/")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox = row.get("bbox", None)
        if bbox is not None and not (isinstance(bbox, float)):  # handles NaN bbox gracefully
            bbox = ast.literal_eval(bbox) if isinstance(bbox, str) else bbox
            x1, y1, x2, y2 = bbox
            image = image[y1:y2, x1:x2]

        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = self.class_to_idx[row[self.label_col]]
        return image, label
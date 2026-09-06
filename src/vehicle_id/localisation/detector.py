#detector for localisation

import os
from ultralytics import YOLO

_models = {}
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")

def load_model(weights: str = "yolo26n.pt"):
    weights_path = os.path.join(MODELS_DIR, weights)
    if weights not in _models:
        _models[weights] = YOLO(weights_path)
    return _models[weights]

def detect_vehicles(image_path: str, weights: str = "yolo26n.pt") -> list[dict]:
    model = load_model(weights)

    results = model(image_path)
    vehicle_boxes = []
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls)]
        if cls_name in ("car", "truck", "bus"):  # COCO vehicle classes
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            vehicle_boxes.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf)
            })
    return vehicle_boxes
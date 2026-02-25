import torch
import torchreid
import cv2
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

model.eval()
model.to(device)

transform = torchreid.data.transforms.build_transforms(
    height=256,
    width=128,
    is_train=False
)[0]


def extract_embedding(image, box):
    x1, y1, x2, y2 = box

    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = Image.fromarray(crop)

    crop = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(crop)

    return embedding.cpu().numpy().flatten()
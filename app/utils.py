import base64
import cv2
import numpy as np

def read_image(file):
    contents = file.file.read()

    if not contents:
        raise ValueError("Uploaded file is empty")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image file")

    return img


def encode_crop(image, box):
    x1, y1, x2, y2 = box

    # Clamp coordinates safely
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    _, buffer = cv2.imencode(".jpg", crop)
    return base64.b64encode(buffer).decode("utf-8")
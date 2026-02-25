from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")  # lightweight model

def detect_persons(image):
    results = model(image)[0]
    boxes = []

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box.tolist())
            boxes.append((x1, y1, x2, y2))

    return boxes
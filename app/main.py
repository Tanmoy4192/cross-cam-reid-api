from fastapi import FastAPI, UploadFile, File, HTTPException
from app.detector import detect_persons
from app.reid import extract_embedding
from app.matcher import match_embeddings
from app.utils import read_image
import cv2

app = FastAPI(title="Cross-Image Person ReID API")

ALLOWED_TYPES = ["image/jpeg", "image/png"]
MAX_IMAGE_SIZE = 1280


# ---------------- VALIDATION ---------------- #

def validate_image(upload_file: UploadFile):
    if upload_file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only JPG and PNG allowed."
        )


def resize_if_large(image):
    h, w = image.shape[:2]
    if max(h, w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


# ---------------- MAIN ENDPOINT ---------------- #

@app.post("/compare")
async def compare(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    try:
        # -------- Validate --------
        validate_image(image1)
        validate_image(image2)

        # -------- Read --------
        img1 = read_image(image1)
        image1.file.seek(0)

        img2 = read_image(image2)
        image2.file.seek(0)

        if img1 is None or img2 is None:
            raise HTTPException(status_code=400, detail="Invalid image input.")

        # -------- Resize --------
        img1 = resize_if_large(img1)
        img2 = resize_if_large(img2)

        # -------- Detect --------
        boxes1 = detect_persons(img1)
        boxes2 = detect_persons(img2)

        # -------- Extract Embeddings --------
        emb1, valid_boxes1 = [], []
        for b in boxes1:
            emb = extract_embedding(img1, b)
            if emb is not None:
                emb1.append(emb)
                valid_boxes1.append(b)

        emb2, valid_boxes2 = [], []
        for b in boxes2:
            emb = extract_embedding(img2, b)
            if emb is not None:
                emb2.append(emb)
                valid_boxes2.append(b)

        boxes1 = valid_boxes1
        boxes2 = valid_boxes2

        # -------- Matching --------
        matches, unmatched = match_embeddings(emb1, emb2)

        response = {
            "image1_person_count": int(len(boxes1)),
            "image2_person_count": int(len(boxes2)),
            "matches": [],
            "unmatched_image1": [],
            "unmatched_image2": []
        }

        # -------- Format Matches --------
        for i, j, score in matches:
            if i < len(boxes1) and j < len(boxes2):

                x1_1, y1_1, x2_1, y2_1 = boxes1[i]
                x1_2, y1_2, x2_2, y2_2 = boxes2[j]

                response["matches"].append({
                    "image1_index": int(i),
                    "image2_index": int(j),
                    "similarity": float(score),
                    "bbox_image1": [
                        int(x1_1), int(y1_1), int(x2_1), int(y2_1)
                    ],
                    "bbox_image2": [
                        int(x1_2), int(y1_2), int(x2_2), int(y2_2)
                    ]
                })

        # -------- Format Unmatched --------
        for source, idx in unmatched:
            if source == "image1":
                response["unmatched_image1"].append(int(idx))
            elif source == "image2":
                response["unmatched_image2"].append(int(idx))

        return response

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
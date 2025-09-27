from fastapi import APIRouter, File, UploadFile, HTTPException, Request
import cv2
import numpy as np
from ..features.detect import run_detections, draw_boxes
# from ..core.model_loader import load_detection_model as detection_session
from contextlib import asynccontextmanager
import onnxruntime as ort

# detection_session: ort.InferenceSession = None

# @asynccontextmanager
# async def lifespan(app: APIRouter):
#     global detection_session
#     detection_session = load_detection_model(use_gpu=True)
#     print("Detection model loaded and ready.")
#     yield

router = APIRouter()

@router.post("/detect")
async def detect_objects(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return {"error": "Invalid file type. Please upload a PNG or JPG image."}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Could not read the image. Please ensure the file is a valid image."}

    detection_session = request.app.state.detection_session
    boxes, scores, class_ids = run_detections(detection_session, image)

    annotated_image = draw_boxes(image.copy(), boxes, scores, class_ids)

    _, img_encoded = cv2.imencode('.jpg', annotated_image)
    img_bytes = img_encoded.tobytes()

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "detections": [
            {"box": box.tolist(), "score": float(score), "class_id": int(class_id)}
            for box, score, class_id in zip(boxes, scores, class_ids)
        ],
        "annotated_image": img_bytes
    }
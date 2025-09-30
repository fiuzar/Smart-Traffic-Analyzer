import base64
from fastapi import APIRouter, File, UploadFile, Request
import cv2
import numpy as np
from ..features.detect import run_detections, draw_boxes
import onnxruntime as ort

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

    detection_session: ort.InferenceSession = request.app.state.detection_session
    detections = run_detections(detection_session, image)

    # Draw bounding boxes
    annotated_image = draw_boxes(image.copy(), detections)

    # Encode image to base64 for JSON-safe transfer
    _, img_encoded = cv2.imencode('.jpg', annotated_image)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "detections": detections,
        "annotated_image": img_base64  # JSON-safe string
    }

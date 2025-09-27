from fastapi import APIRouter, UploadFile, File, Request
import cv2
import numpy as np
from ..features.detect import run_detections

router = APIRouter()

@router.post("/count")
async def count_vehicles(request: Request, file: UploadFile = File(...)):
    """
    Count vehicles in the uploaded image.

    Args:
        request (Request): FastAPI request object to access app state.
        file (UploadFile): Uploaded image file.
    """

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    detection_session = request.app.state.detection_session
    detections = run_detections(detection_session, image)

    vehicle_classes = {2, 3, 4, 5}

    vehicle_detection = [d for d in detections if d['class_id'] in vehicle_classes]

    return {"vehicle_count": len(vehicle_detection), "detections": vehicle_detection}
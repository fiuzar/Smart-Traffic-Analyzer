import cv2
import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from typing import List

from ..features.detect import run_detections

router = APIRouter()

def detect_violation(detections: List[dict], segmentation_mask: np.ndarray) -> List[dict]:
    """
    Detect traffic violations based on vehicle detections and road segmentation mask.

    Args:
        detections (List[dict]): List of vehicle detections with bounding boxes and class IDs.
        segmentation_mask (np.ndarray): Segmentation mask where road pixels are marked.

    Returns:
        List[dict]: List of detected violations with details.
    """
    violations = []
    road_pixels = np.where(segmentation_mask == 1)  # Assuming road class is labeled as 1

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        vehicle_area = segmentation_mask[y1:y2, x1:x2]
        if np.any(vehicle_area == 0):  # Assuming non-road pixels are labeled as 0
            violations.append({
                "type": "Off-road driving",
                "bbox": det['bbox'],
                "class_id": det['class_id'],
                "confidence": det['confidence']
            })

    return violations

@router.post("/violations")
async def analyze_violations(request: Request, image_file: UploadFile = File(...)):
    """
    Analyze the uploaded image for traffic violations.

    Args:
        request (Request): FastAPI request object to access app state.
        image_file (UploadFile): Uploaded image file for vehicle detection and segmentation.
    """

    try:
        image_contents = await image_file.read()
        nparr = np.frombuffer(image_contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Get model sessions
        detection_session = request.app.state.detection_session
        segmentation_session = request.app.state.segmentation_session

        # Run detection
        detections = run_detections(detection_session, image)

        # Run segmentation
        from ..features.segment import run_segmentation
        segmentation_mask = run_segmentation(segmentation_session, image)

        violations = detect_violation(detections, segmentation_mask)

        return {"violations": violations, "detections": detections}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import APIRouter, HTTPException, UploadFile, File, Request
import numpy as np
import cv2
import base64
from typing import List
from ..features.segment import run_segmentation
from ..features.detect import run_detections

router = APIRouter()

def detect_violation(detections: List[dict], segmentation_mask: np.ndarray, image_shape: tuple) -> List[dict]:
    """
    Detect off-road violations based on vehicle detections and segmentation mask.
    """
    violations = []

    # Resize mask to match image
    mask_resized = cv2.resize(segmentation_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    for det in detections:
        # Support both 'bbox' and 'box' keys
        bbox = det.get('bbox') or det.get('box')
        if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        # Clip to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_shape[1], x2), min(image_shape[0], y2)

        vehicle_area = mask_resized[y1:y2, x1:x2]

        if vehicle_area.size == 0:
            continue

        if np.any(vehicle_area == 0):  # Non-road pixel
            violations.append({
                "type": "Off-road driving",
                "bbox": bbox,
                "class_id": det.get('class_id', det.get('class')),
                "confidence": det.get('confidence', det.get('score'))
            })

    return violations

def create_overlay(image: np.ndarray, segmentation_mask: np.ndarray) -> str:
    """
    Create a colored overlay of the segmentation mask on the original image and return as base64 string.
    """
    mask_resized = cv2.resize(segmentation_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(image)
    road_color = [0, 255, 0]  # Green for road
    colored_mask[mask_resized == 1] = road_color

    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    _, buffer = cv2.imencode(".png", overlay)
    overlay_base64 = base64.b64encode(buffer).decode("utf-8")
    return overlay_base64

@router.post("/violations")
async def analyze_violations(request: Request, file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Get model sessions
        detection_session = request.app.state.detection_session
        segmentation_session = request.app.state.segmentation_session

        # Run models
        detections = run_detections(detection_session, image)
        segmentation_mask = run_segmentation(segmentation_session, image)

        # Detect violations
        violations = detect_violation(detections, segmentation_mask, image.shape)

        # Create overlay for visualization
        overlay_base64 = create_overlay(image, segmentation_mask)

        return {
            "violations": violations,
            "detections": detections,
            "overlay_image": overlay_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

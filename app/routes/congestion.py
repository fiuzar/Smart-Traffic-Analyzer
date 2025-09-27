import cv2
import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from typing import List

from ..features.detect import run_detections

router = APIRouter()

def vehicle_count(detections: List[dict]) -> dict:
    """
    Count
    """

    counts = {
        "car": 0,
        "bus": 0,
        "truck": 0,
        "motorcycle": 0,
        "bicycle": 0,
        "others": 0
    }

    for det in detections:
        label = det.get("class")
        if label in counts:
            counts[label] += 1
        else:
            counts["others"] += 1

    return counts

def get_congestion_level(counts: dict) -> str:
    total_vehicles = sum(counts.values())
    if total_vehicles < 10:
        return "Low"
    elif total_vehicles < 30:
        return "Moderate"
    else:
        return "High"
    
@router.post("/congestion", summary="Analyze congestion level from an image")
async def analyze_congestion(request: Request, file: UploadFile = File(...)):
    """
    
    """

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        detection_session: ort.InferenceSession = request.app.state.detection_session
        if detection_session is None:
            raise HTTPException(status_code=500, detail="Detection model not loaded")
        
        detections = run_detections(detection_session, image)
        counts = vehicle_count(detections)

        congestion_level = get_congestion_level(counts)

        return {
            "vehicle counts": counts,
            "congestion_level": congestion_level,
            "detections": detections
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
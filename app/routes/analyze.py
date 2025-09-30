# app/routes/analyze.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64

from ..features.detect import run_detections
from .violations import detect_violation
from .congestion import vehicle_count
from .congestion import get_congestion_level as assess_congestion
from ..features.segment import run_segmentation, apply_mask_to_image

# from app.routes.detect import run_detection
# from app.routes.vehicle_count import count_vehicles
# from app.routes.violation import check_violations
# from app.routes.congestion import assess_congestion
# from app.routes.segment import run_segmentation
# from app.core.model_loader import load_detection_model, load_segmentation_model

router = APIRouter()

# Load models once at startup
detection_session = None
segmentation_session = None


# @router.on_event("startup")
# async def startup_event():
#     global detection_session, segmentation_session
#     try:
#         detection_session = load_detection_model()
#         segmentation_session = load_segmentation_model()
#     except Exception as e:
#         raise RuntimeError(f"Model loading failed: {str(e)}")


@router.post("/analyze")
async def analyze_frame(request: Request, file: UploadFile = File(...)):
    """
    Analyze a single frame for detection, vehicle count, violations,
    congestion, and segmentation.
    Returns JSON response only.
    """
    try:
        # Read image
        file_bytes = await file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        detection_session = request.app.state.detection_session
        segmentation_session = request.app.state.segmentation_session

        # --- Run pipeline ---
        detections = run_detections(detection_session, image)
        segmentations = run_segmentation(segmentation_session, image)
        segmentation_result = apply_mask_to_image(image, segmentations)  # For visualization if needed
        
        _, img_encoded = cv2.imencode('.png', segmentation_result)
        img_bytes = img_encoded.tobytes()
        
        count_vehicle = vehicle_count(detections)
        violations = detect_violation(detections, segmentations, image.shape)
        congestion = assess_congestion(count_vehicle)

        # --- Build response ---
        result = {
            "vehicle_count": count_vehicle,
            "detections": detections,
            "violations": violations,
            "congestion": congestion,
            "lane_segmentation": base64.b64encode(img_bytes).decode("utf-8")  # Base64 encoded image
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing frame: {str(e)}")

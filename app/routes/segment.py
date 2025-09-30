from fastapi import APIRouter, HTTPException, Request
from fastapi import UploadFile, File
from ..features.segment import preprocess_image, run_segmentation, apply_mask_to_image
import numpy as np
import cv2
from contextlib import asynccontextmanager
import onnxruntime as ort
import base64

# segementation_session: ort.InferenceSession = None

# @asynccontextmanager
# async def lifespan(app: APIRouter):
#     global segementation_session
#     segementation_session = load_segmentation_model(use_gpu=True)
#     print("Segmentation model loaded and ready.")
#     yield

router = APIRouter()

@router.post("/segment")
async def segment_image(request: Request, file: UploadFile = File(...)):
    # Read image

    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    segmentation_session = request.app.state.segmentation_session

    # Preprocess image
    mask = run_segmentation(segmentation_session, image)
    overlay = apply_mask_to_image(image, mask)

    # Encode image to send back
    _, img_encoded = cv2.imencode('.png', overlay)
    img_bytes = img_encoded.tobytes()  

    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "segmented_image": img_base64
    }
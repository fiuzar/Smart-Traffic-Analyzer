from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/health", summary="Check API health status")
async def health_check(request: Request):
    """
    Health check endpoint to verify API is running
    """

    detection_session = getattr(request.app.state, "detection_session", None)
    segmentation_session = getattr(request.app.state, "segmentation_session", None)

    return {
        "status": "ok",
        "detection_model_loaded": detection_session is not None,
        "segmentation_model_loaded": segmentation_session is not None
    }
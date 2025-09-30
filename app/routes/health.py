from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import time

router = APIRouter()

metrics = {
    "request_total": 0,
    "last_request_time": None,
    "errors_total": 0,
    "uptime_start": time.time()
}

# @router.middleware_stack("http")
# async def count_requests(request, call_next):
#     """
    
#     """

#     metrics["request_total"] += 1
#     metrics["last_request_time"] = time.time()

#     try:
#         response = call_next(request)

#     except Exception:
#         metrics["errors_total"] += 1
#         raise


@router.get("/metrics")
async def get_metrics():
    """
    """

    uptime_seconds = int(time.time() - metrics["uptime_start"])

    return {
        "request_total": metrics["request_total"],
        "errors_total": metrics["errors_total"],
        "last_request_time": metrics["last_request_time"],
        "uptime_seconds": uptime_seconds
    }


@router.get("/ready", summary="Check API health status")
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


@router.get("/health", tags=["System Health"])
async def health_check():
    """
    Health check endpoint to verify system is healthy
    """

    return JSONResponse(content={"status": "ok", "message": "System is healthy"})
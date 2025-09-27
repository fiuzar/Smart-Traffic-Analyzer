import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import aiofiles
import numpy as np
import cv2
from contextlib import asynccontextmanager

from .core.model_loader import load_detection_model, load_segmentation_model
from .routes import detect, segment, vehicle_count, violations

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detection_session = load_detection_model(use_gpu=True)
    app.state.segmentation_session = load_segmentation_model(use_gpu=True)
    print("Models loaded and ready.")

    yield

app = FastAPI(title="Smart Traffic Analyzer", lifespan=lifespan)

app.include_router(detect.router, prefix="/api/v1/traffic", tags=["Vehicle Detection"])
app.include_router(segment.router, prefix="/api/v1/road", tags =["Road Segmentation"])
app.include_router(vehicle_count.router, prefix="/api/v1/vehicle", tags=["Vehicle Counting"])
app.include_router(violations.router, prefix="/api/v1/traffic", tags=["Traffic Violations"])

@app.get("/")
def index() :
    return { "message": "returned success good" }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="localhost", port=8000, reload=True)
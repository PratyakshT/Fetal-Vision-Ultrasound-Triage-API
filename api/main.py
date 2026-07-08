"""
Fetal-Vision FastAPI Backend.
Integrates in-memory preprocessing, binary caching, and PyTorch inference 
with resilient custom exception handling.
"""

import sys
import os
import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse

# Ensure the src directory is accessible for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exceptions import (
    FetalVisionError, 
    MetadataMissingError, 
    CorruptedImageError, 
    ResolutionMismatchError
)
from src.model import FetalHCModel
from src.preprocess import preprocess_image
from src.cache import get_cached_prediction, set_cached_prediction

app = FastAPI(
    title="Fetal-Vision HC18 Triage API",
    description="High-throughput, cached inference engine for fetal head circumference prediction."
)

# 1. Load the model globally at startup (prevents reloading on every request)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FetalHCModel(pretrained=False)
# In a real deployment, load your trained weights here:
weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fetal_vision_weights.pth")
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
else:
    raise RuntimeError(f"CRITICAL OVERRIDE: Trained weights not found at {weights_path}. API boot aborted.")
model.to(device)
model.eval()

# 2. Resilient System Design: Global Exception Handlers
@app.exception_handler(FetalVisionError)
async def custom_fetal_vision_handler(request: Request, exc: FetalVisionError):
    """Catches domain-specific pipeline errors and prevents server crashes."""
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "error_type": exc.__class__.__name__,
            "message": str(exc)
        },
    )

# 3. The Core Inference Endpoint
@app.post("/api/v1/predict_hc")
async def predict_head_circumference(
    file: UploadFile = File(...),
    pixel_size: float = Form(...)
):
    # A. Validate Metadata
    if pixel_size <= 0:
        raise MetadataMissingError(file.filename, "Pixel size must be greater than zero.")

    # B. Read image safely into memory
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise CorruptedImageError(file.filename)

    # C. Caching Layer (Bypass ML entirely if seen before)
    cached_result = get_cached_prediction(img, pixel_size)
    if cached_result:
        return {"status": "success", "source": "binary_cache", "data": cached_result}

    # D. In-Memory Preprocessing (Matching the Data Engine logic)
    target_mm_per_pixel = 0.1
    target_shape = (512, 512)

    scale_factor = pixel_size / target_mm_per_pixel
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    
    # Mathematical bounds check
    if new_width > 2000 or new_height > 2000:
        raise ResolutionMismatchError(
            (new_height, new_width), 
            target_shape, 
            "Calculated dimensions are clinically improbable based on provided pixel size."
        )

    final_tensor, scale_factor, pad_top, pad_left, start_h, start_w = preprocess_image(
        img, pixel_size, target_mm_per_pixel, target_shape
    )
    
    # E. PyTorch Inference
    input_tensor = torch.tensor(final_tensor).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        preds = outputs[0].cpu().numpy()
        
    # F. Inverse Transform Predictions to Original Image Space
    # Target normalization scales coordinates by 512.0 and angle by 180.0
    cx_final = float(preds[0]) * target_shape[1]
    cy_final = float(preds[1]) * target_shape[0]
    a_final = float(preds[2]) * target_shape[1]
    b_final = float(preds[3]) * target_shape[0]
    angle = float(preds[4]) * 180.0
    
    cx_orig = (cx_final - pad_left + start_w) / scale_factor
    cy_orig = (cy_final - pad_top + start_h) / scale_factor
    a_orig = a_final / scale_factor
    b_orig = b_final / scale_factor
    
    prediction_dict = {
        "cx": float(preds[0] * 512.0),
        "cy": float(preds[1] * 512.0),
        "a": float(preds[2] * 512.0),     
        "b": float(preds[3] * 512.0),     
        "angle": float(preds[4] * 180.0)
    }

    # G. Save to cache for future requests
    set_cached_prediction(img, pixel_size, prediction_dict)

    return {"status": "success", "source": "model_inference", "data": prediction_dict}
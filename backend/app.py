"""
FastAPI Backend for Text-Conditioned Image Segmentation
Provides REST API for image + prompt → segmentation mask
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
import os
import sys
import tempfile
from datetime import datetime

# Add current directory to path so imports work on Render
sys.path.insert(0, os.path.dirname(__file__))

# Import local modules
from inference import predict, PROMPT_TO_MODE, DEVICE
from model import ResNetSegmentation

# ============================================
# APP INITIALIZATION
# ============================================
app = FastAPI(
    title="Text-Conditioned Segmentation API",
    description="Crack and drywall taping detection via natural language prompts",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# GLOBAL STATE
# ============================================
MODEL = None
MODEL_PATH = "../best_model.pth"

def load_model():
    """Load model once at startup"""
    global MODEL
    if MODEL is None:
        print(f"Loading model from {MODEL_PATH}...")
        MODEL = ResNetSegmentation().to(DEVICE)
        
        if os.path.exists(MODEL_PATH):
            MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model file {MODEL_PATH} not found. Using untrained weights.")
        
        MODEL.eval()
    return MODEL

@app.on_event("startup")
async def startup_event():
    """Load model on app startup"""
    load_model()
    print("API ready!")

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def health():
    return {"status": "MANTIS backend running"}

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Predict segmentation mask for uploaded image with given prompt.
    """
    try:
        # Validate prompt
        if prompt not in PROMPT_TO_MODE:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid prompt: '{prompt}'"
            )
            
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference
        mask = predict(image, prompt, MODEL_PATH)

        # Convert mask to PNG in memory
        buf = io.BytesIO()
        Image.fromarray(mask).save(buf, format="PNG")
        buf.seek(0)
        
        return Response(content=buf.getvalue(), media_type="image/png")
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict_endpoint(
    images: list[UploadFile] = File(...),
    prompt: str = Form(...)
):
    """
    Run inference on multiple images with the same prompt.
    """
    try:
        if prompt not in PROMPT_TO_MODE:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid prompt: '{prompt}'"
            )
        
        results = []
        
        for idx, image in enumerate(images):
            contents = await image.read()
            img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img_pil.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                _, mask = predict(tmp_path, prompt, MODEL_PATH)
                
                prompt_slug = prompt.replace(" ", "_").lower()
                filename = f"batch_{idx}__{prompt_slug}.png"
                
                # In a real batch scenario, we'd save these to a bucket
                # Here we just mock success
                
                results.append({
                    "image_id": idx,
                    "filename": image.filename,
                    "status": "success",
                    "bbox_count": "N/A" # Could count from mask
                })
            
            except Exception as e:
                results.append({
                    "image_id": idx,
                    "filename": image.filename,
                    "status": "failed",
                    "error": str(e)
                })
            
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        return {"results": results}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts")
async def list_prompts():
    """List all available prompts"""
    return {
        "prompts": list(PROMPT_TO_MODE.keys()),
        "grouped": {
            "crack": ["segment crack", "segment wall crack"],
            "drywall": ["segment taping area", "segment joint", "segment drywall seam"]
        }
    }

@app.get("/")
async def root():
    """API documentation"""
    return {
        "title": "Text-Conditioned Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "GET /prompts": "List available prompts",
            "POST /predict": "Single image prediction",
            "POST /batch-predict": "Multiple image prediction",
            "GET /": "This documentation"
        },
        "usage": {
            "single": "POST /predict with image file and prompt",
            "batch": "POST /batch-predict with list of images and prompt"
        }
    }

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

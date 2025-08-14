import threading
import logging
import numpy as np

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from starlette.concurrency import run_in_threadpool

from api.plotter import plot_points
from models.models import CNNModel

log = logging.getLogger(__name__)
app = FastAPI()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_params=7)
state = torch.load("output/best_model.pth", map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# If GPU, optionally serialize GPU calls
inference_lock = threading.Lock() if device.type == "cuda" else None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class Point(BaseModel):
    x: float
    y: float

class PointsRequest(BaseModel):
    points: List[Point]

def _infer_on_tensor(img_t):
    # runs in a worker thread
    if inference_lock:
        with inference_lock:
            with torch.inference_mode():
                out = model(img_t)
    else:
        with torch.inference_mode():
            out = model(img_t)
    return out.cpu().squeeze(0).tolist()

@app.post("/estimate/")
async def get_polynomial_estimate(data: PointsRequest):
    try:
        if not data.points or len(data.points) < 50:
            raise HTTPException(status_code=400, detail="Not enough points drawn")

        point_tuples = [(p.x, p.y) for p in data.points]

        # plotting is blocking => run in threadpool
        pil_img = await run_in_threadpool(plot_points, point_tuples)

        # quick transform (not usually blocking)
        img_t = transform(pil_img).unsqueeze(0).to(device)

        # inference (blocking) -> run in threadpool
        coeffs = await run_in_threadpool(_infer_on_tensor, img_t)

        return {"coeffs": coeffs}

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))

# debug route for testing
@app.post("/test/")
async def get_polynomial_estimate_from_poly():
    try:
        # Generate 52 points from -5 to 5
        x_values = np.linspace(-5, 5, 52)

        def poly(x):
            return 0.222*x**5 - 0.469*x**4 - 0.357*x**3 + 0.162*x**2 - 0.138*x + 2.099

        y_values = poly(x_values)
        points = list(zip(x_values, y_values))

        # Plot image from points (blocking, so run in threadpool)
        pil_img = await run_in_threadpool(plot_points, points)

        # Transform image tensor (fast)
        img_t = transform(pil_img).unsqueeze(0).to(device)

        # Run inference in threadpool
        coeffs = await run_in_threadpool(_infer_on_tensor, img_t)

        return {"coeffs": coeffs}

    except Exception as e:
        log.exception("Inference from poly failed")
        raise HTTPException(status_code=500, detail=str(e))

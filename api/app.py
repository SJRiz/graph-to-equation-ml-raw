import threading
import asyncio
import logging

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from starlette.concurrency import run_in_threadpool

from plotter import plot_points
from models.models import CNNModel

log = logging.getLogger(__name__)
app = FastAPI()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_params=7)
state = torch.load("model.pth", map_location=device)
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

from typing import List
from fastapi import FastAPI

app = FastAPI()

@app.get("/estimate/")
async def get_polynomial_estimate(
    points: List[tuple[float]]
):
    pass
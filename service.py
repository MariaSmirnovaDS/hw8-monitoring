import time
import cv2
import numpy as np
from fastapi import FastAPI, Response
from prometheus_client import Histogram, generate_latest, CONTENT_TYPE_LATEST
from ultralytics import YOLO
import uvicorn

model = YOLO("yolo11n.pt")

LATENCY = Histogram("request_latency_seconds", "Request latency in seconds")

app = FastAPI()

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/detect")
async def detect():
    start = time.time()

    time.sleep(2.0)

    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(dummy_frame)
    num_objects = len(results[0].boxes) if results[0].boxes else 0

    LATENCY.observe(time.time() - start)
    return {"status": "ok", "objects_detected": num_objects, "latency_seconds": time.time() - start}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

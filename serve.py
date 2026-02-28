"""
Minimal API: load saved model and expose /predict, /metrics, /health.
Run: uvicorn serve:app --reload
"""
from pathlib import Path

import time

import joblib
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

MODEL_PATH = Path("models/model.joblib")
app = FastAPI()
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

REQUEST_COUNT = Counter("mlops_requests_total", "Total requests", ["method", "path"])
REQUEST_LATENCY = Histogram("mlops_request_duration_seconds", "Request latency", ["path"])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_COUNT.labels(method=request.method, path=request.url.path).inc()
    REQUEST_LATENCY.labels(path=request.url.path).observe(duration)
    return response

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(features: list[float]):
    if model is None:
        return {"error": "No model. Run train.py first."}
    pred = model.predict([features])
    return {"prediction": int(pred[0])}

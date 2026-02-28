"""
API: load model, expose /predict, /metrics, /health.
Run: uvicorn app.serve:app --reload (from repo root)
"""
import json
import logging
import os
import time
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.requests import Request

_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = _ROOT / "models" / "model.joblib"
app = FastAPI()
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops.api")
LOG_JSON = os.environ.get("LOG_JSON", "").lower() in ("1", "true", "yes")

REQUEST_COUNT = Counter("mlops_requests_total", "Total requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("mlops_request_duration_seconds", "Request latency", ["path"])
PREDICTIONS_TOTAL = Counter("mlops_predictions_total", "Total predictions", ["prediction_class"])
PREDICTION_ERRORS = Counter("mlops_prediction_errors_total", "Prediction errors", ["reason"])


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    path = request.url.path
    status = response.status_code
    REQUEST_COUNT.labels(method=request.method, path=path, status=status).inc()
    REQUEST_LATENCY.labels(path=path).observe(duration)
    if LOG_JSON:
        logger.info(json.dumps({"method": request.method, "path": path, "status": status, "duration_s": round(duration, 4)}))
    else:
        logger.info("%s %s %s %.3fs", request.method, path, status, duration)
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
        PREDICTION_ERRORS.labels(reason="no_model").inc()
        return {"error": "No model. Run train.py first."}
    if len(features) != 4:
        PREDICTION_ERRORS.labels(reason="bad_input").inc()
        return {"error": "Expected 4 features."}
    pred = model.predict([features])
    pred_class = int(pred[0])
    PREDICTIONS_TOTAL.labels(prediction_class=str(pred_class)).inc()
    return {"prediction": pred_class}

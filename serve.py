"""
Minimal API: load saved model and expose /predict.
Run: uvicorn serve:app --reload
"""
from pathlib import Path

import joblib
from fastapi import FastAPI

MODEL_PATH = Path("models/model.joblib")
app = FastAPI()
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(features: list[float]):
    if model is None:
        return {"error": "No model. Run train.py first."}
    pred = model.predict([features])
    return {"prediction": int(pred[0])}

"""
Basic training script: load data, train model, save artifact.
Run: python train.py
"""
from pathlib import Path

import joblib
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

def main():
    import os
    # In CI, keep MLflow under workspace to avoid permission/path issues
    if os.environ.get("CI"):
        os.environ.setdefault("MLFLOW_TRACKING_URI", str(Path.cwd() / "mlruns"))

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if not os.environ.get("CI"):
        with mlflow.start_run():
            mlflow.log_param("n_estimators", 10)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, name="model", input_example=X_train[:1])

    path = MODEL_DIR / "model.joblib"
    joblib.dump(model, path)
    print(f"Accuracy: {acc:.4f}, model saved to {path}")

if __name__ == "__main__":
    main()

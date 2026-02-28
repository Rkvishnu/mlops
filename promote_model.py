"""
Copy a logged model from MLflow into models/model.joblib (for deploy).
Usage:
  From MLflow server: MLFLOW_TRACKING_URI=http://localhost:5000 python promote_model.py RUN_ID
  From local backup:  python promote_model.py /path/to/mlflow/model/dir
  (Model dir = folder with MLmodel + model.pkl, e.g. mlflow-artifacts-backup/0/models/m-xxx/artifacts)
"""
import os
import sys
import tempfile
from pathlib import Path

# Avoid indefinite hang on artifact download (seconds)
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")

import joblib
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def main():
    if len(sys.argv) != 2:
        print("Usage: python promote_model.py RUN_ID  or  python promote_model.py /path/to/model/dir", file=sys.stderr)
        sys.exit(1)
    arg = sys.argv[1].strip().replace("runs:/", "")
    local_dir = Path(arg).resolve()

    if local_dir.is_dir() and (local_dir / "MLmodel").exists():
        print(f"Loading model from local path {local_dir}...", flush=True)
        model = load_model(f"file://{local_dir}")
    else:
        run_id = arg
        print(f"Downloading model from run {run_id}...", flush=True)
        client = MlflowClient()
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = client.download_artifacts(run_id, "model", dst_path=tmp)
            print("Loading model...", flush=True)
            model = load_model(f"file://{os.path.abspath(local_dir)}")

    print("Saving to models/model.joblib...", flush=True)
    path = MODEL_DIR / "model.joblib"
    joblib.dump(model, path)
    print(f"Promoted model to {path}", flush=True)

if __name__ == "__main__":
    main()

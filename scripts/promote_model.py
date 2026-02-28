"""
Copy a logged model from MLflow into models/model.joblib (for deploy).
Run from repo root: python scripts/promote_model.py RUN_ID  or  python scripts/promote_model.py /path/to/model/dir
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")

import joblib
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient

_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = _ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/promote_model.py RUN_ID  or  python scripts/promote_model.py /path/to/model/dir", file=sys.stderr)
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

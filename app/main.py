from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

from .config import settings
from .inference import run_inference

app = FastAPI(title="Generic Model Inference API")

class PredictRequest(BaseModel):
    inputs: Any
    parameters: Optional[Dict[str, Any]] = None

def _log_prediction(payload: dict):
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = settings.LOG_DIR / f"predictions_{settings.ENVIRONMENT}.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow().isoformat()}] {payload}\n")

@app.get("/health")
def health():
    return {"status": "ok", "env": settings.ENVIRONMENT}

@app.post("/predict")
def predict(request: PredictRequest):
    payload = request.dict()
    result = run_inference(payload)
    _log_prediction({"input": payload, "output": result})
    return result

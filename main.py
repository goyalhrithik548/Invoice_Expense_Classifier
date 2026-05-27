import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.preprocessor import preprocess

MODEL_PATH = os.path.join("model", "model.pkl")

app = FastAPI(
    title="Invoice Expense Classifier",
    description="Classifies invoice text into expense categories using ML.",
    version="1.0.0",
)

# Load model once at startup
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model not found at '{MODEL_PATH}'. Run `python train.py` first."
    )

pipeline = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    text: str

    model_config = {"json_schema_extra": {"example": {"text": "AWS monthly cloud hosting bill"}}}


class PredictResponse(BaseModel):
    category: str
    confidence: float


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Invoice Expense Classifier is running."}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Request text cannot be empty.")

    cleaned = preprocess(request.text)
    category = pipeline.predict([cleaned])[0]
    probabilities = pipeline.predict_proba([cleaned])[0]
    confidence = round(float(max(probabilities)), 4)

    return PredictResponse(category=category, confidence=confidence)

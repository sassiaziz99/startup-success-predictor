"""
api.py
FastAPI service that loads the trained model and serves predictions.

Run with: uvicorn api:app --reload   (from inside the src/ folder)
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import os

# Works whether run locally from src/ (uvicorn api:app) or from the project
# root in Docker (uvicorn src.api:app) - resolves relative to this file's location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.joblib")

app = FastAPI(
    title="Startup Success Predictor",
    description="Predicts whether an angel-funded startup will raise a follow-on round (series-a/b/c+).",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)


class StartupInput(BaseModel):
    angel_amount_usd: float = Field(..., example=500000, description="Amount raised in the angel round (USD)")
    years_to_angel: float | None = Field(None, example=1.0, description="Years between founding and angel round")
    category_code: str = Field(..., example="software", description="Sector/category of the startup")
    country_code: str = Field(..., example="USA", description="Country code (e.g. USA, GBR)")


class PredictionOutput(BaseModel):
    will_raise_followon: bool
    probability: float


@app.get("/")
def root():
    return {"message": "Startup Success Predictor API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(startup: StartupInput):
    input_df = pd.DataFrame([{
        "angel_amount_usd": startup.angel_amount_usd,
        "years_to_angel": startup.years_to_angel,
        "category_code": startup.category_code,
        "country_code": startup.country_code,
    }])

    proba = model.predict_proba(input_df)[0][1]
    prediction = bool(proba >= 0.5)

    return PredictionOutput(
        will_raise_followon=prediction,
        probability=round(float(proba), 4),
    )
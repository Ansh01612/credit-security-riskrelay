from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "credit risk modeling" / "credit_risk_pipeline.joblib"
FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Purpose",
]


class ApplicationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(alias="Age", ge=18, le=100)
    sex: Literal["female", "male"] = Field(alias="Sex")
    job: Literal[0, 1, 2, 3] = Field(alias="Job")
    housing: Literal["free", "own", "rent"] = Field(alias="Housing")
    savings: Literal[
        "Not provided", "little", "moderate", "quite rich", "rich"
    ] = Field(alias="Saving accounts")
    checking: Literal["Not provided", "little", "moderate", "rich"] = Field(
        alias="Checking account"
    )
    credit_amount: int = Field(alias="Credit amount", ge=250, le=18424)
    duration: int = Field(alias="Duration", ge=1, le=72)
    purpose: Literal[
        "business",
        "car",
        "domestic appliances",
        "education",
        "furniture/equipment",
        "radio/TV",
        "repairs",
        "vacation/others",
    ] = Field(alias="Purpose")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applications: list[ApplicationInput] = Field(min_length=1, max_length=250)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}.")
    return joblib.load(MODEL_PATH)


app = FastAPI(
    title="CreditLens Risk API",
    description="Risk Relay model inference API.",
    version="1.0.0",
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, list[dict[str, object]]]:
    try:
        model = load_model()
        records = [
            item.model_dump(by_alias=True)
            for item in request.applications
        ]
        features = pd.DataFrame(records, columns=FEATURE_COLUMNS)
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, EOFError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    results: list[dict[str, object]] = []
    for record, prediction, probability_row in zip(
        records,
        predictions,
        probabilities,
        strict=True,
    ):
        result: dict[str, object] = {
            **record,
            "Predicted risk": str(prediction),
        }
        for index, label in enumerate(model.classes_):
            result[f"{str(label).title()} probability (%)"] = round(
                float(probability_row[index] * 100),
                1,
            )
        results.append(result)

    return {"predictions": results}

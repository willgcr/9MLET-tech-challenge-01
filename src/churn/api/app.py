"""FastAPI inference application."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request

from churn.api.middleware import LatencyMiddleware
from churn.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)
from churn.logging_config import configure_logging, get_logger
from churn.modeling.predictor import ChurnPredictor

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Loading model artifacts")
    app.state.predictor = ChurnPredictor.from_artifacts()
    logger.info(
        "Model artifacts loaded",
        extra={
            "model_version": app.state.predictor.model_version,
            "threshold": app.state.predictor.threshold,
        },
    )
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Churn Prediction API",
    description="Telco customer churn prediction served by a PyTorch MLP.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(LatencyMiddleware)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor: ChurnPredictor | None = getattr(request.app.state, "predictor", None)
    return HealthResponse(
        status="ok" if predictor else "starting",
        model_loaded=predictor is not None,
        model_version=predictor.model_version if predictor else None,
        threshold=predictor.threshold if predictor else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures, request: Request) -> PredictionResponse:
    predictor: ChurnPredictor = request.app.state.predictor
    df = pd.DataFrame([payload.model_dump()])
    probas, decisions = predictor.predict_dataframe(df)
    return PredictionResponse(
        churn_probability=probas[0],
        churn_predicted=decisions[0],
        model_version=predictor.model_version,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    payload: BatchPredictRequest, request: Request
) -> BatchPredictResponse:
    predictor: ChurnPredictor = request.app.state.predictor
    df = pd.DataFrame([c.model_dump() for c in payload.customers])
    probas, decisions = predictor.predict_dataframe(df)
    return BatchPredictResponse(
        predictions=[
            PredictionResponse(
                churn_probability=p,
                churn_predicted=d,
                model_version=predictor.model_version,
            )
            for p, d in zip(probas, decisions, strict=True)
        ]
    )

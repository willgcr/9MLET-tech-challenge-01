"""Pydantic request and response models for the inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
AddOnTrinary = Literal["Yes", "No", "No internet service"]
MultiLines = Literal["Yes", "No", "No phone service"]
InternetService = Literal["DSL", "Fiber optic", "No"]
Contract = Literal["Month-to-month", "One year", "Two year"]
PaymentMethod = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerFeatures(BaseModel):
    """Single customer record - matches the raw Telco Churn columns minus ID and target."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: MultiLines
    InternetService: InternetService
    OnlineSecurity: AddOnTrinary
    OnlineBackup: AddOnTrinary
    DeviceProtection: AddOnTrinary
    TechSupport: AddOnTrinary
    StreamingTV: AddOnTrinary
    StreamingMovies: AddOnTrinary
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_predicted: bool
    model_version: str


class BatchPredictRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    threshold: float | None = None

"""Smoke tests for the FastAPI inference service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn.modeling.predictor import METADATA_PATH, MODEL_PATH, PREPROCESSOR_PATH


def _artifacts_present() -> bool:
    return MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() and METADATA_PATH.exists()


SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not _artifacts_present():
        pytest.skip("Model artifacts missing - run `make train` first")
    from churn.api.app import app

    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["threshold"] is not None


def test_predict_returns_valid_probability(client: TestClient) -> None:
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_predicted"], bool)
    assert "X-Process-Time-Ms" in response.headers


def test_predict_rejects_unknown_category(client: TestClient) -> None:
    bad = {**SAMPLE_CUSTOMER, "gender": "ALIEN"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    bad = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "gender"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_batch(client: TestClient) -> None:
    response = client.post(
        "/predict/batch", json={"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    for pred in body["predictions"]:
        assert 0.0 <= pred["churn_probability"] <= 1.0

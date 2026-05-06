"""Pandera schema for the raw Telco Churn dataframe."""

from __future__ import annotations

import pandera as pa

YES_NO = ["Yes", "No"]
INTERNET_SERVICE = ["DSL", "Fiber optic", "No"]
ADDON_TRINARY = ["Yes", "No", "No internet service"]
MULTI_LINES = ["Yes", "No", "No phone service"]
CONTRACT = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

raw_schema = pa.DataFrameSchema(
    {
        "customerID": pa.Column(str, unique=True),
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female"])),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "Partner": pa.Column(str, pa.Check.isin(YES_NO)),
        "Dependents": pa.Column(str, pa.Check.isin(YES_NO)),
        "tenure": pa.Column(int, pa.Check.ge(0)),
        "PhoneService": pa.Column(str, pa.Check.isin(YES_NO)),
        "MultipleLines": pa.Column(str, pa.Check.isin(MULTI_LINES)),
        "InternetService": pa.Column(str, pa.Check.isin(INTERNET_SERVICE)),
        "OnlineSecurity": pa.Column(str, pa.Check.isin(ADDON_TRINARY)),
        "OnlineBackup": pa.Column(str, pa.Check.isin(ADDON_TRINARY)),
        "DeviceProtection": pa.Column(str, pa.Check.isin(ADDON_TRINARY)),
        "TechSupport": pa.Column(str, pa.Check.isin(ADDON_TRINARY)),
        "StreamingTV": pa.Column(str, pa.Check.isin(ADDON_TRINARY)),
        "StreamingMovies": pa.Column(str, pa.Check.isin(ADDON_TRINARY)),
        "Contract": pa.Column(str, pa.Check.isin(CONTRACT)),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(YES_NO)),
        "PaymentMethod": pa.Column(str, pa.Check.isin(PAYMENT_METHOD)),
        "MonthlyCharges": pa.Column(float, pa.Check.ge(0)),
        # TotalCharges arrives as object due to whitespace strings in some rows.
        # We accept the raw column as object here; the preprocessing pipeline
        # coerces it to float and imputes the missing values.
        "TotalCharges": pa.Column(str),
        "Churn": pa.Column(str, pa.Check.isin(YES_NO)),
    },
    strict=True,
    coerce=False,
)

"""
modules/regression/flight_fare/model.py

Flight fare prediction using XGBoost with engineered features.

Dataset: EaseMyTrip (~300k domestic Indian flights)
Target:  Price (INR)

Feature engineering:
  - Route encoding (source → destination)
  - Departure/arrival hour bins
  - Days until departure (booking lead time)
  - Stop count
  - Airline target encoding
  - Duration in minutes

Training:
    python -m modules.regression.flight_fare.model
"""

import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from core.config import FLIGHT_FARE_MODEL, DATA_DIR


FEATURE_COLS = [
    "airline_encoded",
    "source_encoded",
    "destination_encoded",
    "stops",
    "duration_minutes",
    "departure_hour",
    "arrival_hour",
    "days_before_departure",
    "is_weekend",
    "is_morning_flight",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering pipeline to raw flight data."""
    df = df.copy()

    # Duration → minutes
    if "duration" in df.columns:
        df["duration_minutes"] = df["duration"].str.extract(r"(\d+)h").astype(float) * 60
        df["duration_minutes"] += df["duration"].str.extract(r"(\d+)m").fillna(0).astype(float)

    # Stops → ordinal
    stop_map = {"zero": 0, "one": 1, "two_or_more": 2}
    if "stops" in df.columns:
        df["stops"] = df["stops"].str.lower().map(stop_map).fillna(1)

    # Time features
    if "departure_time" in df.columns:
        df["departure_hour"] = pd.to_datetime(df["departure_time"], format="%H:%M", errors="coerce").dt.hour
        df["is_morning_flight"] = (df["departure_hour"].between(6, 12)).astype(int)

    if "arrival_time" in df.columns:
        df["arrival_hour"] = pd.to_datetime(df["arrival_time"], format="%H:%M", errors="coerce").dt.hour

    if "date_of_journey" in df.columns:
        journey_date = pd.to_datetime(df["date_of_journey"], dayfirst=True, errors="coerce")
        df["days_before_departure"] = (journey_date - pd.Timestamp.today()).dt.days.clip(0, 365)
        df["is_weekend"] = journey_date.dt.dayofweek.isin([5, 6]).astype(int)

    # Label encode categoricals
    for col, enc_col in [("airline", "airline_encoded"), ("source", "source_encoded"), ("destination", "destination_encoded")]:
        if col in df.columns:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[col].astype(str))

    return df


def train(data_path: Optional[str] = None) -> dict:
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    # Load data
    csv_path = data_path or DATA_DIR / "flight_fare.csv"
    df = pd.read_csv(csv_path)
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLS + ["price"])

    X = df[FEATURE_COLS]
    y = df["price"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    metrics = {
        "mae":  round(float(mean_absolute_error(y_val, preds)), 2),
        "rmse": round(float(mean_squared_error(y_val, preds) ** 0.5), 2),
        "r2":   round(float(r2_score(y_val, preds)), 4),
    }

    with open(FLIGHT_FARE_MODEL, "wb") as f:
        pickle.dump(model, f)

    print(f"Flight fare model saved → {FLIGHT_FARE_MODEL}")
    print(f"Metrics: {metrics}")
    return metrics


def load_model():
    if not FLIGHT_FARE_MODEL.exists():
        raise FileNotFoundError(
            f"No weights at {FLIGHT_FARE_MODEL}. Train first: python -m modules.regression.flight_fare.model"
        )
    with open(FLIGHT_FARE_MODEL, "rb") as f:
        return pickle.load(f)


def predict(features: dict) -> tuple[float, float]:
    """
    Returns (predicted_price_inr, inference_ms).
    features: dict with keys matching FEATURE_COLS
    """
    model = load_model()
    row = pd.DataFrame([{col: features.get(col, 0) for col in FEATURE_COLS}])
    t0 = time.time()
    pred = float(model.predict(row)[0])
    ms = (time.time() - t0) * 1000
    return round(pred, 2), round(ms, 2)


if __name__ == "__main__":
    train()

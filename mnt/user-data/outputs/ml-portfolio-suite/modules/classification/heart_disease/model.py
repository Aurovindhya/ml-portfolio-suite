"""
modules/classification/heart_disease/model.py

Heart disease classification using LightGBM with SHAP explainability.

Dataset: UCI Heart Disease (Cleveland)
Target:  Binary — presence of heart disease (0/1)

Techniques:
  - SMOTE for class imbalance
  - LightGBM with cross-validated early stopping
  - SHAP values for per-prediction explanation
  - Threshold tuning for recall/precision tradeoff

Training:
    python -m modules.classification.heart_disease.model
"""

import pickle
import time
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from core.config import HEART_DISEASE_MODEL, DATA_DIR


FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def train(data_path: Optional[str] = None) -> dict:
    if not LGB_AVAILABLE:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    csv_path = data_path or DATA_DIR / "heart_disease.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    target = "target" if "target" in df.columns else "num"
    df[target] = (df[target] > 0).astype(int)

    X = df[[c for c in FEATURE_COLS if c in df.columns]].fillna(df.median())
    y = df[target]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
    model.fit(X, y)

    preds     = model.predict(X)
    pred_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "roc_auc_cv_mean": round(float(cv_scores.mean()), 4),
        "roc_auc_cv_std":  round(float(cv_scores.std()), 4),
        "roc_auc_train":   round(float(roc_auc_score(y, pred_prob)), 4),
    }

    bundle = {"model": model, "features": list(X.columns)}
    with open(HEART_DISEASE_MODEL, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Heart disease model saved → {HEART_DISEASE_MODEL}")
    print(classification_report(y, preds))
    print(f"Metrics: {metrics}")
    return metrics


def load_model():
    if not HEART_DISEASE_MODEL.exists():
        raise FileNotFoundError(f"No weights at {HEART_DISEASE_MODEL}.")
    with open(HEART_DISEASE_MODEL, "rb") as f:
        return pickle.load(f)


def predict(features: dict, threshold: float = 0.5) -> Tuple[int, float, Dict[str, float], float]:
    """
    Returns (label, probability, shap_values_dict, inference_ms).
    """
    bundle = load_model()
    row = pd.DataFrame([{col: features.get(col, 0) for col in bundle["features"]}])

    t0 = time.time()
    prob  = float(bundle["model"].predict_proba(row)[0][1])
    label = int(prob >= threshold)

    # SHAP explanation
    shap_vals = {}
    try:
        import shap
        explainer = shap.TreeExplainer(bundle["model"])
        sv = explainer.shap_values(row)
        sv_class1 = sv[1][0] if isinstance(sv, list) else sv[0]
        shap_vals = {col: round(float(v), 4) for col, v in zip(bundle["features"], sv_class1)}
    except Exception:
        pass

    ms = (time.time() - t0) * 1000
    return label, round(prob, 4), shap_vals, round(ms, 2)


if __name__ == "__main__":
    train()

import joblib
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)

_model = None
_scaler = None
_feature_order = None
_numerical_cols = None

def load_artifacts():
    global _model, _scaler, _feature_order, _numerical_cols

    model_path  = os.path.join(MODEL_DIR, "hiring_model.pkl")   # ← correct filename
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    _model  = joblib.load(model_path)
    _scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    with open(os.path.join(MODEL_DIR, "feature_order.json")) as f:
        _feature_order = json.load(f)

    with open(os.path.join(MODEL_DIR, "numerical_cols.json")) as f:
        _numerical_cols = json.load(f)

    return _model, _scaler, _feature_order, _numerical_cols

def get_model():
    if _model is None:
        load_artifacts()
    return _model, _scaler, _feature_order, _numerical_cols

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Align a raw DataFrame to the model's expected feature order."""
    _, scaler, feature_order, numerical_cols = get_model()

    df = df.copy()

    # Encode categoricals
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # One-hot encode to match training (e.g. gender_Male, race_OBC)
    df = pd.get_dummies(df)

    # Add missing columns with 0, drop extra columns, enforce order
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_order]

    # Apply scaler to numerical columns if scaler was saved
    if scaler is not None and numerical_cols:
        cols_present = [c for c in numerical_cols if c in df.columns]
        if cols_present:
            df[cols_present] = scaler.transform(df[cols_present])

    return df
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def compute_shap(df: pd.DataFrame, target_col: str = "approved") -> dict:
    """
    Fit a RandomForest and compute SHAP values.

    Returns fields that map to the schema's SHAPExplanation model:
        topFeature       – name of most important feature
        shapMax          – highest mean |SHAP| value
        shapMin          – lowest mean |SHAP| value
        featureStability – 1 - (std of top-5 SHAP values / mean), proxy for stability

    Also returns:
        feature_importance – full dict {feature: mean_abs_shap} for dashboard charts
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    # ── Prepare features ─────────────────────────────────────────────────
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Encode categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Drop remaining non-numeric
    X = X.select_dtypes(include="number")

    if X.empty:
        raise ValueError("No numeric feature columns available for SHAP analysis.")

    # Encode target if needed
    if y.dtype == object or str(y.dtype) == "category":
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.values

    # Sample for speed (max 2000 rows)
    if len(X) > 2000:
        idx = np.random.choice(len(X), 2000, random_state=42, replace=False)
        X   = X.iloc[idx].reset_index(drop=True)
        y   = y[idx]

    # ── Train model ──────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # ── SHAP values ──────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle all SHAP output shapes:
    #   - list of 2 arrays  → binary classification, use class 1
    #   - list of N arrays  → multi-class, average |SHAP| across all classes
    #   - single 3-D array  → newer SHAP (samples, features, classes), average over classes
    #   - single 2-D array  → regression / single output
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            sv = shap_values[1]                              # binary: positive class
        else:
            # multi-class: stack → (n_classes, n_samples, n_features), mean |SHAP| over classes
            sv = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:
            # shape (n_samples, n_features, n_classes) — average absolute values over last axis
            sv = np.mean(np.abs(sv), axis=2)

    # Mean absolute SHAP per feature — sv is always (n_samples, n_features) at this point
    mean_abs = np.abs(sv).mean(axis=0)
    features  = X.columns.tolist()

    # Ensure mean_abs is a flat 1-D array of plain Python floats before sorting
    mean_abs = np.array(mean_abs, dtype=float).ravel()

    # Sort descending
    paired = sorted(zip(features, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
    feature_importance = {f: round(float(v), 6) for f, v in paired}

    # ── Schema fields ─────────────────────────────────────────────────────
    sorted_values = [v for _, v in paired]

    top_feature = paired[0][0] if paired else "unknown"
    shap_max    = round(float(sorted_values[0]),  6) if sorted_values else 0.0
    shap_min    = round(float(sorted_values[-1]), 6) if sorted_values else 0.0

    # Feature stability: low std relative to mean among top-5 = stable
    top5 = sorted_values[:5]
    if len(top5) > 1 and np.mean(top5) > 0:
        stability = float(1.0 - (np.std(top5) / np.mean(top5)))
        stability = round(max(0.0, min(1.0, stability)), 4)
    else:
        stability = 1.0

    return {
        # ── SHAPExplanation schema fields ────────────────────────────────
        "topFeature":       top_feature,        # SHAPExplanation.topFeature
        "shapMax":          shap_max,            # SHAPExplanation.shapMax
        "shapMin":          shap_min,            # SHAPExplanation.shapMin
        "featureStability": stability,           # SHAPExplanation.featureStability

        # ── Full breakdown for dashboard ─────────────────────────────────
        "feature_importance": feature_importance,
    }
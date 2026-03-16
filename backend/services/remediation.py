"""
remediation.py
Bias Remediation Toolkit — suggests specific fixes based on evaluation results.
Does NOT modify the model. All suggestions are human-actionable recommendations.

Returns:
- Prioritized list of fixes
- Before/after estimated improvement
- Code snippets
- Effort estimates
"""

import numpy as np
from typing import Dict, Any, List


# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "disparateImpact":    0.80,
    "demographicParity":  0.05,   # gap (lower = better)
    "individualFairness": 0.80,
    "groupFairness":      0.80,
    "calibrationError":   0.05,   # error (lower = better)
    "counterfactual":     0.80,
    "intersectional":     0.80,
    "privacy_score":      0.75,
    "robustness_score":   0.75,
    "transparency_score": 0.70,
    "accountability_score": 0.80,
}

DIMENSION_LABELS = {
    "disparateImpact":     "Disparate Impact",
    "demographicParity":   "Demographic Parity",
    "individualFairness":  "Individual Fairness",
    "groupFairness":       "Group Fairness",
    "calibrationError":    "Calibration",
    "counterfactual":      "Counterfactual Fairness",
    "intersectional":      "Intersectional Fairness",
    "privacy_score":       "Privacy",
    "robustness_score":    "Robustness",
    "transparency_score":  "Transparency",
    "accountability_score": "Accountability",
}


def _failing(key: str, value: float) -> bool:
    """Return True if this metric is below its threshold."""
    if value is None:
        return False
    t = THRESHOLDS.get(key)
    if t is None:
        return False
    # For error metrics (lower = better), failing means value > threshold
    if key in ("demographicParity", "calibrationError"):
        return value > t
    return value < t


def _improvement_estimate(key: str, current: float) -> float:
    """Rough estimate of how much improvement a standard fix would yield."""
    t = THRESHOLDS.get(key, 0.80)
    gap = abs(t - current)
    # Conservative: expect 50-70% of the gap to be closed
    return round(min(gap * 0.60, 0.20), 4)


REMEDIATION_LIBRARY = {
    "disparateImpact": [
        {
            "title": "Post-processing Threshold Calibration",
            "description": (
                "Adjust classification thresholds per demographic group so that the positive "
                "rate ratio meets the 0.80 threshold. This is the lowest-risk fix — model untouched."
            ),
            "code": (
                "from fairlearn.postprocessing import ThresholdOptimizer\n"
                "mitigator = ThresholdOptimizer(\n"
                "    estimator=model,\n"
                "    constraints='demographic_parity',\n"
                "    objective='balanced_accuracy_score'\n"
                ")\n"
                "mitigator.fit(X_train, y_train, sensitive_features=sensitive_col)"
            ),
            "effort": "Medium",
            "risk": "Low",
            "expected_improvement": "Bring disparate impact ratio from current value to ≥0.80",
        },
        {
            "title": "Reweighting Training Data",
            "description": "Assign higher weights to underrepresented groups during training.",
            "code": (
                "from fairlearn.reductions import ExponentiatedGradient, DemographicParity\n"
                "mitigator = ExponentiatedGradient(\n"
                "    estimator=base_model,\n"
                "    constraints=DemographicParity()\n"
                ")\n"
                "mitigator.fit(X_train, y_train, sensitive_features=sensitive_col)"
            ),
            "effort": "High",
            "risk": "Medium",
            "expected_improvement": "Typically reduces disparate impact gap by 40-60%",
        },
    ],
    "demographicParity": [
        {
            "title": "Adversarial Debiasing",
            "description": "Train an adversary to predict sensitive attributes from model outputs — penalizes the model for encoding them.",
            "code": (
                "# Use AIF360 AdversarialDebiasing\n"
                "from aif360.algorithms.inprocessing import AdversarialDebiasing\n"
                "# Or use fairlearn's ExponentiatedGradient with DemographicParity constraint"
            ),
            "effort": "High",
            "risk": "Medium",
            "expected_improvement": "Can reduce DP gap by 50-80% with accuracy trade-off of 1-3%",
        },
        {
            "title": "Label Balancing / Resampling",
            "description": "Oversample or undersample to balance positive outcome rates across groups.",
            "code": (
                "# Oversample minority-positive group\n"
                "minority_pos = df[(df[sensitive_col] == minority_val) & (df[target] == 1)]\n"
                "df_balanced = pd.concat([df, minority_pos.sample(n=target_count, replace=True)])"
            ),
            "effort": "Medium",
            "risk": "Low",
            "expected_improvement": "Directly reduces demographic parity gap in training data",
        },
    ],
    "individualFairness": [
        {
            "title": "Lipschitz Fairness Regularization",
            "description": "Add a regularization term that penalizes the model for giving different predictions to similar individuals.",
            "code": (
                "# Add to training loss:\n"
                "# L_fair = lambda * sum(|f(x_i) - f(x_j)| / dist(x_i, x_j))\n"
                "# for pairs (i,j) where dist is a task-appropriate similarity metric"
            ),
            "effort": "High",
            "risk": "Medium",
            "expected_improvement": "Improves consistency for similar individuals by 15-30%",
        },
    ],
    "groupFairness": [
        {
            "title": "Equalized Odds Post-Processing",
            "description": "Adjust decision thresholds to equalize true positive and false positive rates across groups.",
            "code": (
                "from fairlearn.postprocessing import ThresholdOptimizer\n"
                "mitigator = ThresholdOptimizer(\n"
                "    estimator=model,\n"
                "    constraints='equalized_odds'\n"
                ")\n"
                "mitigator.fit(X_val, y_val, sensitive_features=sensitive_col)"
            ),
            "effort": "Medium",
            "risk": "Low",
            "expected_improvement": "Equalizes opportunity across groups with minimal accuracy loss",
        },
    ],
    "calibrationError": [
        {
            "title": "Per-Group Platt Scaling",
            "description": "Apply separate Platt scaling calibration for each demographic group to equalize confidence accuracy.",
            "code": (
                "from sklearn.calibration import CalibratedClassifierCV\n"
                "for group in df[sensitive_col].unique():\n"
                "    mask = df[sensitive_col] == group\n"
                "    calibrator = CalibratedClassifierCV(method='sigmoid')\n"
                "    calibrator.fit(X_val[mask], y_val[mask])\n"
                "    # Use group-specific calibrator at inference time"
            ),
            "effort": "Medium",
            "risk": "Low",
            "expected_improvement": "Brings Brier score error below 0.05 threshold per group",
        },
    ],
    "counterfactual": [
        {
            "title": "Remove or De-Weight Proxy Features",
            "description": "Identify and remove features that act as proxies for protected attributes. Changing a protected attribute shouldn't change predictions if proxies are removed.",
            "code": (
                "# Step 1: Find proxy features\n"
                "proxy_corrs = df.corr()[sensitive_col].abs().sort_values(ascending=False)\n"
                "proxy_features = proxy_corrs[proxy_corrs > 0.4].index.tolist()\n\n"
                "# Step 2: Drop or de-weight them\n"
                "df_debiased = df.drop(columns=proxy_features)"
            ),
            "effort": "Medium",
            "risk": "Low",
            "expected_improvement": "Directly reduces prediction changes when protected attrs are flipped",
        },
    ],
    "intersectional": [
        {
            "title": "Stratified Resampling for Intersectional Groups",
            "description": "Ensure all intersectional subgroups (e.g., Female + SC caste) have sufficient representation.",
            "code": (
                "# Create intersectional group column\n"
                "df['intersect_group'] = df[attr1].astype(str) + '_' + df[attr2].astype(str)\n\n"
                "# Oversample rare intersectional groups\n"
                "min_count = df['intersect_group'].value_counts().min()\n"
                "df_balanced = df.groupby('intersect_group').apply(\n"
                "    lambda x: x.sample(min_count, replace=True)\n"
                ").reset_index(drop=True)"
            ),
            "effort": "High",
            "risk": "Low",
            "expected_improvement": "Reduces worst-case intersectional parity gap by 30-50%",
        },
    ],
    "privacy_score": [
        {
            "title": "PII Removal / Pseudonymization",
            "description": "Remove or hash personally identifiable information before model training and evaluation.",
            "code": (
                "import hashlib\n"
                "# Hash PII columns instead of removing\n"
                "for col in pii_columns:\n"
                "    df[col] = df[col].apply(\n"
                "        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8]\n"
                "    )"
            ),
            "effort": "Low",
            "risk": "Low",
            "expected_improvement": "Eliminates direct PII exposure — privacy score improvement ~0.30+",
        },
    ],
    "robustness_score": [
        {
            "title": "Confidence Calibration",
            "description": "Calibrate the model so predictions near the boundary (0.4–0.6) are more meaningful.",
            "code": (
                "from sklearn.calibration import CalibratedClassifierCV\n"
                "calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')\n"
                "calibrated_model.fit(X_val, y_val)"
            ),
            "effort": "Low",
            "risk": "Low",
            "expected_improvement": "Reduces uncertain-zone predictions — improves robustness by 10-20%",
        },
    ],
    "transparency_score": [
        {
            "title": "Feature Importance Documentation",
            "description": "Document and publish SHAP feature importance for all model versions.",
            "code": (
                "import shap\n"
                "explainer = shap.TreeExplainer(model)\n"
                "shap_values = explainer.shap_values(X_sample)\n"
                "shap.summary_plot(shap_values, X_sample, show=False)\n"
                "plt.savefig('shap_summary.png', bbox_inches='tight')"
            ),
            "effort": "Low",
            "risk": "Low",
            "expected_improvement": "Makes model fully explainable — transparency score to ≥0.80",
        },
    ],
    "accountability_score": [
        {
            "title": "Declare All Sensitive Attributes",
            "description": "Always specify protected attributes when running evaluations.",
            "code": (
                'sensitive_attributes = [\n'
                '    "gender", "race", "caste", "age", "religion"\n'
                ']\n'
                '# Pass to evaluation pipeline'
            ),
            "effort": "Low",
            "risk": "Low",
            "expected_improvement": "Improves accountability score by 0.15-0.20",
        },
    ],
}


def generate_remediation_plan(
    fairness_results: dict,
    privacy_results: dict,
    robustness_results: dict,
    transparency_results: dict,
    accountability_results: dict,
    ethical_score: float,
    sensitive_attrs: list,
    shap_results: dict,
) -> Dict[str, Any]:
    """
    Generate a prioritized remediation plan from all evaluation results.
    All suggestions are non-destructive — model is never touched.
    """
    all_fixes = []
    dimension_status = {}

    # Gather all scores to check
    scores = {
        "disparateImpact":     fairness_results.get("disparateImpact"),
        "demographicParity":   fairness_results.get("demographicParity"),
        "individualFairness":  fairness_results.get("individualFairness"),
        "groupFairness":       fairness_results.get("groupFairness"),
        "calibrationError":    fairness_results.get("calibrationError"),
        "counterfactual":      fairness_results.get("counterfactual"),
        "intersectional":      fairness_results.get("intersectional"),
        "privacy_score":       privacy_results.get("privacy_score") if privacy_results else None,
        "robustness_score":    robustness_results.get("robustness_score") if robustness_results else None,
        "transparency_score":  transparency_results.get("transparency_score") if transparency_results else None,
        "accountability_score": accountability_results.get("accountability_score") if accountability_results else None,
    }

    for key, value in scores.items():
        if value is None:
            continue
        label = DIMENSION_LABELS.get(key, key)
        failing = _failing(key, value)
        t = THRESHOLDS.get(key, 0.80)

        # Convert error metrics to score for display
        display_value = value
        if key in ("demographicParity", "calibrationError"):
            display_score = round(max(0.0, 1.0 - value), 4)
        else:
            display_score = value

        dimension_status[key] = {
            "label":    label,
            "value":    round(value, 4),
            "score":    display_score,
            "failing":  failing,
            "threshold": t,
        }

        if failing and key in REMEDIATION_LIBRARY:
            for i, fix in enumerate(REMEDIATION_LIBRARY[key]):
                improvement = _improvement_estimate(key, display_score)
                severity = "CRITICAL" if display_score < 0.60 else "WARNING"
                all_fixes.append({
                    "id":           f"{key}_{i}",
                    "dimension":    label,
                    "dimension_key": key,
                    "severity":     severity,
                    "current_score": display_score,
                    "threshold":    t if key not in ("demographicParity", "calibrationError") else f"≤ {t}",
                    "title":        fix["title"],
                    "description":  fix["description"],
                    "code":         fix["code"],
                    "effort":       fix["effort"],
                    "risk":         fix["risk"],
                    "expected_improvement": improvement,
                    "estimated_score_after": round(min(1.0, display_score + improvement), 4),
                })

    # Sort: CRITICAL first, then by score ascending (worst first)
    all_fixes.sort(key=lambda x: (
        0 if x["severity"] == "CRITICAL" else 1,
        x["current_score"]
    ))

    # Overall summary
    failing_count   = sum(1 for v in dimension_status.values() if v["failing"])
    passing_count   = sum(1 for v in dimension_status.values() if not v["failing"])
    critical_count  = sum(1 for f in all_fixes if f["severity"] == "CRITICAL")

    # Top 3 priority fixes
    quick_wins = [f for f in all_fixes if f["effort"] == "Low"][:3]
    high_impact = [f for f in all_fixes if f["severity"] == "CRITICAL"][:3]

    return {
        "ethical_score":      round(ethical_score, 4),
        "failing_dimensions": failing_count,
        "passing_dimensions": passing_count,
        "critical_issues":    critical_count,
        "dimension_status":   dimension_status,
        "all_fixes":          all_fixes,
        "quick_wins":         quick_wins,
        "high_impact_fixes":  high_impact,
        "total_fixes":        len(all_fixes),
        "deployment_blocked": ethical_score < 0.60 or critical_count > 0,
        "deployment_message": (
            "🔴 DEPLOYMENT BLOCKED — Critical fairness issues must be resolved first."
            if (ethical_score < 0.60 or critical_count > 0)
            else "🟡 CONDITIONAL DEPLOYMENT — Remediation recommended before production." 
            if ethical_score < 0.80
            else "✅ APPROVED — All dimensions pass ethical thresholds."
        ),
    }


def generate_counterfactual_examples(
    df,
    preds_df,
    target_col: str,
    sensitive_attrs: List[str],
    n_examples: int = 5,
) -> List[Dict[str, Any]]:
    """
    Find concrete counterfactual pairs from the dataset.
    Pairs where similar people get different outcomes,
    differing mainly on a sensitive attribute.
    
    No model access needed — uses existing predictions.
    """
    import pandas as pd

    examples = []

    # Get prediction column
    pred_col = None
    for c in preds_df.columns:
        if preds_df[c].between(0, 1).all():
            pred_col = c
            break
    if pred_col is None:
        return []

    preds_series = preds_df[pred_col].reset_index(drop=True)
    df_reset = df.reset_index(drop=True)

    # Get numeric columns (excluding target and sensitive)
    numeric_cols = df_reset.select_dtypes(include='number').columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col and c not in sensitive_attrs]

    if not numeric_cols or len(df_reset) < 10:
        return []

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors

        # Scale numeric features
        scaler = StandardScaler()
        X_num = df_reset[numeric_cols[:10]].fillna(0)  # cap at 10 features for speed
        X_scaled = scaler.fit_transform(X_num)

        # Build NN index
        nn = NearestNeighbors(n_neighbors=6, metric='euclidean')
        nn.fit(X_scaled)
        distances, indices = nn.kneighbors(X_scaled)

        for attr in sensitive_attrs[:2]:  # check first 2 sensitive attrs
            if attr not in df_reset.columns:
                continue

            found = 0
            for i in range(min(len(df_reset), 2000)):
                if found >= n_examples:
                    break
                pred_i = float(preds_series.iloc[i])

                for j_idx, j in enumerate(indices[i][1:], 1):
                    if j >= len(df_reset):
                        continue
                    pred_j = float(preds_series.iloc[j])

                    # Different outcomes
                    outcome_i = int(pred_i >= 0.5)
                    outcome_j = int(pred_j >= 0.5)
                    if outcome_i == outcome_j:
                        continue

                    # Different sensitive attribute value
                    val_i = str(df_reset[attr].iloc[i])
                    val_j = str(df_reset[attr].iloc[j])
                    if val_i == val_j:
                        continue

                    # Similar distance (nearby)
                    dist = float(distances[i][j_idx])
                    if dist > 2.5:
                        continue

                    # Build example
                    profile_i = {c: df_reset[c].iloc[i] for c in numeric_cols[:5]}
                    profile_j = {c: df_reset[c].iloc[j] for c in numeric_cols[:5]}

                    examples.append({
                        "attribute":     attr,
                        "person_a": {
                            "index":         int(i),
                            "sensitive_val": val_i,
                            "prediction":    round(pred_i, 4),
                            "outcome":       "APPROVED" if outcome_i else "REJECTED",
                            "profile":       {k: round(float(v), 2) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in profile_i.items()},
                        },
                        "person_b": {
                            "index":         int(j),
                            "sensitive_val": val_j,
                            "prediction":    round(pred_j, 4),
                            "outcome":       "APPROVED" if outcome_j else "REJECTED",
                            "profile":       {k: round(float(v), 2) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in profile_j.items()},
                        },
                        "similarity_distance": round(dist, 4),
                        "prediction_gap":      round(abs(pred_i - pred_j), 4),
                        "plain_english": (
                            f"Person A ({attr}={val_i}) was {('APPROVED' if outcome_i else 'REJECTED')} "
                            f"with {round(pred_i*100,1)}% confidence. "
                            f"Person B ({attr}={val_j}) with a nearly identical profile was "
                            f"{('APPROVED' if outcome_j else 'REJECTED')} "
                            f"with {round(pred_j*100,1)}% confidence. "
                            f"The only notable difference is their {attr}."
                        ),
                    })
                    found += 1
                    break

    except Exception as e:
        print(f"[counterfactual_examples] Error: {e}")

    return examples[:n_examples]
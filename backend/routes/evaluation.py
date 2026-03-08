from flask import Blueprint, request, jsonify
import pandas as pd
import os
import uuid
import threading

from services.fairness import compute_fairness
from services.shap_explainer import compute_shap
from services.model_loader import get_model, preprocess

evaluation_routes = Blueprint("evaluation_routes", __name__)

UPLOAD_FOLDER = "uploads"

# In-memory evaluation store — replace with Redis/DB once Prisma is wired up.
evaluations: dict = {}

VALID_REPORT_TYPES = {"developer", "regulator", "enduser", "all"}
REPORT_TYPE_MAP    = {
    "developer": "DEVELOPER",
    "regulator": "REGULATOR",
    "enduser":   "ENDUSER",
    "all":       "ALL",
}


def _find_file(uid: str) -> str:
    """Locate an uploaded file by its UUID prefix."""
    matches = [
        f for f in os.listdir(UPLOAD_FOLDER)
        if f.startswith(uid)
    ]
    if not matches:
        raise FileNotFoundError(f"No uploaded file found for id: {uid}")
    return os.path.join(UPLOAD_FOLDER, matches[0])


def _read_file(path: str) -> pd.DataFrame:
    """Read CSV / Excel / JSON into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _run_evaluation(
    eval_id: str,
    dataset_path: str,
    target_variable: str,
    prediction_variable: str,   # kept for backward compat but no longer required
    sensitive_attributes: list,
    fairness_weights: dict,
    report_type: str,
    model_id: str,
):
    try:
        # Step 1: Data validation — UNCHANGED
        evaluations[eval_id].update({"status": "running", "current_step": 1})
        df = _read_file(dataset_path)
        records = len(df)
        columns = list(df.columns)
        evaluations[eval_id]["records"] = records
        evaluations[eval_id]["columns"] = columns

        # Step 2: EDA — UNCHANGED
        evaluations[eval_id]["current_step"] = 2
        target_col = target_variable if target_variable in columns else None
        if target_col is None:
            for col in columns:
                if df[col].nunique() <= 5 and df[col].dtype != object:
                    target_col = col
                    break
        if target_col is None:
            raise ValueError(
                f"Could not determine target variable. "
                f"Please specify it explicitly. Available columns: {columns}"
            )
        evaluations[eval_id]["resolved_target"] = target_col

        # ── Step 3: CHANGED — use model to generate predictions ──────────────
        evaluations[eval_id]["current_step"] = 3

        try:
            # Attempt to use the pre-trained model from models/
            model, _, _, _ = get_model()
            feature_df = df.drop(columns=[target_col])
            X_input    = preprocess(feature_df)
            y_prob_arr = model.predict_proba(X_input)[:, 1]  # positive class probability

            # Write model predictions into df as a new column
            df["_prediction_score"] = y_prob_arr
            pred_col = "_prediction_score"
            evaluations[eval_id]["resolved_prediction"] = pred_col
            evaluations[eval_id]["prediction_source"]   = "pretrained_model"

        except Exception as model_err:
            # Fallback: read prediction column from dataset if model fails
            import traceback
            print(f"[model_loader fallback] {model_err}\n{traceback.format_exc()}")

            pred_col = None
            if prediction_variable and prediction_variable in columns:
                pred_col = prediction_variable
            else:
                for candidate in ["prediction", "prediction_score", "score", "label", "output", "pred"]:
                    if candidate in columns:
                        pred_col = candidate
                        break
            if pred_col is None:
                raise ValueError(
                    f"Pre-trained model failed AND no prediction column found. "
                    f"Model error: {model_err}. "
                    f"Available columns: {columns}"
                )
            evaluations[eval_id]["resolved_prediction"] = pred_col
            evaluations[eval_id]["prediction_source"]   = "dataset_column"

        # Resolve sensitive attributes — UNCHANGED
        valid_attrs = [a for a in (sensitive_attributes or []) if a in columns]
        if not valid_attrs:
            for col in columns:
                if col not in (target_col, pred_col) and df[col].dtype == object:
                    valid_attrs = [col]
                    break
        evaluations[eval_id]["resolved_attrs"] = valid_attrs

        # Step 4: Fairness analysis — ONE LINE CHANGED (drop pred_col not target_col)
        evaluations[eval_id]["current_step"] = 4

        preds_df = df[[pred_col]].rename(columns={pred_col: "prediction_score"})

        fairness_results = compute_fairness(
            df.drop(columns=[pred_col]),   # features + target, no pred col
            preds_df,
            target_col=target_col,
            sensitive_attrs=valid_attrs,
        )

        # ── Step 5: CHANGED — use model-generated y_prob instead of df[pred_col] ──
        evaluations[eval_id]["current_step"] = 5

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        import numpy as np

        y_true = df[target_col].astype(int)
        raw    = df[pred_col]                          # now always _prediction_score (0-1 probs)
        n      = min(len(y_true), len(raw))
        y_true = y_true.iloc[:n].reset_index(drop=True)
        y_prob = raw.iloc[:n].reset_index(drop=True)
        y_pred = (y_prob >= 0.5).astype(int)          # always threshold at 0.5 (model outputs probs)

        model_metrics = {}
        try:
            model_metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
        except Exception:
            pass
        try:
            model_metrics["f1_score"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        except Exception:
            pass
        try:
            model_metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            pass

        evaluations[eval_id]["model_metrics"] = model_metrics

        # ── Step 6: CHANGED — pass feature df without pred_col or target_col ──
        evaluations[eval_id]["current_step"] = 6

        shap_results = compute_shap(
            df.drop(columns=[pred_col], errors="ignore"),  
            target_col=target_col
        )
        # Step 7: Report generation — UNCHANGED
        evaluations[eval_id]["current_step"] = 7
        report_type_enum = REPORT_TYPE_MAP.get(report_type.lower(), "DEVELOPER")
        ethical_score    = fairness_results["fairness_score"]

        sensitive_attr_records  = [{"name": a} for a in valid_attrs]
        fairness_weight_records = [
            {"dimension": k, "weight": v}
            for k, v in (fairness_weights or {}).items()
        ]

        evaluations[eval_id].update({
            "status":       "complete",
            "current_step": 7,
            "ethical_score":  ethical_score,
            "report_type":    report_type_enum,
            "report_type_original": report_type,
            "model_id":       model_id,
            "fairness": {
                "individualFairness": fairness_results["individualFairness"],
                "groupFairness":      fairness_results["groupFairness"],
                "demographicParity":  fairness_results["demographicParity"],
                "disparateImpact":    fairness_results["disparateImpact"],
                "calibrationError":   fairness_results["calibrationError"],
                "counterfactual":     fairness_results["counterfactual"],
                "intersectional":     fairness_results["intersectional"],
                "fairness_score":     ethical_score,
                "per_attribute":      fairness_results.get("per_attribute", {}),
                "records_evaluated":  fairness_results["records_evaluated"],
                "target_column":      fairness_results["target_column"],
                "prediction_column":  pred_col,
            },
            "shap": {
                "topFeature":         shap_results["topFeature"],
                "shapMax":            shap_results["shapMax"],
                "shapMin":            shap_results["shapMin"],
                "featureStability":   shap_results["featureStability"],
                "feature_importance": shap_results.get("feature_importance", {}),
            },
            "sensitive_attributes": sensitive_attr_records,
            "fairness_weights":     fairness_weight_records,
            "model_metrics":        model_metrics,
        })

    except Exception as e:
        evaluations[eval_id].update({"status": "error", "error": str(e)})
        import traceback
        print(f"[evaluation error] {eval_id}:\n{traceback.format_exc()}")


# ── Routes ────────────────────────────────────────────────────────────────────

@evaluation_routes.route("/evaluate", methods=["POST"])
def start_evaluation():
    """
    Start an async evaluation.

    Required body fields:
        dataset_id           – from POST /datasets
        model_id             – from POST /models
        target_variable      – ground-truth label column name in dataset
        prediction_variable  – prediction/score column name in dataset
        report_type          – developer | regulator | enduser | all
        sensitive_attributes – list of column names
        fairness_weights     – dict {dimension: weight}

    Note: prediction_id is NO LONGER required. Predictions are read directly
    from the dataset using the prediction_variable column name.
    """
    data = request.json or {}

    dataset_id           = data.get("dataset_id")
    model_id             = data.get("model_id")
    sensitive_attributes = data.get("sensitive_attributes", [])
    fairness_weights     = data.get("fairness_weights", {})
    report_type          = data.get("report_type", "developer").lower()
    target_variable      = (data.get("target_variable")     or "").strip()
    prediction_variable  = (data.get("prediction_variable") or "").strip()

    # Validate required fields
    missing = [f for f, v in [
        ("dataset_id",      dataset_id),
        ("model_id",        model_id),
        ("target_variable", target_variable),
    ] if not v]
    if missing:
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing)}",
            "tip":   "Provide target_variable and prediction_variable as column names from your dataset."
        }), 400

    if report_type not in VALID_REPORT_TYPES:
        return jsonify({
            "error": f"Invalid report_type '{report_type}'. Must be one of: {VALID_REPORT_TYPES}"
        }), 400

    # Locate uploaded dataset
    try:
        dataset_path = _find_file(dataset_id)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    # Create evaluation record
    eval_id = str(uuid.uuid4())
    evaluations[eval_id] = {
        "status":              "queued",
        "current_step":        0,
        "records":             0,
        "dataset_id":          dataset_id,
        "model_id":            model_id,
        "target_variable":     target_variable,
        "prediction_variable": prediction_variable,
        "report_type":         REPORT_TYPE_MAP.get(report_type, "DEVELOPER"),
    }

    # Launch background thread
    thread = threading.Thread(
        target=_run_evaluation,
        args=(
            eval_id,
            dataset_path,
            target_variable,
            prediction_variable,
            sensitive_attributes,
            fairness_weights,
            report_type,
            model_id,
        ),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "evaluation_id":  eval_id,
        "status":         "queued",
        "estimated_time": "2-3 minutes",
        "poll_url":       f"/evaluate/{eval_id}",
    }), 202


@evaluation_routes.route("/evaluate/<eval_id>", methods=["GET"])
def get_evaluation(eval_id: str):
    """
    Poll evaluation progress.
    Returns the full result when status = 'complete'.
    """
    result = evaluations.get(eval_id)
    if result is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404

    return jsonify({"evaluation_id": eval_id, **result}), 200


@evaluation_routes.route("/evaluate", methods=["GET"])
def list_evaluations():
    """List all evaluations (summary only)."""
    summary = [
        {
            "evaluation_id": eid,
            "status":        v.get("status"),
            "current_step":  v.get("current_step"),
            "ethical_score": v.get("ethical_score"),
            "report_type":   v.get("report_type"),
        }
        for eid, v in evaluations.items()
    ]
    return jsonify({"evaluations": summary, "count": len(summary)}), 200
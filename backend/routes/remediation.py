"""
routes/remediation.py
Bias Remediation Toolkit API + CI/CD deployment gate.

Endpoints:
  GET  /remediation/<eval_id>        → full remediation plan
  GET  /deploy/check/<eval_id>       → deployment gate (pass/block)
  GET  /deploy/regression/<eval_id>  → regression vs previous eval
"""

from flask import Blueprint, jsonify, request
from routes.evaluation import evaluations

remediation_routes = Blueprint("remediation_routes", __name__)


@remediation_routes.route("/remediation/<eval_id>", methods=["GET"])
def get_remediation(eval_id: str):
    """
    GET /remediation/<eval_id>

    Full remediation plan with:
    - Prioritized fix list
    - Before/after score estimates
    - Code snippets
    - Counterfactual examples
    - Quick wins vs high impact fixes
    """
    ev = evaluations.get(eval_id)
    if ev is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404

    status = ev.get("status", "queued")
    if status in ("queued", "running"):
        return jsonify({"evaluation_id": eval_id, "status": status}), 200
    if status == "error":
        return jsonify({"evaluation_id": eval_id, "status": "error", "error": ev.get("error")}), 200

    remediation = ev.get("remediation", {})
    cf_examples = ev.get("counterfactual_examples", [])
    fairness    = ev.get("fairness", {})
    privacy     = ev.get("privacy", {})
    robustness  = ev.get("robustness", {})
    transparency = ev.get("transparency", {})
    accountability = ev.get("accountability", {})

    # Aggregate all findings across dimensions
    all_findings = []
    for dim_key, dim_data in [
        ("Privacy",       privacy),
        ("Robustness",    robustness),
        ("Transparency",  transparency),
        ("Accountability", accountability),
    ]:
        if isinstance(dim_data, dict):
            for f in dim_data.get("findings", []):
                all_findings.append({**f, "dimension": dim_key})

    # Aggregate all recommendations
    all_recs = []
    for dim_key, dim_data in [
        ("Privacy",       privacy),
        ("Robustness",    robustness),
        ("Transparency",  transparency),
        ("Accountability", accountability),
    ]:
        if isinstance(dim_data, dict):
            for r in dim_data.get("recommendations", []):
                all_recs.append({**r, "dimension": dim_key})

    payload = {
        "evaluation_id":           eval_id,
        "status":                  "complete",

        # Core remediation plan
        "remediation":             remediation,

        # Counterfactual examples (concrete "what-if" pairs)
        "counterfactual_examples": cf_examples,

        # Findings from new dimensions
        "findings":                all_findings,
        "recommendations":         all_recs,

        # Score summary
        "scores": {
            "overall":        ev.get("ethical_score", 0),
            "fairness":       ev.get("fairness_score"),
            "privacy":        ev.get("privacy_score"),
            "robustness":     ev.get("robustness_score"),
            "transparency":   ev.get("transparency_score"),
            "accountability": ev.get("accountability_score"),
        },

        # Deployment status
        "deployment_blocked":  remediation.get("deployment_blocked", False),
        "deployment_message":  remediation.get("deployment_message", ""),
        "records":             ev.get("records", 0),
        "model_id":            ev.get("model_id", ""),
    }

    return jsonify(payload), 200


@remediation_routes.route("/deploy/check/<eval_id>", methods=["GET"])
def deployment_check(eval_id: str):
    """
    GET /deploy/check/<eval_id>?threshold=0.75

    CI/CD deployment gate.
    Returns PASS / BLOCK with reason.

    Query params:
        threshold (float, default 0.75) — minimum ethical score to pass
    """
    ev = evaluations.get(eval_id)
    if ev is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404

    status = ev.get("status", "queued")
    if status != "complete":
        return jsonify({
            "evaluation_id": eval_id,
            "gate_status":   "PENDING",
            "reason":        f"Evaluation not complete (status: {status})",
        }), 200

    try:
        threshold = float(request.args.get("threshold", 0.75))
    except (TypeError, ValueError):
        threshold = 0.75

    overall_score = ev.get("ethical_score", 0.0)
    remediation   = ev.get("remediation", {})
    critical      = remediation.get("critical_issues", 0)
    failing_dims  = remediation.get("failing_dimensions", 0)

    # Gate logic
    passed = overall_score >= threshold and critical == 0

    # Build detailed check list for CI/CD output
    checks = [
        {
            "name":    "Overall Ethical Score",
            "passed":  overall_score >= threshold,
            "value":   round(overall_score * 100, 1),
            "threshold": round(threshold * 100, 1),
            "unit":    "%",
        },
        {
            "name":    "Critical Issues",
            "passed":  critical == 0,
            "value":   critical,
            "threshold": 0,
            "unit":    "issues",
        },
        {
            "name":    "Failing Dimensions",
            "passed":  failing_dims <= 1,
            "value":   failing_dims,
            "threshold": 1,
            "unit":    "dimensions",
        },
        {
            "name":    "Fairness Score",
            "passed":  (ev.get("fairness_score") or 0) >= 0.70,
            "value":   round((ev.get("fairness_score") or 0) * 100, 1),
            "threshold": 70,
            "unit":    "%",
        },
        {
            "name":    "Privacy Score",
            "passed":  (ev.get("privacy_score") or 0.5) >= 0.65,
            "value":   round((ev.get("privacy_score") or 0.5) * 100, 1),
            "threshold": 65,
            "unit":    "%",
        },
    ]

    checks_passed = sum(1 for c in checks if c["passed"])
    checks_failed = len(checks) - checks_passed

    return jsonify({
        "evaluation_id":  eval_id,
        "gate_status":    "PASS" if passed else "BLOCK",
        "gate_icon":      "✅" if passed else "🔴",
        "passed":         passed,
        "overall_score":  round(overall_score * 100, 1),
        "threshold_used": round(threshold * 100, 1),
        "critical_issues": critical,
        "failing_dimensions": failing_dims,
        "checks":         checks,
        "checks_passed":  checks_passed,
        "checks_failed":  checks_failed,
        "reason": (
            "All ethical thresholds met — approved for deployment."
            if passed else
            f"Deployment blocked: {checks_failed} check(s) failed. "
            f"Score: {round(overall_score*100,1)}% < threshold {round(threshold*100,1)}%."
            + (f" {critical} critical issue(s) must be resolved." if critical > 0 else "")
        ),
        "remediation_url": f"/remediation/{eval_id}",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
    }), 200


@remediation_routes.route("/deploy/regression/<eval_id>", methods=["GET"])
def regression_check(eval_id: str):
    """
    GET /deploy/regression/<eval_id>?baseline=<baseline_eval_id>

    Compare new evaluation to a baseline.
    Returns dimension-by-dimension regression analysis.
    If no baseline provided, uses the previous completed evaluation.
    """
    ev = evaluations.get(eval_id)
    if ev is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404

    baseline_id = request.args.get("baseline")

    # Find baseline: explicit param or previous completed eval
    baseline_ev = None
    if baseline_id:
        baseline_ev = evaluations.get(baseline_id)
    else:
        # Find the most recent completed eval that isn't this one
        completed = [
            (eid, e) for eid, e in evaluations.items()
            if e.get("status") == "complete" and eid != eval_id
        ]
        if completed:
            baseline_id, baseline_ev = completed[-1]

    if baseline_ev is None:
        return jsonify({
            "evaluation_id": eval_id,
            "baseline_id":   baseline_id,
            "has_baseline":  False,
            "message":       "No baseline evaluation found. Run a second evaluation to enable regression testing.",
        }), 200

    # Compare scores
    score_keys = [
        ("ethical_score",        "Overall Ethics"),
        ("fairness_score",       "Fairness"),
        ("privacy_score",        "Privacy"),
        ("robustness_score",     "Robustness"),
        ("transparency_score",   "Transparency"),
        ("accountability_score", "Accountability"),
    ]

    regressions = []
    improvements = []
    neutral = []

    for key, label in score_keys:
        new_val  = ev.get(key) or 0
        base_val = baseline_ev.get(key) or 0
        delta    = round(new_val - base_val, 4)
        delta_pct = round(delta * 100, 2)

        row = {
            "dimension":   label,
            "key":         key,
            "baseline":    round(base_val, 4),
            "current":     round(new_val, 4),
            "delta":       delta,
            "delta_pct":   delta_pct,
            "direction":   "improvement" if delta > 0.005 else "regression" if delta < -0.005 else "stable",
            "delta_label": f"+{delta_pct}% ↑" if delta > 0.005 else f"{delta_pct}% ↓" if delta < -0.005 else "— stable",
        }

        if delta < -0.005:
            regressions.append(row)
        elif delta > 0.005:
            improvements.append(row)
        else:
            neutral.append(row)

    has_regression = len(regressions) > 0
    overall_delta  = round(
        (ev.get("ethical_score") or 0) - (baseline_ev.get("ethical_score") or 0),
        4
    )

    return jsonify({
        "evaluation_id": eval_id,
        "baseline_id":   baseline_id,
        "has_baseline":  True,
        "has_regression": has_regression,
        "overall_delta":  overall_delta,
        "overall_direction": "improvement" if overall_delta > 0.005 else "regression" if overall_delta < -0.005 else "stable",
        "regressions":    regressions,
        "improvements":   improvements,
        "neutral":        neutral,
        "all_dimensions": regressions + improvements + neutral,
        "verdict": (
            f"⚠️ REGRESSION DETECTED in {len(regressions)} dimension(s). "
            f"Overall score changed by {round(overall_delta*100,2)}%."
        ) if has_regression else (
            f"✅ No regression detected. "
            f"Overall score {'improved' if overall_delta > 0 else 'stable'} by {abs(round(overall_delta*100,2))}%."
        ),
        "block_deployment": has_regression and overall_delta < -0.05,
    }), 200
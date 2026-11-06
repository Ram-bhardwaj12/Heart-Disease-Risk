import os

os.environ.setdefault("MPLBACKEND", "Agg")

"""FastAPI application for CardioLens.

Stateless, zero request body logging, strict security headers, and input validation.
"""

import json
from pathlib import Path
from typing import Any
import urllib.parse

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from engine.content import (
    DISCLAIMER,
    INFERRED_CODES,
    INFERRED_CODES_NOTE,
    OPTIONAL_FIELDS,
    PRODUCT_NAME,
    REQUIRED_FIELDS,
    TAGLINE,
    UNVERIFIED_CODES,
    build_summary,
)
from engine.core import (
    CONTINUOUS_NUMERIC_FEATURES,
    REPO_ROOT,
    explain_grouped,
    get_cutoffs,
    get_feature_specs,
    load_data_audit_summary,
    load_evaluation_summary,
    load_shap_summary,
    predict_score,
    risk_tier,
    tree_agreement,
    validate_inputs,
    what_if_curve,
)

MAX_BODY_SIZE = 20 * 1024  # 20 KB limit

app = FastAPI(
    title=PRODUCT_NAME,
    description=TAGLINE,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# -----------------------------------------------------------------------------
# Middleware: Body-Size Limit, No-Store, Security Headers
# -----------------------------------------------------------------------------


@app.middleware("http")
async def security_and_limit_middleware(request: Request, call_next):
    # Reject oversized requests
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BODY_SIZE:
                return Response(content="Payload Too Large", status_code=413)
        except ValueError:
            return Response(content="Invalid Content-Length", status_code=400)

    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        if len(body) > MAX_BODY_SIZE:
            return Response(content="Payload Too Large", status_code=413)

    response = await call_next(request)

    if request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = str(err["loc"][-1]) if err["loc"] else "payload"
        errors.append({"field": field, "message": err["msg"]})
    return JSONResponse(status_code=422, content={"errors": errors})


# -----------------------------------------------------------------------------
# Request Models (extra="forbid")
# -----------------------------------------------------------------------------


class PredictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: dict[str, Any]


class WhatIfPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: dict[str, Any]
    feature: str
    n_points: int = Field(default=25, ge=5, le=100)


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/schema")
def get_schema():
    """Field schema, product metadata, decision cut-offs, and inferred codes."""
    specs = get_feature_specs()
    low_c, high_c = get_cutoffs()

    fields_list = list(specs.values())

    return {
        "product_name": PRODUCT_NAME,
        "tagline": TAGLINE,
        "disclaimer": DISCLAIMER,
        "cutoffs": {"low": low_c, "high": high_c},
        "inferred_codes": INFERRED_CODES,
        "inferred_codes_note": INFERRED_CODES_NOTE,
        "unverified_codes": UNVERIFIED_CODES,
        "fields": fields_list,
        "field_specs": specs,
    }


@app.post("/api/predict")
def predict_endpoint(payload: PredictPayload):
    """Predict risk score and return tier, local contributions, and summary."""
    val_res = validate_inputs(payload.inputs)
    if not val_res.is_valid:
        return JSONResponse(
            status_code=422,
            content={"errors": val_res.errors},
        )

    score = predict_score(val_res.cleaned_inputs)
    score_pct = round(score * 100.0, 1)
    tier_info = risk_tier(score)
    low_c, high_c = get_cutoffs()

    df_grouped = explain_grouped(val_res.cleaned_inputs)
    base_val = float(df_grouped.attrs.get("base_value", 0.0))

    contributions = []
    for _, r in df_grouped.iterrows():
        amt = float(r["amount"])
        contributions.append({
            "key": r["key"],
            "label": r["label"],
            "short_label": str(r.get("short_label", r["label"])),
            "value_display": str(r["value_display"]),
            "amount": amt,
            "direction": "raised" if amt >= 0 else "lowered",
        })
    contributions.sort(key=lambda x: abs(x["amount"]), reverse=True)

    raising = [c for c in contributions if c["direction"] == "raised" and c["amount"] >= 0.01]
    raising.sort(key=lambda x: x["amount"], reverse=True)
    top_raising = raising[:3]

    lowering = [c for c in contributions if c["direction"] == "lowered" and abs(c["amount"]) >= 0.01]
    lowering.sort(key=lambda x: abs(x["amount"]), reverse=True)
    top_lowering = lowering[:3]

    summary_sentence = build_summary(score, tier_info, top_raising, top_lowering)

    tree_scores, share_ge_05, mean_score = tree_agreement(val_res.cleaned_inputs)
    model_agreement = {
        "tree_count": int(len(tree_scores)),
        "share_at_or_above_half": float(share_ge_05),
        "mean_score": float(mean_score),
    }

    return {
        "score": score,
        "score_percent": score_pct,
        "tier": {
            "key": tier_info["key"],
            "label": tier_info["label"],
            "text": tier_info["text"],
        },
        "cutoffs": {"low": low_c, "high": high_c},
        "warnings": val_res.warnings,
        "missing_used": val_res.missing_used,
        "summary": summary_sentence,
        "contributions": contributions,
        "top_raising": top_raising,
        "top_lowering": top_lowering,
        "base_value": base_val,
        "model_agreement": model_agreement,
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/whatif")
def whatif_endpoint(payload: WhatIfPayload):
    """Compute sensitivity curve by varying one continuous numeric feature."""
    feature = payload.feature
    if feature not in CONTINUOUS_NUMERIC_FEATURES:
        return JSONResponse(
            status_code=422,
            content={
                "errors": [
                    {
                        "field": "feature",
                        "message": (
                            f"Feature '{feature}' is not an allowed continuous numeric feature for what-if analysis. "
                            f"Allowed features: {CONTINUOUS_NUMERIC_FEATURES}."
                        ),
                    }
                ]
            },
        )

    val_res = validate_inputs(payload.inputs)
    if not val_res.is_valid:
        return JSONResponse(
            status_code=422,
            content={"errors": val_res.errors},
        )

    df_curve = what_if_curve(val_res.cleaned_inputs, feature, n_points=payload.n_points)
    low_c, high_c = get_cutoffs()

    curr_val = float(val_res.cleaned_inputs[feature])
    curr_score = predict_score(val_res.cleaned_inputs)

    curve_list = [
        {"value": float(row["value"]), "score": float(row["score"])}
        for _, row in df_curve.iterrows()
    ]

    return {
        "feature": feature,
        "current_value": curr_val,
        "current_score": curr_score,
        "cutoffs": {"low": low_c, "high": high_c},
        "curve": curve_list,
    }


@app.get("/api/model-info")
def model_info_endpoint():
    """Retrieve comprehensive model metadata and evaluation reports."""
    data_audit = load_data_audit_summary()
    eval_summary = load_evaluation_summary()
    shap_summ = load_shap_summary()

    # Training history
    th_path = REPO_ROOT / "models" / "training_history.json"
    training_history = {}
    if th_path.exists():
        with open(th_path, "r", encoding="utf-8") as f:
            training_history = json.load(f)

    # Figures list
    fig_dir = REPO_ROOT / "reports" / "figures"
    fig_files = sorted([p.name for p in fig_dir.glob("*.png")]) if fig_dir.exists() else []
    figure_urls = [f"/figures/{fname}" for fname in fig_files]

    # Dataset facts
    dataset_facts = {
        "raw_rows": data_audit.get("raw_rows"),
        "unique_rows": data_audit.get("unique_rows"),
        "duplicate_rows": data_audit.get("duplicate_rows"),
        "class_balance": data_audit.get("class_balance_after"),
    }

    # Cross-validation table
    cv_table = {}
    if "models" in training_history:
        for m_name, m_data in training_history["models"].items():
            cv_table[m_name] = m_data.get("cv_metrics_at_best_index", {})

    # Held-out test table with 95% CIs
    primary_res = eval_summary.get("primary_results_threshold_0_5", {})
    operating_point = eval_summary.get("recall_oriented_operating_point", {})

    # Confusion matrices at 0.5 and operating threshold
    confusion_matrices = {
        "threshold_0_5": {
            "logistic_regression": primary_res.get("logistic_regression", {}).get(
                "confusion_matrix", {}
            ),
            "random_forest": primary_res.get("random_forest", {}).get("confusion_matrix", {}),
        },
        "operating_point": {
            "logistic_regression": operating_point.get("chosen_threshold", {})
            .get("logistic_regression", {})
            .get("secondary_test_result", {})
            .get("confusion_matrix", {}),
            "random_forest": operating_point.get("chosen_threshold", {})
            .get("random_forest", {})
            .get("secondary_test_result", {})
            .get("confusion_matrix", {}),
        },
    }

    # Paired bootstrap difference (RF - LR)
    bootstrap_diff = eval_summary.get("bootstrap_comparison", {}).get("diff_rf_minus_lr", {})

    # Top 5 grouped SHAP features
    top5_shap = shap_summ.get("rankings", {}).get("grouped_13_original_features", [])[:5]

    return {
        "dataset_facts": dataset_facts,
        "cross_validation": cv_table,
        "held_out_test": primary_res,
        "confusion_matrices": confusion_matrices,
        "random_forest_minus_logistic_regression": bootstrap_diff,
        "top_5_grouped_shap": top5_shap,
        "figure_urls": figure_urls,
    }


# -----------------------------------------------------------------------------
# Static Figure Serving (Read-only, GET only, traversal protected)
# -----------------------------------------------------------------------------


@app.get("/figures/{filename:path}")
def serve_figure(filename: str):
    """Serve figures strictly from reports/figures allowlist; prevent path traversal."""
    unquoted = urllib.parse.unquote(filename)
    if ".." in unquoted or "/" in unquoted or "\\" in unquoted:
        raise HTTPException(status_code=404, detail="File not found")

    fig_dir = REPO_ROOT / "reports" / "figures"
    fig_path = (fig_dir / unquoted).resolve()

    # Allowlist check: Must exist directly inside reports/figures and end in .png
    if not (fig_path.exists() and fig_path.parent == fig_dir.resolve() and fig_path.suffix == ".png"):
        raise HTTPException(status_code=404, detail="Figure not found")

    return FileResponse(fig_path, media_type="image/png")


# -----------------------------------------------------------------------------
# Mount Static Web Assets at Root
# -----------------------------------------------------------------------------

web_dir = REPO_ROOT / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

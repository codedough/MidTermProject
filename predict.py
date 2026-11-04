#!/usr/bin/env python3
"""
Web service for the produce shortage early-warning model.

    GET  /health    service status and model info
    POST /predict   probability of a shortage `horizon` months after `month`

Run locally:
    python predict.py                                   # Flask dev server on port 9696
    gunicorn --bind 0.0.0.0:9696 predict:app            # production-style server

Request body (JSON). Required: store_id, product, horizon, month, avail.
Every other feature is optional; missing ones are treated as unknown, but
supplying them gives better predictions. See README.md for the full list.
"""
import math
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import sklearn
from flask import Flask, jsonify, request

import produce_features as pf

MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "shortage_model.bin"))

with open(MODEL_PATH, "rb") as f:
    BUNDLE = pickle.load(f)

if BUNDLE["sklearn_version"] != sklearn.__version__:
    warnings.warn(f"model was trained with scikit-learn {BUNDLE['sklearn_version']} "
                  f"but {sklearn.__version__} is installed; results may differ")

REQUIRED = ["store_id", "product", "horizon", "month", "avail"]
# Optional numeric inputs: everything the model uses except the two features
# the service computes itself (target_month_sin / target_month_cos).
OPTIONAL = [c for c in pf.ALL_NUM
            if c not in ("avail", "target_month_sin", "target_month_cos")]
PERCENT_FIELDS = ["avail", "avail_lag1", "avail_lag2", "avail_lag3", "avail_lag12"]
COUNT_FIELDS = ["frost_days", "frost_days_3m", "heatwave_days", "heatwave_days_3m"]
ALLOWED = set(REQUIRED) | set(OPTIONAL)

app = Flask(__name__)


class BadRequest(ValueError):
    pass


def number(payload, key):
    v = payload[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise BadRequest(f"'{key}' must be a finite number")
    return float(v)


def build_row(payload):
    """Validate the request and return (one-row DataFrame, horizon, missing feature names)."""
    if not isinstance(payload, dict):
        raise BadRequest("request body must be a JSON object")

    unknown = sorted(set(payload) - ALLOWED)
    if unknown:
        raise BadRequest(f"unknown field(s): {unknown}")
    missing_required = [k for k in REQUIRED if k not in payload]
    if missing_required:
        raise BadRequest(f"missing required field(s): {missing_required}")

    store, product = payload["store_id"], payload["product"]
    if (store, product) not in BUNDLE["valid_pairs"]:
        raise BadRequest(f"unknown store/product combination ({store!r}, {product!r}); "
                         f"stores: {BUNDLE['stores']}, products: {BUNDLE['products']}")

    h = number(payload, "horizon")
    if h != int(h) or int(h) not in BUNDLE["horizons"]:
        raise BadRequest(f"'horizon' must be one of {BUNDLE['horizons']}")
    h = int(h)

    month = number(payload, "month")
    if month != int(month) or not 1 <= int(month) <= 12:
        raise BadRequest("'month' must be an integer from 1 to 12 (the current month)")
    month = int(month)

    row = {"store_id": store, "product": product}
    for key in ["avail"] + OPTIONAL:
        if key in payload and payload[key] is not None:
            row[key] = number(payload, key)
            if key in PERCENT_FIELDS and not 0 <= row[key] <= 100:
                raise BadRequest(f"'{key}' must be between 0 and 100")
            if key in COUNT_FIELDS and not 0 <= row[key] <= 31:
                raise BadRequest(f"'{key}' must be between 0 and 31")
        else:
            row[key] = np.nan
    row["target_month_sin"], row["target_month_cos"] = (float(x) for x in pf.target_month_features(month, h))

    missing = [k for k in OPTIONAL if np.isnan(row[k])]
    return pd.DataFrame([row])[BUNDLE["features"]], h, month, missing


@app.get("/health")
def health():
    return jsonify(status="ok", horizons=BUNDLE["horizons"], trained_at=BUNDLE["trained_at"],
                   sklearn_version=BUNDLE["sklearn_version"], refit_all=BUNDLE["refit_all"])


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="request body must be valid JSON"), 400
    try:
        X, h, month, missing = build_row(payload)
    except BadRequest as e:
        return jsonify(error=str(e)), 400

    entry = BUNDLE["models"][h]
    prob = float(entry["model"].predict_proba(X)[0, 1])
    return jsonify(
        store_id=payload["store_id"],
        product=payload["product"],
        horizon=h,
        month_predicted=(month - 1 + h) % 12 + 1,
        shortage_probability=round(prob, 4),
        shortage_warning=bool(prob >= entry["threshold"]),
        threshold=round(entry["threshold"], 4),
        missing_features=missing,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)

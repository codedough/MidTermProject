"""
Load the saved model and predict shortage risk. Used by serve.py, or from the command line:

    python predict.py sample_request.json
    python predict.py '{"store_id":"S04","product":"strawberries","horizon":2,"month":9,"avail":84.5}'

Input: what is known at the end of month t (see README). Output: the probability of
a shortage `horizon` months later and a yes/no warning at the validated threshold.
"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

import produce_features as pf

MODEL_PATH = Path(__file__).resolve().parent / "model" / "shortage_model.bin"

REQUIRED = ["store_id", "product", "horizon", "month", "avail"]
# numeric inputs the caller may supply (the two month-of-year features are computed here)
OPTIONAL = [c for c in pf.ALL_NUM if c != "avail" and not c.startswith("target_month")]
PERCENT = {"avail", "avail_lag1", "avail_lag2", "avail_lag3", "avail_lag12"}          # 0-100
DAYS = {"frost_days", "frost_days_3m", "heatwave_days", "heatwave_days_3m"}           # 0-31


class InputError(ValueError):
    """The request cannot be scored (unknown value, out of range, ...)."""


def load_bundle(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def _number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InputError(f"{name} must be a finite number")
    return float(value)


def validate(bundle, record):
    """Return (clean numeric dict, store, product, horizon, month) or raise InputError."""
    if not isinstance(record, dict):
        raise InputError("request must be a JSON object")
    unknown = sorted(set(record) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise InputError(f"unknown field(s): {', '.join(unknown)}")
    missing = [k for k in REQUIRED if record.get(k) is None]
    if missing:
        raise InputError(f"missing required field(s): {', '.join(missing)}")

    store, product = record["store_id"], record["product"]
    if store not in bundle["stores"]:
        raise InputError(f"store_id must be one of {bundle['stores']}")
    if product not in bundle["products"]:
        raise InputError(f"product must be one of {bundle['products']}")
    if (store, product) not in bundle["valid_pairs"]:
        sold = [p for s, p in bundle["valid_pairs"] if s == store]
        raise InputError(f"store {store} does not carry {product}; it carries {sold}")

    horizon, month = record["horizon"], record["month"]
    if horizon not in bundle["horizons"] or isinstance(horizon, bool):
        raise InputError(f"horizon must be one of {bundle['horizons']}")
    if isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12:
        raise InputError("month must be an integer from 1 to 12")

    nums = {}
    for name in ["avail"] + OPTIONAL:
        if record.get(name) is None:
            continue
        x = _number(name, record[name])
        if name in PERCENT and not 0 <= x <= 100:
            raise InputError(f"{name} must be between 0 and 100")
        if name in DAYS and not 0 <= x <= 31:
            raise InputError(f"{name} must be between 0 and 31")
        nums[name] = x
    return nums, store, product, int(horizon), int(month)


def predict(bundle, record):
    nums, store, product, horizon, month = validate(bundle, record)
    sin, cos = pf.target_month_features(month, horizon)
    nums["target_month_sin"], nums["target_month_cos"] = float(sin), float(cos)
    features = {c: nums.get(c, np.nan) for c in bundle["numeric_features"]}   # NaN = unknown
    features.update(store_id=store, product=product)

    entry = bundle["models"][horizon]
    p = float(entry["model"].predict_proba([features])[0, 1])
    return {
        "store_id": store, "product": product, "horizon": horizon,
        "month_predicted": (month - 1 + horizon) % 12 + 1,
        "shortage_probability": round(p, 4),
        "shortage_warning": bool(p >= entry["threshold"]),
        "threshold": round(entry["threshold"], 4),
        "missing_features": sorted(c for c in bundle["numeric_features"]
                                   if c not in nums and not c.startswith("target_month")),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    arg = sys.argv[1]
    text = Path(arg).read_text() if Path(arg).is_file() else arg
    try:
        print(json.dumps(predict(load_bundle(), json.loads(text)), indent=2))
    except (InputError, json.JSONDecodeError) as e:
        sys.exit(f"error: {e}")

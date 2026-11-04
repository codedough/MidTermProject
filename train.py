#!/usr/bin/env python3
"""
Train the final shortage early-warning models and save them to a file.

One gradient-boosting model is trained for each horizon (1, 2 and 3 months
ahead). For each horizon the script:
  1. fits on the training years and picks a decision threshold on the
     validation years (the threshold that maximises F1),
  2. refits on train + validation years,
  3. scores the untouched test years once and prints the metrics,
  4. saves everything (models, thresholds, metrics, feature list, allowed
     store/product values) in one pickle file for predict.py.

Hyperparameters come from model_params.json, which the notebook writes after
tuning. If the file is missing, defaults are used.

Usage:
    python train.py
    python train.py --refit-all      # final model also learns from the test years
"""
import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

import produce_features as pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_produce_prices_1990_2025.csv")
    ap.add_argument("--params", default="model_params.json")
    ap.add_argument("--out", default="model/shortage_model.bin")
    ap.add_argument("--refit-all", action="store_true",
                    help="refit on all years (including the test years) before saving")
    a = ap.parse_args()

    params = dict(pf.DEFAULT_HGB_PARAMS)
    if os.path.exists(a.params):
        with open(a.params) as f:
            params.update(json.load(f))
        print(f"using hyperparameters from {a.params}: {params}")
    else:
        print(f"{a.params} not found, using defaults: {params}")

    df = pf.load_data(a.data)
    panel = pf.build_panel(df)
    print(f"data: {len(df)} rows, {panel.groupby(['store_id', 'product']).ngroups} store-product series")

    models, metrics = {}, {}
    for h in pf.HORIZONS:
        d = pf.horizon_frame(panel, h)
        tr, va, te = pf.split_by_target_year(d)

        # 1. threshold from validation years
        probe = pf.make_model("boosting", pf.ALL_NUM, params).fit(tr[pf.FEATURES], tr["y"])
        threshold = pf.best_f1_threshold(va["y"], probe.predict_proba(va[pf.FEATURES])[:, 1])

        # 2. final fit on train + validation, 3. one look at the test years
        trva = pd.concat([tr, va])
        model = pf.make_model("boosting", pf.ALL_NUM, params).fit(trva[pf.FEATURES], trva["y"])
        p_te = model.predict_proba(te[pf.FEATURES])[:, 1]
        precision, recall = pf.precision_recall_at(te["y"], (p_te >= threshold).astype(int))
        m = dict(test_pr_auc=round(float(average_precision_score(te["y"], p_te)), 4),
                 test_roc_auc=round(float(roc_auc_score(te["y"], p_te)), 4),
                 test_precision=round(precision, 4), test_recall=round(recall, 4),
                 test_shortage_rate=round(float(te["y"].mean()), 4),
                 threshold=round(threshold, 4), n_train=len(trva), n_test=len(te))
        print(f"h={h}: {m}")

        if a.refit_all:
            model = pf.make_model("boosting", pf.ALL_NUM, params).fit(d[pf.FEATURES], d["y"])
        models[h] = dict(model=model, threshold=threshold)
        metrics[h] = m

    pairs = sorted({(s, p) for s, p in zip(df["store_id"], df["product"])})
    bundle = dict(
        models=models,
        metrics=metrics,
        features=pf.FEATURES,
        numeric_features=pf.ALL_NUM,
        horizons=list(pf.HORIZONS),
        stores=sorted(df["store_id"].unique()),
        products=sorted(df["product"].unique()),
        valid_pairs=pairs,
        params=params,
        refit_all=a.refit_all,
        sklearn_version=sklearn.__version__,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"saved model bundle to {a.out}")


if __name__ == "__main__":
    main()

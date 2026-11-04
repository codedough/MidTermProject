"""
Train the final shortage-warning models and save them to model/shortage_model.bin.

For each horizon (1, 2, 3 months ahead):
  1. fit on the training years, choose the alert threshold on the validation years (best F1)
  2. refit on training + validation years with that threshold fixed
  3. score the test years once and print precision, recall, F1, PR-AUC, ROC-AUC
Use --refit-all to also learn from the test years before saving (no test metrics are
then meaningful for the saved model; the metrics printed are from step 3).

The model type and hyperparameters come from model_config.json, written by notebook.ipynb.

    python train.py
"""
import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, roc_auc_score

import produce_features as pf

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "synthetic_produce_prices_1990_2025.csv"))
    ap.add_argument("--config", default=str(ROOT / "model_config.json"))
    ap.add_argument("--out", default=str(ROOT / "model" / "shortage_model.bin"))
    ap.add_argument("--refit-all", action="store_true", help="also train on the test years before saving")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    kind, params = config["kind"], config["params"]
    print(f"model: {kind} {params}")

    df = pf.load_data(args.data)
    panel = pf.build_panel(df)

    models, metrics = {}, {}
    for h in pf.HORIZONS:
        d = pf.horizon_frame(panel, h)
        tr, va, te = pf.split_by_target_year(d)

        m = pf.make_model(kind, params).fit(pf.to_records(tr), tr["y"])
        threshold = pf.best_f1_threshold(va["y"], m.predict_proba(pf.to_records(va))[:, 1])

        trva = pd.concat([tr, va])
        m = pf.make_model(kind, params).fit(pf.to_records(trva), trva["y"])
        p = m.predict_proba(pf.to_records(te))[:, 1]
        prec, rec = pf.precision_recall_at(te["y"], p >= threshold)
        metrics[h] = {
            "threshold": round(threshold, 4),
            "pr_auc": round(float(average_precision_score(te["y"], p)), 4),
            "roc_auc": round(float(roc_auc_score(te["y"], p)), 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "persistence_pr_auc": round(float(average_precision_score(te["y"], te["persist"])), 4),
            "test_rows": int(len(te)), "test_shortages": int(te["y"].sum()),
        }
        print(f"horizon {h}: " + "  ".join(f"{k}={v}" for k, v in metrics[h].items()))

        if args.refit_all:
            m = pf.make_model(kind, params).fit(pf.to_records(d), d["y"])
        models[h] = {"model": m, "threshold": threshold}

    pairs = sorted(df[["store_id", "product"]].drop_duplicates().itertuples(index=False, name=None))
    bundle = {
        "models": models, "metrics": metrics, "config": config,
        "numeric_features": pf.ALL_NUM, "horizons": list(pf.HORIZONS),
        "stores": sorted(df["store_id"].unique()), "products": sorted(df["product"].unique()),
        "valid_pairs": pairs, "refit_all": args.refit_all,
        "sklearn_version": sklearn.__version__, "xgboost_version": xgboost.__version__,
        "python_version": sys.version.split()[0],
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""
Shared code for the produce shortage early-warning project.

Used by the notebook, train.py and predict.py so that features are built the
same way in experiments, training and serving.

Question: at the end of month t, will this product at this store have a
shortage (availability < 82%) in month t+h, for h = 1, 2 or 3 months?
Every feature is something known at month t.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HORIZONS = (1, 2, 3)
TRAIN_END = 2015     # last predicted-month year used for training
VAL_END = 2020       # last predicted-month year used for validation; later years = test

WEATHER_BASE = ["drought_index", "frost_days", "heatwave_days", "temp_anomaly_c"]

CATEGORICAL = ["store_id", "product"]
BASE_NUM = (["avail"] + [f"avail_lag{k}" for k in (1, 2, 3, 12)]
            + ["real_logprice", "real_logprice_lag1", "real_logprice_lag12",
               "fuel_cost_index", "target_month_sin", "target_month_cos"])
WEATHER_NUM = ([c for w in WEATHER_BASE for c in (w, w + "_3m")]
               + ["enso_index", "enso_index_3m"])
ALL_NUM = BASE_NUM + WEATHER_NUM
FEATURES = ALL_NUM + CATEGORICAL

DEFAULT_HGB_PARAMS = dict(max_iter=250, learning_rate=0.05, max_depth=4, random_state=1)


# --------------------------------------------------------------------------
# Data and features
# --------------------------------------------------------------------------
def load_data(path):
    return pd.read_csv(path, parse_dates=["date"])


def real_log_price(price, cpi_index):
    """Log price with inflation removed (85% pass-through, as in the data generator)."""
    return np.log(price) - 0.85 * np.log(cpi_index / 100)


def build_panel(df, horizons=HORIZONS):
    """One row per (store, product, month t) on a complete monthly calendar.

    Gaps (missing price reports) stay as NaN so that every shift is by date,
    not by row position.
    """
    months = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")

    national = (df.drop_duplicates("date").set_index("date")
                [["enso_index", "cpi_index", "fuel_cost_index"]].reindex(months))
    regional = df.drop_duplicates(["region", "date"]).set_index(["region", "date"])[WEATHER_BASE]
    store_region = df.drop_duplicates("store_id").set_index("store_id")["region"]
    cpi = national["cpi_index"].reset_index(drop=True)

    frames = []
    for (store, prod), g in df.groupby(["store_id", "product"]):
        g = g.set_index("date").reindex(months)
        out = pd.DataFrame({"t": months})
        out["store_id"] = store
        out["product"] = prod

        avail = g["availability_pct"].reset_index(drop=True)
        flag = g["shortage_flag"].reset_index(drop=True)
        rlp = real_log_price(g["price_usd_per_lb"].reset_index(drop=True), cpi)

        out["avail"] = avail
        for k in (1, 2, 3, 12):
            out[f"avail_lag{k}"] = avail.shift(k)
        out["real_logprice"] = rlp
        for k in (1, 12):
            out[f"real_logprice_lag{k}"] = rlp.shift(k)
        out["shortage_now"] = flag

        w = regional.loc[store_region[store]].reindex(months).reset_index(drop=True)
        for c in WEATHER_BASE:
            out[c] = w[c]
            out[c + "_3m"] = w[c].rolling(3, min_periods=1).mean()
        enso = national["enso_index"].reset_index(drop=True)
        out["enso_index"] = enso
        out["enso_index_3m"] = enso.rolling(3, min_periods=1).mean()
        out["fuel_cost_index"] = national["fuel_cost_index"].reset_index(drop=True)

        for h in horizons:
            out[f"y_h{h}"] = flag.shift(-h)                  # shortage h months later
            out[f"seasonal_h{h}"] = flag.shift(12 - h)       # same month last year
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def target_month_features(month_t, h):
    """Sine/cosine of the calendar month being predicted (month t plus h)."""
    tm = (np.asarray(month_t) - 1 + h) % 12 + 1
    return np.sin(2 * np.pi * tm / 12), np.cos(2 * np.pi * tm / 12)


def horizon_frame(panel, h):
    """Rows for horizon h that have a known outcome, with target-month features."""
    d = panel.dropna(subset=[f"y_h{h}"]).copy()
    d["y"] = d[f"y_h{h}"].astype(int)
    d["target_date"] = d["t"] + pd.DateOffset(months=h)
    d["target_month_sin"], d["target_month_cos"] = target_month_features(d["t"].dt.month, h)
    d["persist"] = d["shortage_now"].fillna(0)          # baseline: same as now
    d["seasonal"] = d[f"seasonal_h{h}"].fillna(0)       # baseline: same month last year
    return d


def split_by_target_year(d, train_end=TRAIN_END, val_end=VAL_END):
    year = d["target_date"].dt.year
    return (d[year <= train_end], d[(year > train_end) & (year <= val_end)], d[year > val_end])


# --------------------------------------------------------------------------
# Models and metrics
# --------------------------------------------------------------------------
def make_model(kind, num_cols, params=None):
    """kind: 'logistic', 'forest' or 'boosting'. `params` are passed to the classifier."""
    params = dict(params or {})
    if kind == "logistic":
        num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        clf = LogisticRegression(**{"C": 1.0, "max_iter": 2000, **params})
    elif kind == "boosting":
        num = "passthrough"                                  # boosting handles NaN itself
        clf = HistGradientBoostingClassifier(**{**DEFAULT_HGB_PARAMS, **params})
    elif kind == "forest":
        num = SimpleImputer(strategy="median")
        clf = RandomForestClassifier(**{"n_estimators": 300, "min_samples_leaf": 5, "n_jobs": -1,
                                        "random_state": 1, **params})
    else:
        raise ValueError(kind)
    pre = ColumnTransformer([
        ("num", num, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def best_f1_threshold(y, scores):
    p, r, thr = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def precision_recall_at(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)

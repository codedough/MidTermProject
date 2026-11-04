# Fresh Produce Shortage Early Warning

Can we predict, one to three months ahead, whether a grocery store will run short of a fresh fruit or vegetable, using weather, past availability and prices? This project trains a model to answer that and serves it as a web service.

## What is the problem to solve

Fresh produce availability swings with the seasons, the weather, fuel costs and supply disruptions. Weather affects supply with a delay of a month or two, so it should give some advance notice of trouble. A store that knows a shortage is likely next month can reorder, switch suppliers or warn customers.

**Task.** At the end of month *t*, predict whether a given product at a given store will have a **shortage** in month *t+h*, for *h* = 1, 2 or 3 months. A shortage means availability below 82% of the month (`shortage_flag = 1`). About 9% of rows are shortages, so the classes are imbalanced. The main metric is **PR-AUC** (a model with no skill scores about 0.08; 1.0 is perfect).

**Questions the project answers**

1. How much warning can a model give, and how fast does it fall off with the horizon?
2. Do weather features help beyond what last month's availability already tells you?
3. Which features matter most?

**The data is simulated.** Seasonality, weather, inflation, fuel cost, store effects, supply shocks are not collected from stores. The project validates a method; it does not describe the real produce market.

## 2. What is in this repository

```
README.md                  this file

data/
  synthetic_produce_prices_1990_2025.csv   the dataset (committed; 10,000 rows)

notebook.ipynb             cleaning, EDA, feature importance, model selection, tuning

produce_features.py        feature building shared by notebook, train.py and predict.py

train.py                   trains the final models and saves them

model_params.json          tuned hyperparameters (written by the notebook, read by train.py)

model/shortage_model.bin   the trained models (committed, so the service runs without training)

predict.py                 Flask web service

predict_test.py            sample client for the service

sample_request.json        an example request

Dockerfile, .dockerignore  container for the web service

```

## 3. Data

The dataset is committed at `data/synthetic_produce_prices_1990_2025.csv`, so nothing needs downloading. To recreate it exactly (same seed gives the same file):

```bash
python generate_produce_data.py --seed 42 --rows 10000
```

One row is one product at one store in one month (Jan 1990 to Dec 2025): 12 products, 4 stores (one per US region), 24 store-product series. About 3.5% of months have no report. The column reference is at the end of this file.

## 3.1 Requirements

Python 3.10 or newer (tested with 3.13).

**Service and training** (`train.py`, `predict.py`). scikit-learn is pinned exactly because the saved model is a pickle, which is only guaranteed to load with the version that wrote it.

```text
numpy==2.5.3
pandas==3.0.5
scikit-learn==1.9.1
flask==3.1.3
gunicorn==26.2.0
```

**Notebook** (`notebook.ipynb`). Everything above, plus plotting and Jupyter:

```text
matplotlib==3.11.2
ipykernel==7.4.0
nbconvert==7.17.2
jupyterlab
```

## 4. How to run

You need Python 3.10 or newer. Everything was tested with Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install numpy==2.5.3 pandas==3.0.5 scikit-learn==1.9.1 flask==3.1.3 gunicorn==26.2.0
```

(With `uv`: `uv venv && uv pip install numpy==2.5.3 pandas==3.0.5 scikit-learn==1.9.1 flask==3.1.3 gunicorn==26.2.0`.)

### 4.1 Explore and tune: the notebook

```bash
pip install matplotlib==3.11.2 ipykernel==7.4.0 nbconvert==7.17.2 jupyterlab
jupyter lab notebook.ipynb
```

The notebook is already executed, so you can read it as is. Running it from scratch takes about 4 minutes, mostly the tuning section. It writes `model_params.json`.

### 4.2 Train the final model

```bash
python train.py
```

This trains one gradient-boosting model per horizon (1, 2 and 3 months), picks a decision threshold on validation years, prints test-year metrics, and saves everything to `model/shortage_model.bin`. Use `--refit-all` to also learn from the test years before saving.

### 4.3 Run the web service

Locally, with Flask's development server:

```bash
python predict.py
```

Or with gunicorn, the way the container runs it:

```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```

Then, in another terminal, send the sample requests:

```bash
python predict_test.py
```

### 4.4 Run it in Docker

```bash
docker build -t produce-shortage-warning .
docker run --rm -p 9696:9696 produce-shortage-warning
python predict_test.py
```

The image contains only `predict.py`, `produce_features.py` and the saved model. It runs gunicorn on port 9696 as a non-root user and has a health check.

> **Note:** Docker was not available in the environment where this project was built, so the image itself has not been built. I tested the same steps by copying exactly the files the Dockerfile copies into an empty folder, installing the pinned packages from the Requirements section in a fresh Python 3.13 environment, and running the Dockerfile's gunicorn command. If `docker build` fails on your machine, please check that the `python:3.13-slim` base image can be pulled.

If you retrain with a different scikit-learn version, update the scikit-learn pin in the Requirements section and the Dockerfile to match, because the model is a pickle file and is only guaranteed to load with the version that saved it.

## 5. Using the service

### `GET /health`

Returns the status, the horizons served, when the model was trained and the scikit-learn version.

### `POST /predict`

Send a JSON object describing what you know at the end of month *t*. The service returns the probability of a shortage `horizon` months later.

**Required fields**

| Field | Meaning |
|---|---|
| `store_id` | `S01` Northeast, `S02` Midwest, `S03` South, `S04` West |
| `product` | tomatoes, lettuce, strawberries, apples, bananas, oranges, potatoes, carrots, broccoli, avocados, grapes, onions. Each store carries only 6 (see the table in the dataset reference); other combinations are rejected. |
| `horizon` | 1, 2 or 3 months ahead |
| `month` | The current month *t*, 1-12. The service predicts month `month + horizon`. |
| `avail` | Availability this month, 0-100 (%) |

**Optional fields.** Anything you leave out is treated as unknown. Predictions are better when you supply them, because the model was trained with all of them.

| Field | Meaning |
|---|---|
| `avail_lag1`, `avail_lag2`, `avail_lag3`, `avail_lag12` | Availability 1, 2, 3 and 12 months ago |
| `real_logprice`, `real_logprice_lag1`, `real_logprice_lag12` | Inflation-adjusted log price now, 1 and 12 months ago. Compute as `ln(price) - 0.85 * ln(cpi_index / 100)`, with `cpi_index` as in the dataset (Jan 1990 = 100). |
| `fuel_cost_index` | Fuel cost index (1990 = 100) |
| `drought_index`, `drought_index_3m` | Regional drought index now and its 3-month average (negative = dry) |
| `frost_days`, `frost_days_3m` | Frost days this month and the 3-month average |
| `heatwave_days`, `heatwave_days_3m` | Heat-stress days this month and the 3-month average |
| `temp_anomaly_c`, `temp_anomaly_c_3m` | Temperature anomaly now and its 3-month average |
| `enso_index`, `enso_index_3m` | ENSO-like climate index now and its 3-month average |

**Example** (a real case from the test years: tomatoes at the South store in February 2022, during a dry spell; a shortage did follow in March):

```bash
curl -s -X POST http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d @sample_request.json
```

```json
{
  "horizon": 1,
  "missing_features": [],
  "month_predicted": 3,
  "product": "tomatoes",
  "shortage_probability": 0.832,
  "shortage_warning": true,
  "store_id": "S03",
  "threshold": 0.2815
}
```

`shortage_warning` is true when the probability is at or above the horizon's `threshold`, chosen to balance precision and recall on validation data. Invalid input (unknown field, unknown store/product, out-of-range value, bad JSON) returns HTTP 400 with an `error` message.

A minimal request needs only the five required fields:

```bash
curl -s -X POST http://localhost:9696/predict -H 'Content-Type: application/json' \
  -d '{"store_id":"S04","product":"strawberries","horizon":2,"month":9,"avail":84.5}'
```

## 6. Method and results

Everything is in `notebook.ipynb`; `train.py` repeats the final step.

- **Setup:** features are built for each store-product on a complete monthly calendar, so lags are by date even where a month is missing. Every feature is known at month *t*. Data is split by the month being predicted: train 1991-2015, validation 2016-2020, test 2021-2025.
- **Feature importance:** at 1 month ahead, current availability dominates. At 3 months ahead it barely matters; store and product, inflation-adjusted price, calendar month and weather take over, and weather's share grows with the horizon.
- **Model selection:** logistic regression, random forest and gradient boosting, each with and without weather features, against three baselines (never shortage, same as now, same month last year). The top three models were close on validation. Weather helped logistic regression clearly but changed the tree models only slightly.
- **Tuning:** random search with time-ordered cross-validation. Tuned boosting (PR-AUC 0.52) beat the random forest (0.49) and logistic regression (0.48).
- **Final test results (2021-2025)**

| | 1 month | 2 months | 3 months |
|---|---|---|---|
| Final model, PR-AUC | **0.59** | **0.46** | **0.41** |
| Final model, ROC-AUC | 0.93 | 0.89 | 0.87 |
| Same as now (persistence), PR-AUC | 0.31 | 0.22 | 0.12 |
| No skill, PR-AUC | 0.08 | 0.08 | 0.08 |

At 1 month ahead the model catches about half of the shortages (recall 0.55) and about 57% of its warnings are correct (precision 0.57).

**Limitations.** The data is simulated. The test years contain only about 115 shortages and include an unusual stretch (the 2020-2021 spike, then calm years), so results could differ on another period; the model under-warns when conditions are more stressed than in most of its training data. Results vary a lot by product, and for products with very few shortages (potatoes, bananas, carrots, onions) the model has little to learn from. Weather features add only modest accuracy in this data. Regional weather stands in for supply conditions at the store, not for where the produce is grown.

---

# Dataset reference

- **Rows:** 10,000, **columns:** 25, no missing cells
- **Grain:** one row = one product at one store in one month
- **Gaps:** the full panel is 10,368 rows; 368 (3.5%) are dropped at random to mimic missing price reports
- **Series:** each store carries 6 of the 12 products and each product is sold by exactly 2 stores

| Store | Region | Tier | Carries |
|---|---|---|---|
| S01 FreshMart Northeast | Northeast | premium | tomatoes, lettuce, strawberries, apples, bananas, avocados |
| S02 ValueGrocer Midwest | Midwest | discount | potatoes, carrots, onions, apples, bananas, oranges |
| S03 SunCoast Market | South | mainstream | oranges, tomatoes, lettuce, broccoli, grapes, onions |
| S04 PacificFresh | West | premium | strawberries, broccoli, avocados, grapes, potatoes, carrots |

| Column | Meaning |
|---|---|
| `record_id`, `date`, `year`, `month` | Row id and month start date |
| `store_id`, `store_name`, `region`, `store_tier` | Store, its US region and price tier |
| `product`, `unit` | One of the 12 products; unit is lb |
| `price_usd_per_lb` | Retail price in nominal USD per pound |
| `availability_pct` | Share of the month the item was in stock, 0-100 |
| `shortage_flag` | 1 if `availability_pct` < 82 (the target in this project) |
| `avg_temp_c`, `temp_anomaly_c` | Regional monthly mean temperature and its departure from the 1990 baseline (includes a warming trend) |
| `precip_mm` | Regional monthly precipitation |
| `drought_index` | Palmer-style index. **Negative = dry, positive = wet** |
| `frost_days`, `heatwave_days` | Count of frost days and heat-stress days in the month |
| `enso_index` | National ENSO-like climate index |
| `cpi_index` | Consumer price index, Jan 1990 = 100 (approximate US inflation) |
| `fuel_cost_index` | Stylised fuel/freight cost index, 1990 = 100 (simulated) |
| `import_disruption_index` | Shipping/import disruption (standardised; higher = worse) |
| `promotion_flag` | 1 if the item was on promotion that month |
| `covid_period` | 1 for Mar 2020 to Dec 2021 |

**Built-in cause and effect.** Prices follow CPI (85% pass-through) plus a small per-product drift, and are highest off-season. Heat, drought, excess rain and frost raise prices and cut availability, with a lag (50% same month, 30% one month earlier, 20% two months earlier), and sensitivity differs by product (avocados and tomatoes react to drought, lettuce and broccoli to heat, bananas and onions barely react). Premium stores cost more, discount stores less. Fuel cost, import disruptions (strongest for bananas and avocados), promotions (about 10% lower price) and the covid period also matter.
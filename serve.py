"""
Web service for the shortage early-warning model (FastAPI).

    python serve.py                      # listens on http://0.0.0.0:9696
    uvicorn serve:app --port 9696        # equivalent

    GET  /health    -> {"status": "ok", ...}
    POST /predict   -> JSON in, prediction out (see README)
Interactive docs: http://localhost:9696/docs
"""
import os
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import predict as pred

app = FastAPI(title="Fresh produce shortage early warning", version="1.0.0")
bundle = pred.load_bundle()          # loaded once at start-up


def _pct():
    return Field(default=None, ge=0, le=100)


def _days():
    return Field(default=None, ge=0, le=31)


class Request(BaseModel):
    """What is known at the end of month `month`. Only the first five fields are required."""
    model_config = ConfigDict(extra="forbid")      # unknown fields -> 422

    store_id: Literal["S01", "S02", "S03", "S04"]
    product: Literal["tomatoes", "lettuce", "strawberries", "apples", "bananas", "oranges",
                     "potatoes", "carrots", "broccoli", "avocados", "grapes", "onions"]
    horizon: Literal[1, 2, 3] = Field(description="months ahead to predict")
    month: int = Field(ge=1, le=12, description="current month 1-12")
    avail: float = Field(ge=0, le=100, description="availability this month, %")
    avail_lag1: Optional[float] = _pct()
    avail_lag2: Optional[float] = _pct()
    avail_lag3: Optional[float] = _pct()
    avail_lag12: Optional[float] = _pct()
    real_logprice: Optional[float] = None
    real_logprice_lag1: Optional[float] = None
    real_logprice_lag12: Optional[float] = None
    fuel_cost_index: Optional[float] = None
    drought_index: Optional[float] = None
    drought_index_3m: Optional[float] = None
    frost_days: Optional[float] = _days()
    frost_days_3m: Optional[float] = _days()
    heatwave_days: Optional[float] = _days()
    heatwave_days_3m: Optional[float] = _days()
    temp_anomaly_c: Optional[float] = None
    temp_anomaly_c_3m: Optional[float] = None
    enso_index: Optional[float] = None
    enso_index_3m: Optional[float] = None


class Response(BaseModel):
    store_id: str
    product: str
    horizon: int
    month_predicted: int
    shortage_probability: float
    shortage_warning: bool
    threshold: float
    missing_features: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "horizons": bundle["horizons"], "model": bundle["config"]["kind"],
            "trained_at": bundle["trained_at"]}


@app.post("/predict", response_model=Response)
def predict(req: Request):
    try:
        return pred.predict(bundle, req.model_dump(exclude_none=True))
    except pred.InputError as e:                   # e.g. a store that does not carry the product
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 9696)))

#!/usr/bin/env python3
"""
Synthetic fresh-produce price & availability dataset (monthly, 1990-2025).

THIS DATA IS SIMULATED. It is built from explicit, documented cause-and-effect
rules (seasonality, weather, inflation, fuel cost, store effects, supply
shocks) so that a machine-learning model has real structure to recover.
It is not real market data. Say so in your project write-up.

Usage:
    python generate_produce_data.py --seed 42 --rows 10000 --out data/synthetic_produce_prices_1990_2025.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. Inputs
# --------------------------------------------------------------------------

START, END = "1990-01-01", "2025-12-01"
BASE_YEAR = 1990          # CPI and fuel indexes equal 100 here; base prices are 1990 prices

# Multiplier on all weather effects (price and availability). Raise it to make weather matter more.
WEATHER_STRENGTH = 1.6

# A month counts as a shortage when availability falls below this many percent.
SHORTAGE_THRESHOLD = 82

# Approximate US annual CPI inflation by year (rounded; used for the macro
# trend only). 2025 is an estimate.
ANNUAL_INFLATION = {
    1990: .054, 1991: .042, 1992: .030, 1993: .030, 1994: .026, 1995: .028,
    1996: .030, 1997: .023, 1998: .016, 1999: .022, 2000: .034, 2001: .028,
    2002: .016, 2003: .023, 2004: .027, 2005: .034, 2006: .032, 2007: .028,
    2008: .038, 2009: -.004, 2010: .016, 2011: .032, 2012: .021, 2013: .015,
    2014: .016, 2015: .001, 2016: .013, 2017: .021, 2018: .024, 2019: .018,
    2020: .012, 2021: .047, 2022: .080, 2023: .041, 2024: .029, 2025: .027,
}

# Stylised (simulated) fuel-cost index, 1990 = 100. Anchors are
# (fractional year, index); values in between are interpolated.
FUEL_ANCHORS = [
    (1990.0, 100), (1990.8, 142), (1991.5, 92), (1998.9, 58), (2000.5, 117),
    (2003.0, 117), (2005.0, 167), (2008.5, 267), (2009.2, 117), (2011.0, 200),
    (2014.0, 208), (2015.5, 100), (2016.0, 75), (2018.5, 142), (2020.3, 67),
    (2021.0, 117), (2022.4, 225), (2023.5, 167), (2025.0, 142), (2025.99, 133),
]

# Each store carries 6 of the 12 products, and each product is carried by
# exactly 2 stores (24 store-product series in total).
STORES = [
    dict(store_id="S01", store_name="FreshMart Northeast", region="Northeast",
         tier="premium", price_mult=1.12, avail_adj=3.0,
         carries=["tomatoes", "lettuce", "strawberries", "apples", "bananas", "avocados"]),
    dict(store_id="S02", store_name="ValueGrocer Midwest", region="Midwest",
         tier="discount", price_mult=0.92, avail_adj=-3.0,
         carries=["potatoes", "carrots", "onions", "apples", "bananas", "oranges"]),
    dict(store_id="S03", store_name="SunCoast Market", region="South",
         tier="mainstream", price_mult=1.00, avail_adj=0.0,
         carries=["oranges", "tomatoes", "lettuce", "broccoli", "grapes", "onions"]),
    dict(store_id="S04", store_name="PacificFresh", region="West",
         tier="premium", price_mult=1.08, avail_adj=2.0,
         carries=["strawberries", "broccoli", "avocados", "grapes", "potatoes", "carrots"]),
]

# temp = mean annual temp (C), amp = seasonal swing (C), precip = mean monthly
# precipitation (mm), pseas/pmonth = precipitation seasonality,
# enso_* = sensitivity to the national ENSO-like climate index.
# Regional weather is used as a proxy for supply conditions for that store.
CLIMATE = {
    "Northeast": dict(temp=10, amp=12, precip=95, pseas=0.10, pmonth=7, enso_temp=0.25, enso_precip=0.3),
    "Midwest":   dict(temp=10, amp=14, precip=80, pseas=0.35, pmonth=6, enso_temp=0.35, enso_precip=-0.3),
    "South":     dict(temp=19, amp=9,  precip=110, pseas=0.25, pmonth=8, enso_temp=-0.2, enso_precip=0.8),
    "West":      dict(temp=15, amp=7,  precip=40, pseas=0.60, pmonth=1, enso_temp=0.3, enso_precip=1.0),
}

# base = 1990 price (USD/lb); trend = annual log drift (productivity);
# curve = (peak month of domestic season, seasonal cycles per year);
# p_amp = seasonal price swing (log); avail_depth = off-season availability
# drop (points); import_share = exposure to the import/shipping shock;
# *_sens = weather sensitivities (log-price units); fuel_share = freight cost
# sensitivity; region_adj = price discount/premium where the item is grown locally.
PRODUCTS = {
    "tomatoes": dict(base=1.00, trend=-0.004, curve=(8, 1), p_amp=0.18, avail_depth=12,
                     import_share=0.35, temp_sens=0.020, drought_sens=0.025, wet_sens=0.004,
                     frost_sens=0.012, frost_months=(4, 5, 9, 10), heat_sens=0.006,
                     fuel_share=0.05, region_adj={"South": -0.05}),
    "lettuce": dict(base=0.70, trend=-0.002, curve=(4.5, 2), p_amp=0.10, avail_depth=8,
                    import_share=0.10, temp_sens=0.030, drought_sens=0.020, wet_sens=0.006,
                    frost_sens=0.006, frost_months=(10, 11), heat_sens=0.012,
                    fuel_share=0.06, region_adj={"West": -0.07}),
    "strawberries": dict(base=1.60, trend=-0.003, curve=(5, 1), p_amp=0.25, avail_depth=14,
                         import_share=0.40, temp_sens=0.025, drought_sens=0.020, wet_sens=0.012,
                         frost_sens=0.010, frost_months=(3, 4, 5), heat_sens=0.008,
                         fuel_share=0.07, region_adj={"West": -0.08}),
    "apples": dict(base=0.70, trend=-0.001, curve=(10, 1), p_amp=0.10, avail_depth=8,
                   import_share=0.10, temp_sens=0.010, drought_sens=0.012, wet_sens=0.004,
                   frost_sens=0.015, frost_months=(3, 4, 5), heat_sens=0.004,
                   fuel_share=0.04, region_adj={"Northeast": -0.04}),
    "bananas": dict(base=0.43, trend=-0.006, curve=(1, 0), p_amp=0.02, avail_depth=1,
                    import_share=1.00, temp_sens=0.002, drought_sens=0.000, wet_sens=0.000,
                    frost_sens=0.000, frost_months=(), heat_sens=0.000,
                    fuel_share=0.08, region_adj={"South": -0.03}),
    "oranges": dict(base=0.45, trend=-0.002, curve=(2, 1), p_amp=0.10, avail_depth=6,
                    import_share=0.20, temp_sens=0.008, drought_sens=0.015, wet_sens=0.003,
                    frost_sens=0.008, frost_months=(12, 1, 2), heat_sens=0.004,
                    fuel_share=0.05, region_adj={"South": -0.05}),
    "potatoes": dict(base=0.35, trend=-0.002, curve=(10, 1), p_amp=0.06, avail_depth=3,
                     import_share=0.05, temp_sens=0.006, drought_sens=0.020, wet_sens=0.008,
                     frost_sens=0.004, frost_months=(4, 5, 9, 10), heat_sens=0.004,
                     fuel_share=0.05, region_adj={"Midwest": -0.03}),
    "carrots": dict(base=0.50, trend=-0.002, curve=(11, 1), p_amp=0.07, avail_depth=4,
                    import_share=0.05, temp_sens=0.008, drought_sens=0.015, wet_sens=0.004,
                    frost_sens=0.005, frost_months=(10, 11), heat_sens=0.004,
                    fuel_share=0.05, region_adj={"West": -0.04}),
    "broccoli": dict(base=0.95, trend=-0.003, curve=(4, 2), p_amp=0.12, avail_depth=7,
                     import_share=0.15, temp_sens=0.025, drought_sens=0.018, wet_sens=0.006,
                     frost_sens=0.006, frost_months=(3, 10, 11), heat_sens=0.014,
                     fuel_share=0.06, region_adj={"West": -0.06}),
    "avocados": dict(base=1.20, trend=-0.001, curve=(6, 1), p_amp=0.20, avail_depth=9,
                     import_share=0.55, temp_sens=0.015, drought_sens=0.030, wet_sens=0.002,
                     frost_sens=0.010, frost_months=(12, 1, 2), heat_sens=0.010,
                     fuel_share=0.06, region_adj={"West": -0.10}),
    "grapes": dict(base=1.50, trend=-0.003, curve=(9, 1), p_amp=0.18, avail_depth=10,
                   import_share=0.45, temp_sens=0.012, drought_sens=0.015, wet_sens=0.010,
                   frost_sens=0.008, frost_months=(4, 5), heat_sens=0.010,
                   fuel_share=0.06, region_adj={"West": -0.08}),
    "onions": dict(base=0.40, trend=-0.002, curve=(9, 1), p_amp=0.05, avail_depth=2,
                   import_share=0.10, temp_sens=0.006, drought_sens=0.012, wet_sens=0.006,
                   frost_sens=0.002, frost_months=(), heat_sens=0.004,
                   fuel_share=0.04, region_adj={"Midwest": -0.02}),
}


# --------------------------------------------------------------------------
# 2. Helpers
# --------------------------------------------------------------------------
def ar1(rng, n, phi, sd):
    """AR(1) series with innovation std `sd`."""
    e = rng.normal(0, sd, n)
    x = np.zeros(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def season_curve(month, peak, cycles):
    """1 at the domestic-season peak, 0 at the opposite point; flat if cycles=0."""
    if cycles == 0:
        return np.ones_like(month, dtype=float)
    return 0.5 + 0.5 * np.cos(2 * np.pi * cycles * (month - peak) / 12)


def lag(x, k):
    """Shift an array k steps back, padding the start with its first value."""
    return np.concatenate([np.full(k, x[0]), x[:-k]])


# --------------------------------------------------------------------------
# 3. Generator
# --------------------------------------------------------------------------
def build_panel(seed):
    rng = np.random.default_rng(seed)
    months = pd.date_range(START, END, freq="MS")
    T = len(months)
    year = months.year.values
    month = months.month.values
    frac = year + (month - 1) / 12

    # ---- national / macro series ----
    infl = np.array([ANNUAL_INFLATION[y] for y in year])
    cpi = 100 * np.cumprod(np.concatenate([[1.0], (1 + infl[1:]) ** (1 / 12)]))

    fx, fy = zip(*FUEL_ANCHORS)
    fuel = np.interp(frac, fx, fy) * np.exp(ar1(rng, T, 0.9, 0.04))

    enso = np.clip(ar1(rng, T, 0.93, 0.30), -3, 3)           # ENSO-like climate index
    import_shock = ar1(rng, T, 0.80, 1.0)                    # shipping / import disruption
    covid = ((months >= "2020-03-01") & (months <= "2021-12-01")).astype(int)

    # ---- regional weather ----
    weather = {}
    for region, c in CLIMATE.items():
        clim_temp = c["temp"] + c["amp"] * np.sin(2 * np.pi * (month - 4) / 12)
        temp_anom = 0.03 * (year - 1990) + c["enso_temp"] * enso + ar1(rng, T, 0.5, 1.2)
        avg_temp = clim_temp + temp_anom

        precip_clim = c["precip"] * (1 + c["pseas"] * np.cos(2 * np.pi * (month - c["pmonth"]) / 12))
        log_ratio = ar1(rng, T, 0.3, 0.35) + c["enso_precip"] * 0.12 * enso
        precip = precip_clim * np.exp(log_ratio)
        precip_z = log_ratio / 0.4

        moisture = precip_z - 0.12 * temp_anom
        pdsi = np.zeros(T)
        for t in range(1, T):
            pdsi[t] = 0.88 * pdsi[t - 1] + 0.55 * moisture[t]
        pdsi = np.clip(pdsi, -6, 6)

        frost_days = rng.poisson(np.clip((12 - avg_temp) * 0.7, 0, 20))
        heat_days = rng.poisson(np.clip((avg_temp - 20) * 0.9, 0, 20))

        weather[region] = dict(avg_temp=avg_temp, temp_anom=temp_anom, precip=precip,
                               precip_z=precip_z, pdsi=pdsi, frost_days=frost_days,
                               heat_days=heat_days)

    # ---- national product shocks ----
    product_shock = {p: ar1(rng, T, 0.7, 0.03) for p in PRODUCTS}

    # ---- series-level loop ----
    frames = []
    for s in STORES:
        w = weather[s["region"]]
        for pname in s["carries"]:
            p = PRODUCTS[pname]
            curve = season_curve(month, *p["curve"])

            # weather stress (log-price units), then lagged mix: supply responds with delay
            stress = (p["temp_sens"] * np.maximum(w["temp_anom"], 0)
                      + p["drought_sens"] * np.maximum(-w["pdsi"], 0)
                      + p["wet_sens"] * np.maximum(w["precip_z"], 0)
                      + p["frost_sens"] * np.minimum(w["frost_days"], 10) * np.isin(month, p["frost_months"])
                      + p["heat_sens"] * w["heat_days"])
            stress_mix = WEATHER_STRENGTH * (0.5 * stress + 0.3 * lag(stress, 1) + 0.2 * lag(stress, 2))

            promo = rng.binomial(1, 0.12, T)

            log_price = (
                np.log(p["base"])
                + 0.85 * np.log(cpi / 100)
                + p["trend"] * (year - BASE_YEAR)
                + p["p_amp"] * 2 * (0.5 - curve)
                + stress_mix
                + p["fuel_share"] * np.log(fuel / 100)
                + p["import_share"] * 0.05 * import_shock
                + np.log(s["price_mult"])
                + p["region_adj"].get(s["region"], 0.0)
                + product_shock[pname]
                + ar1(rng, T, 0.6, 0.035)
                - 0.10 * promo
                + 0.03 * covid
            )

            depth = p["avail_depth"] * (1 - 0.012 * (year - BASE_YEAR))   # logistics improve over time
            avail = (
                95.0 + s["avail_adj"]
                - depth * (1 - curve)
                - 55 * stress_mix
                - p["import_share"] * 3.0 * np.maximum(import_shock, 0)
                + 2 * promo
                - 4 * covid
                + ar1(rng, T, 0.5, 1.8)
            )
            avail = np.clip(avail, 30, 100)

            frames.append(pd.DataFrame({
                "date": months,
                "year": year,
                "month": month,
                "store_id": s["store_id"],
                "store_name": s["store_name"],
                "region": s["region"],
                "store_tier": s["tier"],
                "product": pname,
                "unit": "lb",
                "avg_temp_c": w["avg_temp"].round(1),
                "temp_anomaly_c": w["temp_anom"].round(2),
                "precip_mm": w["precip"].round(1),
                "drought_index": w["pdsi"].round(2),
                "frost_days": w["frost_days"],
                "heatwave_days": w["heat_days"],
                "enso_index": enso.round(2),
                "cpi_index": cpi.round(2),
                "fuel_cost_index": fuel.round(1),
                "import_disruption_index": import_shock.round(2),
                "promotion_flag": promo,
                "covid_period": covid,
                "price_usd_per_lb": np.maximum(np.exp(log_price), 0.05).round(2),
                "availability_pct": avail.round(1),
            }))
    return pd.concat(frames, ignore_index=True), rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--out", default="data/synthetic_produce_prices_1990_2025.csv")
    a = ap.parse_args()

    panel, rng = build_panel(a.seed)
    if a.rows > len(panel):
        raise SystemExit(f"Full panel only has {len(panel)} rows; lower --rows or add stores/products.")

    # Real price surveys have gaps. Drop rows at random to land on exactly `rows`.
    keep = np.sort(rng.choice(len(panel), size=a.rows, replace=False))
    df = panel.iloc[keep].copy()

    df["shortage_flag"] = (df["availability_pct"] < SHORTAGE_THRESHOLD).astype(int)
    df = df.sort_values(["date", "store_id", "product"]).reset_index(drop=True)
    df.insert(0, "record_id", np.arange(1, len(df) + 1))
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows x {df.shape[1]} columns -> {a.out}")
    print(f"full panel before dropping missing reports: {len(panel)} rows "
          f"({df.groupby(['store_id', 'product']).ngroups} series)")


if __name__ == "__main__":
    main()

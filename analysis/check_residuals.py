"""
check_residuals.py
-------------------
Re‑compute residual kurtosis and Ljung‑Box p‑values for all production
metrics using the identical model specifications as master_forecast_engine.py.

This script was used during the documentation phase to verify that the
kurtosis and Ljung‑Box values reported in the technical write‑up matched
the current engine output.  It fits every series exactly as the engine
does (same data, same transformations, same exogenous variables), then
prints the Pearson kurtosis and Ljung‑Box p‑value at lag 4 for each.

Why this exists:
    The engine log does not print kurtosis directly, so this script was
    needed to confirm that the kurtosis values in the per‑series results
    table were accurate.  It also serves as a quick diagnostic if the
    pipeline is ever updated and the numbers need re‑checking.

Usage:
    python check_residuals.py

Input:
    data/processed/combined_quarterly.csv

Output:
    Console table: Series | Pearson Kurtosis | Ljung‑Box p(4)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import kurtosis

# ----- Load combined quarterly data -----
df = pd.read_csv("data/processed/combined_quarterly.csv", parse_dates=["quarter"])

# ----- Metric definitions (exact match with master_forecast_engine.py) -----
# Each entry maps the display name used throughout the codebase to the
# raw metric name as it appears in combined_quarterly.csv.
METRIC_RAW = {
    "RTT waiting list (level)": "Incomplete RTT pathways - Total waiting (mil) with estimates for missing data",
    "A&E attendances (flow)": "total_attendances",
    "Workforce FTE (level)": "FTE: All staff groups - All staff groups",
    "Nurse FTE (level)": "FTE: Professionally qualified clinical staff - Nurses & health visitors",
    "Doctor FTE (level)": "FTE: Professionally qualified clinical staff - HCHS doctors - All grades",
    "Bed occupancy (level)": "total_occupied_beds_overnight",
    "RTT % within 18 weeks (performance)": "Incomplete RTT pathways - % within 18 weeks",
    "A&E 12-hour decisions to admit (breach flow)": "number_of_patients_spending_12_hours_from_decision_to_admit_to_admission",
}

# ----- Model orders – must match PROD_ORDER in the engine (or MODEL_CONFIG) -----
ORDER = {
    "RTT waiting list (level)": (0, 1, 1),
    "A&E attendances (flow)": (1, 1, 0),
    "Workforce FTE (level)": (2, 1, 0),
    "Nurse FTE (level)": (2, 1, 0),
    "Doctor FTE (level)": (2, 1, 0),
    "Bed occupancy (level)": (1, 1, 1),          # finalised spec after backtesting
    "RTT % within 18 weeks (performance)": (1, 0, 1),
    "A&E 12-hour decisions to admit (breach flow)": (2, 1, 0),
}

# ----- Seasonal orders -----
SEASONAL = {
    "RTT waiting list (level)": (1, 0, 1, 4),
    "A&E attendances (flow)": (0, 0, 0, 4),
    "Workforce FTE (level)": (1, 0, 1, 4),
    "Nurse FTE (level)": (1, 0, 1, 4),
    "Doctor FTE (level)": (2, 0, 1, 4),
    "Bed occupancy (level)": (1, 0, 1, 4),
    "RTT % within 18 weeks (performance)": (0, 0, 0, 4),
    "A&E 12-hour decisions to admit (breach flow)": (0, 0, 0, 4),
}

# ----- Transformations applied before fitting -----
TRANSFORM = {
    "RTT waiting list (level)": "log",
    "Nurse FTE (level)": "log",
    "Doctor FTE (level)": "log",
    "RTT % within 18 weeks (performance)": "logit",
}

print("Series                         | Kurtosis | Ljung‑Box p(4)")
print("-" * 60)

for display_name, raw_name in METRIC_RAW.items():
    # ---- Extract and clean the series ----
    sub = df[df["metric"] == raw_name].dropna(subset=["value"]).sort_values("quarter")

    # A&E 12h breach uses a restricted estimation window (2021‑01‑01 onward)
    if display_name == "A&E 12-hour decisions to admit (breach flow)":
        sub = sub[sub["quarter"] >= "2021-01-01"]

    endog = sub.set_index("quarter")["value"]
    # Use the same exogenous columns as the engine (t and post_covid_trend_break)
    exog = sub.set_index("quarter")[["t", "post_covid_trend_break"]]

    # ---- Apply transformation ----
    trans = TRANSFORM.get(display_name)
    if trans == "logit":
        # Logit transform for bounded (0,1) proportions
        eps = 1e-6
        y = endog.clip(eps, 1 - eps)
        endog_t = np.log(y / (1 - y))
    elif trans == "log":
        endog_t = np.log(endog.replace(0, np.nan))
    else:
        endog_t = endog

    # ---- Fit the model exactly as the engine does ----
    model = sm.tsa.SARIMAX(
        endog_t,
        exog=exog,
        order=ORDER[display_name],
        seasonal_order=SEASONAL[display_name],
        trend='n',                                # no constant – trend is handled by exog
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=200)

    # ---- Extract residuals and compute diagnostics ----
    resid = res.resid.dropna()

    # Pearson kurtosis (normal distribution = 3)
    pearson_kurt = kurtosis(resid, fisher=False)

    # Ljung‑Box test for autocorrelation at lag 4
    lb_p = sm.stats.acorr_ljungbox(resid, lags=[4]).iloc[0, 1]

    print(f"{display_name:<30} | {pearson_kurt:8.2f} | {lb_p:.4f}")
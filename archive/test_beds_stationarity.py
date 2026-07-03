"""
test_beds_stationarity.py
----------------------------
Stationarity test for Bed occupancy, same ADF/KPSS approach as
test_stationarity.py but isolated to this one series.

Run:
    python src/test_beds_stationarity.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")
import sys

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exog_config import METRIC_NAMES

LABEL = "Bed occupancy (level)"


def run_adf(series):
    result = adfuller(series.dropna(), autolag="AIC")
    return {"stat": result[0], "pvalue": result[1], "lags": result[2]}


def run_kpss(series):
    result = kpss(series.dropna(), regression="c", nlags="auto")
    return {"stat": result[0], "pvalue": result[1], "lags": result[2]}


def verdict(adf_p, kpss_p):
    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p > 0.05
    if adf_stationary and kpss_stationary:
        return "STATIONARY (both agree)"
    if not adf_stationary and not kpss_stationary:
        return "NON-STATIONARY (both agree)"
    return "AMBIGUOUS (tests disagree)"


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])
    metric = METRIC_NAMES[LABEL]
    sub = df[df["metric"] == metric].sort_values("quarter").reset_index(drop=True)

    level = sub["value"]
    diff1 = level.diff().dropna()

    print(f"--- {LABEL} ---")
    print(f"  n obs: {len(level)}")

    adf_lvl = run_adf(level)
    kpss_lvl = run_kpss(level)
    print(f"\n  LEVEL:")
    print(f"    ADF:  stat={adf_lvl['stat']:.3f}  p={adf_lvl['pvalue']:.4f}  (lags={adf_lvl['lags']})")
    print(f"    KPSS: stat={kpss_lvl['stat']:.3f}  p={kpss_lvl['pvalue']:.4f}  (lags={kpss_lvl['lags']})")
    print(f"    → {verdict(adf_lvl['pvalue'], kpss_lvl['pvalue'])}")

    adf_d1 = run_adf(diff1)
    kpss_d1 = run_kpss(diff1)
    print(f"\n  FIRST DIFFERENCE:")
    print(f"    ADF:  stat={adf_d1['stat']:.3f}  p={adf_d1['pvalue']:.4f}  (lags={adf_d1['lags']})")
    print(f"    KPSS: stat={kpss_d1['stat']:.3f}  p={kpss_d1['pvalue']:.4f}  (lags={kpss_d1['lags']})")
    print(f"    → {verdict(adf_d1['pvalue'], kpss_d1['pvalue'])}")

    level_stationary = adf_lvl["pvalue"] < 0.05 and kpss_lvl["pvalue"] > 0.05
    diff_stationary = adf_d1["pvalue"] < 0.05 and kpss_d1["pvalue"] > 0.05

    if level_stationary:
        rec_d = 0
    elif diff_stationary:
        rec_d = 1
    else:
        rec_d = "2 (ambiguous — check overdifferencing before accepting)"

    print(f"\n  RECOMMENDED d: {rec_d}")


if __name__ == "__main__":
    main()

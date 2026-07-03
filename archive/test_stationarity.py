"""
test_stationarity.py
---------------------
Runs ADF and KPSS tests on each key series to determine the order of
integration (d) needed for SARIMAX. Tests both the raw level series and
the first difference, since most NHS/economic series here are expected
to be I(1).

Recall (EC205):
    ADF null hypothesis:  series HAS a unit root (non-stationary)
    KPSS null hypothesis: series is STATIONARY
    -> the tests have opposite nulls, so look for agreement:
         ADF rejects (p < 0.05) AND KPSS fails to reject (p > 0.05)
         = stationary
       ADF fails to reject AND KPSS rejects = non-stationary (unit root)
       Disagreement = ambiguous, lean on visual inspection + economic
       reasoning (e.g. structural breaks like COVID can distort both tests)

Run:
    python src/test_stationarity.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")  # KPSS throws interpolation warnings at small p

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

KEY_METRICS = {
    "RTT waiting list (level)": "Incomplete RTT pathways - Total waiting (mil) with estimates for missing data",
    "A&E attendances (flow)": "total_attendances",
    "Workforce FTE (level)": "FTE: All staff groups - All staff groups",
}


def run_adf(series: pd.Series) -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    return {"stat": result[0], "pvalue": result[1], "lags": result[2]}


def run_kpss(series: pd.Series) -> dict:
    # KPSS with 'c' regression (constant only) — standard for trend-having
    # series we plan to difference; 'ct' adds a deterministic trend term
    result = kpss(series.dropna(), regression="c", nlags="auto")
    return {"stat": result[0], "pvalue": result[1], "lags": result[2]}


def verdict(adf_p: float, kpss_p: float) -> str:
    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p > 0.05
    if adf_stationary and kpss_stationary:
        return "STATIONARY (both agree)"
    if not adf_stationary and not kpss_stationary:
        return "NON-STATIONARY (both agree)"
    return "AMBIGUOUS (tests disagree)"


def test_series(df: pd.DataFrame, metric_name: str, label: str):
    sub = df[df["metric"] == metric_name].sort_values("quarter").copy()
    if sub.empty:
        print(f"  ⚠ {label}: no data found")
        return None

    level = sub["value"].reset_index(drop=True)
    diff1 = level.diff().dropna()

    print(f"\n--- {label} ---")
    print(f"  n obs: {len(level)}")

    # Level
    adf_lvl = run_adf(level)
    kpss_lvl = run_kpss(level)
    print(f"  LEVEL:")
    print(f"    ADF:  stat={adf_lvl['stat']:.3f}  p={adf_lvl['pvalue']:.4f}  (lags={adf_lvl['lags']})")
    print(f"    KPSS: stat={kpss_lvl['stat']:.3f}  p={kpss_lvl['pvalue']:.4f}  (lags={kpss_lvl['lags']})")
    print(f"    → {verdict(adf_lvl['pvalue'], kpss_lvl['pvalue'])}")

    # First difference
    adf_d1 = run_adf(diff1)
    kpss_d1 = run_kpss(diff1)
    print(f"  FIRST DIFFERENCE:")
    print(f"    ADF:  stat={adf_d1['stat']:.3f}  p={adf_d1['pvalue']:.4f}  (lags={adf_d1['lags']})")
    print(f"    KPSS: stat={kpss_d1['stat']:.3f}  p={kpss_d1['pvalue']:.4f}  (lags={kpss_d1['lags']})")
    print(f"    → {verdict(adf_d1['pvalue'], kpss_d1['pvalue'])}")

    # Recommend d
    level_stationary = adf_lvl["pvalue"] < 0.05 and kpss_lvl["pvalue"] > 0.05
    diff_stationary = adf_d1["pvalue"] < 0.05 and kpss_d1["pvalue"] > 0.05

    if level_stationary:
        rec_d = 0
    elif diff_stationary:
        rec_d = 1
    else:
        rec_d = "2 (or ambiguous — inspect manually, may need 2nd diff or structural break handling)"

    print(f"  RECOMMENDED d: {rec_d}")

    return {
        "metric": label,
        "adf_level_p": adf_lvl["pvalue"],
        "kpss_level_p": kpss_lvl["pvalue"],
        "adf_diff1_p": adf_d1["pvalue"],
        "kpss_diff1_p": kpss_d1["pvalue"],
        "recommended_d": rec_d,
    }


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    print("=" * 75)
    print("STATIONARITY TESTS (ADF + KPSS) — determining d for SARIMAX")
    print("=" * 75)

    results = []
    for label, metric in KEY_METRICS.items():
        r = test_series(df, metric, label)
        if r:
            results.append(r)

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    for r in results:
        print(f"  {r['metric']}: d = {r['recommended_d']}")

    out = pd.DataFrame(results)
    out_path = PROCESSED_DIR / "stationarity_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\n✓ Saved results → {out_path}")
    print("\nNote: COVID is a structural break, not just a unit root — these")
    print("tests can be distorted by it. Cross-check d against the exog_config")
    print("dummies already in place; SARIMAX with exog often needs less")
    print("differencing than a plain ARIMA would, since the dummies absorb")
    print("some of the break that would otherwise look like non-stationarity.")


if __name__ == "__main__":
    main()

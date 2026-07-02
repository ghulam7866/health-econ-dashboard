"""
test_residuals.py
-----------------
Runs residual diagnostics on all fixed models to validate model assumptions.

Checks:
1. Ljung-Box test for autocorrelation
2. Jarque-Bera test for normality
3. ARCH test for heteroskedasticity
4. AR/MA root stability check

Usage:
    python test_residuals.py

Outputs:
    - Full console log -> reports/residual_diagnostics_log.txt
    - Summary CSV -> reports/residual_diagnostics_summary.csv

Last updated: 2026-07-02
"""

import sys
import os
import warnings
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(PROJECT_DIR, "data", "processed", "combined_quarterly.csv")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")

sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
import exog_config as ec

METRICS_TO_TEST = [
    "RTT waiting list (level)",
    "A&E attendances (flow)",
    "Workforce FTE (level)",
    "Bed occupancy (level)",
    "RTT % within 18 weeks (performance)",
    "A&E 12-hour decisions to admit (breach flow)",
]

DIAGNOSTIC_ALPHA = 0.05
INSTABILITY_MARGIN = 1.05


class Tee:
    """Write output to both terminal and log file simultaneously."""

    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def load_series(metric_key):
    """
    Load the time series for a given metric key.

    Parameters
    ----------
    metric_key : str
        The metric key from METRIC_NAMES

    Returns
    -------
    pd.DataFrame
        The filtered and sorted series data
    """
    df = pd.read_csv(INPUT_FILE)
    df["quarter"] = pd.to_datetime(df["quarter"])
    metric_label = ec.METRIC_NAMES[metric_key]

    sub = df[df["metric"] == metric_label].dropna(subset=["value"]).sort_values("quarter")

    if metric_key in ec.FIT_START_OVERRIDES:
        cutoff = pd.to_datetime(ec.FIT_START_OVERRIDES[metric_key])
        sub = sub[sub["quarter"] >= cutoff]

    return sub.reset_index(drop=True)


def run_residual_diagnostics(metric_key):
    """
    Run residual diagnostics for a single metric.

    Parameters
    ----------
    metric_key : str
        The metric key to test

    Returns
    -------
    dict
        Diagnostic results summary
    """
    print("\n" + "=" * 90)
    print(f"RESIDUAL DIAGNOSTICS: {metric_key}")
    print("=" * 90)

    sub = load_series(metric_key)
    y = sub["value"].values.astype(float)
    n = len(sub)

    cfg = ec.MODEL_CONFIG[metric_key]
    exog_cols = list(ec.EXOG_CONFIG.get(metric_key, []))

    scale = np.nanmax(np.abs(y))
    y_scaled = y / scale

    exog = sub[exog_cols].values.astype(float) if exog_cols else None

    print(f"Observations: {n}")
    print(f"Model: {cfg['order']} x {cfg['seasonal_order']}, trend={cfg['trend']}")
    print(f"Exog variables: {exog_cols if exog_cols else 'None'}")

    try:
        model = sm.tsa.statespace.SARIMAX(
            y_scaled,
            exog=exog,
            order=cfg["order"],
            seasonal_order=cfg["seasonal_order"],
            trend=cfg["trend"],
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        res = model.fit(disp=False)
        print(f"Fit status: SUCCESS")
        print(f"AICc: {res.aicc:.2f}")
    except Exception as e:
        print(f"Fit status: FAILED - {e}")
        return {
            "metric": metric_key,
            "n_obs": n,
            "fit_status": "FAILED",
            "ljung_box_p": None,
            "ljung_box_status": "FAILED",
            "jarque_bera_p": None,
            "jarque_bera_status": "FAILED",
            "arch_p": None,
            "arch_status": "FAILED",
            "root_status": "FAILED",
        }

    resid = np.asarray(res.resid, dtype=float)
    resid = resid[~np.isnan(resid)]
    n_resid = len(resid)

    print(f"Residuals: {n_resid} non-NaN values")

    # Ljung-Box Test (autocorrelation)
    print("\n--- Ljung-Box Test (H0: no autocorrelation) ---")

    lb_lags = min(10, n_resid // 2 - 1)
    if lb_lags >= 2:
        lb_result = acorr_ljungbox(resid, lags=lb_lags, return_df=True)
        lb_p = lb_result["lb_pvalue"].iloc[-1]
        lb_status = "PASS" if lb_p >= DIAGNOSTIC_ALPHA else "FAIL"
        print(f"  Lags tested: {lb_lags}")
        print(f"  p-value (lag {lb_lags}): {lb_p:.4f}")
        print(f"  Status: {lb_status}")
    else:
        lb_p = None
        lb_status = "INSUFFICIENT DATA"
        print(f"  Insufficient residuals for Ljung-Box test (n={n_resid})")

    # Jarque-Bera Test (normality)
    print("\n--- Jarque-Bera Test (H0: residuals are normal) ---")

    if n_resid >= 8:
        jb_stat, jb_p, skew, kurt = jarque_bera(resid)
        jb_status = "PASS" if jb_p >= DIAGNOSTIC_ALPHA else "FAIL"
        print(f"  Statistic: {jb_stat:.3f}")
        print(f"  p-value: {jb_p:.4f}")
        print(f"  Skewness: {skew:.3f}")
        print(f"  Kurtosis: {kurt:.3f}")
        print(f"  Status: {jb_status}")
    else:
        jb_p = None
        jb_status = "INSUFFICIENT DATA"
        print(f"  Insufficient residuals for Jarque-Bera test (n={n_resid})")

    # ARCH Test (heteroskedasticity)
    print("\n--- ARCH Test (H0: no heteroskedasticity) ---")

    if n_resid >= 12:
        arch_lags = min(4, n_resid // 3)
        try:
            arch_stat, arch_p, f_stat, f_p = het_arch(resid, nlags=arch_lags)
            arch_status = "PASS" if arch_p >= DIAGNOSTIC_ALPHA else "FAIL"
            print(f"  Lags: {arch_lags}")
            print(f"  p-value: {arch_p:.4f}")
            print(f"  Status: {arch_status}")
        except Exception as e:
            arch_p = None
            arch_status = "ERROR"
            print(f"  ARCH test failed: {e}")
    else:
        arch_p = None
        arch_status = "INSUFFICIENT DATA"
        print(f"  Insufficient residuals for ARCH test (n={n_resid})")

    # AR/MA Root Stability
    print("\n--- AR/MA Root Stability ---")

    ar_roots = np.abs(res.arroots) if len(res.arroots) else np.array([])
    ma_roots = np.abs(res.maroots) if len(res.maroots) else np.array([])

    root_status = "PASS"

    if ar_roots.size:
        min_ar = ar_roots.min()
        if min_ar < INSTABILITY_MARGIN:
            print(f"  WARNING: Near-unit-root AR (min|root|={min_ar:.3f})")
            root_status = "FAIL"
        else:
            print(f"  AR roots: min={min_ar:.3f}, max={ar_roots.max():.3f} [OK]")
    else:
        print("  No AR roots")

    if ma_roots.size:
        min_ma = ma_roots.min()
        if min_ma < INSTABILITY_MARGIN:
            print(f"  WARNING: Near-non-invertible MA (min|root|={min_ma:.3f})")
            root_status = "FAIL"
        else:
            print(f"  MA roots: min={min_ma:.3f}, max={ma_roots.max():.3f} [OK]")
    else:
        print("  No MA roots")

    print(f"  Overall root status: {root_status}")

    print("\n" + "-" * 50)
    print("SUMMARY")
    print("-" * 50)
    print(f"  Ljung-Box: {lb_status}")
    print(f"  Jarque-Bera: {jb_status}")
    print(f"  ARCH: {arch_status}")
    print(f"  Roots: {root_status}")

    return {
        "metric": metric_key,
        "n_obs": n,
        "fit_status": "SUCCESS",
        "aicc": res.aicc,
        "ljung_box_p": lb_p,
        "ljung_box_status": lb_status,
        "jarque_bera_p": jb_p,
        "jarque_bera_status": jb_status,
        "arch_p": arch_p,
        "arch_status": arch_status,
        "root_status": root_status,
        "overall_status": "PASS" if all(s in ["PASS", "INSUFFICIENT DATA"] for s in [lb_status, jb_status, arch_status, root_status]) else "FAIL",
    }


def main():
    """Main entry point - runs residual diagnostics for all metrics."""
    print("=" * 90)
    print("RESIDUAL DIAGNOSTICS")
    print("=" * 90)
    print(f"Run started: {datetime.now().isoformat()}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Metrics to test: {METRICS_TO_TEST}")
    print("=" * 90)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(REPORT_DIR, f"residual_diagnostics_log_{timestamp}.txt")
    summary_path = os.path.join(REPORT_DIR, f"residual_diagnostics_summary_{timestamp}.csv")

    sys.stdout = Tee(log_path)

    results = []

    for metric in METRICS_TO_TEST:
        try:
            result = run_residual_diagnostics(metric)
            results.append(result)
        except Exception as e:
            print(f"\nERROR processing {metric}: {e}")
            traceback.print_exc()
            results.append({
                "metric": metric,
                "n_obs": None,
                "fit_status": "ERROR",
                "aicc": None,
                "ljung_box_p": None,
                "ljung_box_status": "ERROR",
                "jarque_bera_p": None,
                "jarque_bera_status": "ERROR",
                "arch_p": None,
                "arch_status": "ERROR",
                "root_status": "ERROR",
                "overall_status": "ERROR",
            })

    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)

    summary_df = pd.DataFrame(results)

    display_cols = ['metric', 'n_obs', 'fit_status', 'aicc',
                    'ljung_box_status', 'jarque_bera_status',
                    'arch_status', 'root_status', 'overall_status']

    print("\n" + summary_df[display_cols].to_string(index=False))

    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Full log saved to: {log_path}")

    print("\n" + "=" * 90)
    print("FINAL VERDICT")
    print("=" * 90)

    failed_metrics = summary_df[summary_df["overall_status"] == "FAIL"]

    if not failed_metrics.empty:
        print(f"\n{len(failed_metrics)} metrics failed residual checks:")
        for _, row in failed_metrics.iterrows():
            failures = []
            if row["ljung_box_status"] == "FAIL":
                failures.append("Ljung-Box (autocorrelation)")
            if row["jarque_bera_status"] == "FAIL":
                failures.append("Jarque-Bera (non-normal residuals)")
            if row["arch_status"] == "FAIL":
                failures.append("ARCH (heteroskedasticity)")
            if row["root_status"] == "FAIL":
                failures.append("AR/MA roots (instability)")
            print(f"  FAIL: {row['metric']}: {', '.join(failures)}")
    else:
        print("\nAll metrics passed residual checks!")

    print("\n" + "=" * 90)
    print("Run completed.")


if __name__ == "__main__":
    main()
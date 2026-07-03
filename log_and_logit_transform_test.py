"""
test_transformations.py
-----------------------
Tests log and logit transformations to address heteroskedasticity /
non-stationarity flags identified in residual diagnostics:

1. RTT waiting list             - log transformation
2. RTT % within 18 weeks        - logit transformation
3. A&E 12-hour breach flow      - log1p transformation (NEW)

Usage:
    python test_transformations.py

Output:
    - Full console log     -> <REPORT_DIR>/transformation_test_log_<timestamp>.txt
    - Summary results      -> <REPORT_DIR>/transformation_test_summary_<timestamp>.csv
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from datetime import datetime

warnings.filterwarnings("ignore")

PROJECT_DIR = r"C:\Users\44782\Desktop\empirical project"
INPUT_FILE = os.path.join(PROJECT_DIR, "data", "processed", "combined_quarterly.csv")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")

# Matches the 22-point horizon seen in your last CLAMP WARNING output.
# If your production horizon differs (e.g. defined in exog_config.py), update this.
AE_FORECAST_HORIZON = 22

sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
import exog_config as ec


class Tee:
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
    df = pd.read_csv(INPUT_FILE)
    df["quarter"] = pd.to_datetime(df["quarter"])
    metric_label = ec.METRIC_NAMES[metric_key]
    sub = df[df["metric"] == metric_label].dropna(subset=["value"]).sort_values("quarter")
    if metric_key in ec.FIT_START_OVERRIDES:
        cutoff = pd.to_datetime(ec.FIT_START_OVERRIDES[metric_key])
        sub = sub[sub["quarter"] >= cutoff]
    return sub.reset_index(drop=True)


def run_diagnostics(y, exog, order, seasonal_order, trend, label, transform_name=""):
    """Run full diagnostics on a model."""
    scale = np.nanmax(np.abs(y))
    y_scaled = y / scale

    try:
        model = sm.tsa.statespace.SARIMAX(
            y_scaled,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        res = model.fit(disp=False)
    except Exception as e:
        print(f"  FIT FAILED: {e}")
        return None

    resid = np.asarray(res.resid, dtype=float)
    resid = resid[~np.isnan(resid)]
    n = len(resid)

    # Ljung-Box
    lb_lags = min(10, n // 2 - 1)
    if lb_lags >= 2:
        lb = acorr_ljungbox(resid, lags=lb_lags, return_df=True)
        lb_p = lb["lb_pvalue"].iloc[-1]
        lb_status = "PASS" if lb_p >= 0.05 else "FAIL"
    else:
        lb_p = None
        lb_status = "INSUFFICIENT"

    # Jarque-Bera
    if n >= 8:
        jb_stat, jb_p, skew, kurt = jarque_bera(resid)
        jb_status = "PASS" if jb_p >= 0.05 else "FAIL"
    else:
        jb_p = None
        jb_status = "INSUFFICIENT"
        skew = kurt = None

    # ARCH
    if n >= 12:
        arch_lags = min(4, n // 3)
        try:
            arch_stat, arch_p, f_stat, f_p = het_arch(resid, nlags=arch_lags)
            arch_status = "PASS" if arch_p >= 0.05 else "FAIL"
        except Exception:
            arch_p = None
            arch_status = "ERROR"
    else:
        arch_p = None
        arch_status = "INSUFFICIENT"

    # Root stability
    ar_roots = np.abs(res.arroots) if len(res.arroots) else np.array([])
    ma_roots = np.abs(res.maroots) if len(res.maroots) else np.array([])

    root_status = "PASS"
    if ar_roots.size and ar_roots.min() < 1.05:
        root_status = "FAIL"
    if ma_roots.size and ma_roots.min() < 1.05:
        root_status = "FAIL"

    return {
        "transform": transform_name,
        "aicc": res.aicc,
        "lb_p": lb_p,
        "lb_status": lb_status,
        "jb_p": jb_p,
        "jb_status": jb_status,
        "arch_p": arch_p,
        "arch_status": arch_status,
        "root_status": root_status,
        "skewness": skew,
        "kurtosis": kurt,
        "n_resid": n,
        "fit_status": "SUCCESS",
    }


def print_result_line(row):
    print(
        f"  {row['transform']:>6} | AICc: {row['aicc']:>8.2f} | "
        f"LB: {row['lb_status']} | JB: {row['jb_status']} | "
        f"ARCH: {row['arch_status']} | Roots: {row['root_status']} | "
        f"Skew: {row['skewness']:.2f} | Kurt: {row['kurtosis']:.2f}"
    )


def pick_best(results_rows):
    best = None
    for row in results_rows:
        if row["lb_status"] == "PASS" and row["root_status"] == "PASS":
            if best is None or row["aicc"] < best["aicc"]:
                best = row
    return best


def main():
    print("=" * 90)
    print("TRANSFORMATION TEST")
    print("=" * 90)
    print(f"Run started: {datetime.now().isoformat()}")
    print("=" * 90)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(REPORT_DIR, f"transformation_test_log_{timestamp}.txt")
    summary_path = os.path.join(REPORT_DIR, f"transformation_test_summary_{timestamp}.csv")
    sys.stdout = Tee(log_path)

    results = []

    # ============================================================
    # 1. RTT waiting list - Log Transformation
    # ============================================================
    print("\n" + "=" * 90)
    print("TEST 1: RTT waiting list - Log Transformation")
    print("=" * 90)

    metric = "RTT waiting list (level)"
    cfg = ec.MODEL_CONFIG[metric]
    exog_cols = ec.EXOG_CONFIG.get(metric, [])
    sub = load_series(metric)
    y = sub["value"].values.astype(float)
    exog = sub[exog_cols].values.astype(float) if exog_cols else None

    print("\n--- Baseline (no transformation) ---")
    orig_results = run_diagnostics(y, exog, cfg["order"], cfg["seasonal_order"], cfg["trend"], metric)
    if orig_results:
        orig_results["metric"] = metric
        orig_results["transform"] = "None"
        results.append(orig_results)
        print_result_line(orig_results)

    print("\n--- Log Transformation ---")
    y_log = np.log(y)
    log_results = run_diagnostics(y_log, exog, cfg["order"], cfg["seasonal_order"], cfg["trend"], metric, "Log")
    if log_results:
        log_results["metric"] = metric
        log_results["transform"] = "Log"
        results.append(log_results)
        print_result_line(log_results)

    # ============================================================
    # 2. RTT % within 18 weeks - Logit Transformation
    # ============================================================
    print("\n" + "=" * 90)
    print("TEST 2: RTT % within 18 weeks - Logit Transformation")
    print("=" * 90)

    metric = "RTT % within 18 weeks (performance)"
    cfg = ec.MODEL_CONFIG[metric]
    exog_cols = ec.EXOG_CONFIG.get(metric, [])
    sub = load_series(metric)
    y = sub["value"].values.astype(float)
    exog = sub[exog_cols].values.astype(float) if exog_cols else None

    y_clipped = np.clip(y, 0.001, 0.999)

    print("\n--- Baseline (no transformation) ---")
    orig_results = run_diagnostics(y, exog, cfg["order"], cfg["seasonal_order"], cfg["trend"], metric)
    if orig_results:
        orig_results["metric"] = metric
        orig_results["transform"] = "None"
        results.append(orig_results)
        print_result_line(orig_results)

    print("\n--- Logit Transformation ---")
    y_logit = np.log(y_clipped / (1 - y_clipped))
    logit_results = run_diagnostics(y_logit, exog, cfg["order"], cfg["seasonal_order"], cfg["trend"], metric, "Logit")
    if logit_results:
        logit_results["metric"] = metric
        logit_results["transform"] = "Logit"
        results.append(logit_results)
        print_result_line(logit_results)

    # ============================================================
    # 3. A&E 12-hour breach flow - Log1p Transformation (NEW)
    # ============================================================
    print("\n" + "=" * 90)
    print("TEST 3: A&E 12-hour breach flow - Log1p Transformation")
    print("=" * 90)

    metric = "A&E 12-hour decisions to admit (breach flow)"
    cfg = ec.MODEL_CONFIG[metric]
    exog_cols = ec.EXOG_CONFIG.get(metric, [])
    sub = load_series(metric)
    y = sub["value"].values.astype(float)
    exog = sub[exog_cols].values.astype(float) if exog_cols else None

    print(f"\n  n_obs after FIT_START_OVERRIDES cutoff: {len(y)}")
    print(f"  value range: {y.min():.1f} -> {y.max():.1f}")

    print("\n--- Baseline (no transformation, current production spec) ---")
    orig_results = run_diagnostics(y, exog, cfg["order"], cfg["seasonal_order"], cfg["trend"], metric)
    if orig_results:
        orig_results["metric"] = metric
        orig_results["transform"] = "None"
        results.append(orig_results)
        print_result_line(orig_results)

    print("\n--- Log1p Transformation (log(1+y), safe against near-zero counts) ---")
    y_log1p = np.log1p(y)
    log1p_results = run_diagnostics(
        y_log1p, exog, cfg["order"], cfg["seasonal_order"], cfg["trend"], metric, "Log1p"
    )
    if log1p_results:
        log1p_results["metric"] = metric
        log1p_results["transform"] = "Log1p"
        results.append(log1p_results)
        print_result_line(log1p_results)

    # ============================================================
    # 4. A&E 12-hour breach flow - Differencing/order comparison (NEW)
    # ============================================================
    print("\n" + "=" * 90)
    print("TEST 4: A&E 12-hour breach flow - Order/Seasonal Differencing Comparison")
    print("=" * 90)
    print(
        "\n  NOTE: future exog here is a simple forward-fill of the last observed row,\n"
        "  NOT the production last-step linear extrapolation used in master_forecast_engine.py.\n"
        "  Use this to compare specs against each other, not as production forecast numbers."
    )

    metric = "A&E 12-hour decisions to admit (breach flow)"
    cfg = ec.MODEL_CONFIG[metric]
    exog_cols = ec.EXOG_CONFIG.get(metric, [])
    sub = load_series(metric)
    y = sub["value"].values.astype(float)
    exog = sub[exog_cols].values.astype(float) if exog_cols else None
    future_exog = np.tile(exog[-1], (AE_FORECAST_HORIZON, 1)) if exog is not None else None

    candidate_specs = [
        {"label": "Production (d=1,D=0)", "order": cfg["order"], "seasonal_order": cfg["seasonal_order"], "trend": cfg["trend"]},
        {"label": "d=0,D=0,trend=c",      "order": (1, 0, 0),    "seasonal_order": (0, 0, 0, 4),          "trend": "c"},
        {"label": "d=0,D=1(seasonal)",    "order": (0, 0, 0),    "seasonal_order": (0, 1, 0, 4),          "trend": None},
        {"label": "d=1,D=0,no seasonal AR","order": (0, 1, 1),   "seasonal_order": (0, 0, 0, 4),          "trend": None},
    ]

    forecast_results = []
    for spec in candidate_specs:
        print(f"\n--- {spec['label']}: order={spec['order']}, seasonal={spec['seasonal_order']}, trend={spec['trend']} ---")
        try:
            model = sm.tsa.statespace.SARIMAX(
                y,
                exog=exog,
                order=spec["order"],
                seasonal_order=spec["seasonal_order"],
                trend=spec["trend"],
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            res = model.fit(disp=False)
        except Exception as e:
            print(f"  FIT FAILED: {e}")
            continue

        fc = res.get_forecast(steps=AE_FORECAST_HORIZON, exog=future_exog)
        mean_fc = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        ci_lower = ci[:, 0] if hasattr(ci, "shape") else ci.iloc[:, 0].values

        n_neg_mean = int(np.sum(mean_fc < 0))
        n_neg_ci = int(np.sum(ci_lower < 0))
        print(f"  AICc: {res.aicc:.2f}")
        print(f"  mean_fc range: [{mean_fc.min():.1f}, {mean_fc.max():.1f}]")
        print(f"  negative mean_fc points: {n_neg_mean}/{AE_FORECAST_HORIZON}")
        print(f"  negative CI-lower points: {n_neg_ci}/{AE_FORECAST_HORIZON}")

        forecast_results.append({
            "spec": spec["label"],
            "aicc": res.aicc,
            "n_neg_mean": n_neg_mean,
            "n_neg_ci": n_neg_ci,
            "mean_fc_min": mean_fc.min(),
            "mean_fc_max": mean_fc.max(),
        })

    print("\n--- Test 4 comparison ---")
    for r in forecast_results:
        print(
            f"  {r['spec']:<28} | AICc: {r['aicc']:>8.2f} | "
            f"neg mean pts: {r['n_neg_mean']:>2}/{AE_FORECAST_HORIZON} | "
            f"neg CI-lower pts: {r['n_neg_ci']:>2}/{AE_FORECAST_HORIZON} | "
            f"mean_fc range: [{r['mean_fc_min']:.0f}, {r['mean_fc_max']:.0f}]"
        )

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON")
    print("=" * 90)

    summary_df = pd.DataFrame(results)

    print("\nRTT waiting list:")
    rtt_results = summary_df[summary_df["metric"] == "RTT waiting list (level)"]
    for _, row in rtt_results.iterrows():
        print_result_line(row)

    print("\nRTT % within 18 weeks:")
    rtt_pct_results = summary_df[summary_df["metric"] == "RTT % within 18 weeks (performance)"]
    for _, row in rtt_pct_results.iterrows():
        print_result_line(row)

    print("\nA&E 12-hour breach flow:")
    ae_results = summary_df[summary_df["metric"] == "A&E 12-hour decisions to admit (breach flow)"]
    for _, row in ae_results.iterrows():
        print_result_line(row)

    # Save summary
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Full log saved to: {log_path}")

    # ============================================================
    # Recommendation
    # ============================================================
    print("\n" + "=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)

    rtt_best = pick_best([r for _, r in rtt_results.iterrows()])
    if rtt_best is not None:
        print(f"\nRTT waiting list: Best model is {rtt_best['transform']} transformation (AICc: {rtt_best['aicc']:.2f})")
        print("  -> Keep current model (no transformation)" if rtt_best["transform"] == "None"
              else f"  -> Consider applying {rtt_best['transform']} transformation")

    rtt_pct_best = pick_best([r for _, r in rtt_pct_results.iterrows()])
    if rtt_pct_best is not None:
        print(f"\nRTT % within 18 weeks: Best model is {rtt_pct_best['transform']} transformation (AICc: {rtt_pct_best['aicc']:.2f})")
        print("  -> Keep current model (no transformation)" if rtt_pct_best["transform"] == "None"
              else f"  -> Consider applying {rtt_pct_best['transform']} transformation")

    ae_best = pick_best([r for _, r in ae_results.iterrows()])
    if ae_best is not None:
        print(f"\nA&E 12-hour breach flow: Best model is {ae_best['transform']} transformation (AICc: {ae_best['aicc']:.2f})")
        print("  -> Keep current model (no transformation)" if ae_best["transform"] == "None"
              else f"  -> Consider applying {ae_best['transform']} transformation")
    else:
        print(
            "\nA&E 12-hour breach flow: NEITHER spec passed both Ljung-Box and root-stability checks. "
            "With only ~22 obs post-2021 cutoff, treat both results as provisional -- "
            "look at AICc and ARCH/JB status manually rather than relying on the auto-pick."
        )

    print("\n" + "=" * 90)
    print("Run completed.")


if __name__ == "__main__":
    main()
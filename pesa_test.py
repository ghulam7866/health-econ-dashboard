"""
test_pesa_annual.py
-------------------
Runs diagnostic, backtest, and residual diagnostics for PESA Health spend (level).
PESA is annual data (not quarterly), so this script handles it separately.

Usage:
    python test_pesa_annual.py

Outputs:
  - Full console log  -> <REPORT_DIR>/pesa_annual_diagnostics_log.txt
  - Summary results   -> <REPORT_DIR>/pesa_annual_diagnostics_summary.csv
"""

import sys
import os
import warnings
import traceback
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_DIR = r"C:\Users\44782\Desktop\empirical project"
INPUT_FILE = os.path.join(PROJECT_DIR, "data", "processed", "combined_quarterly.csv")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")

# Add src to path
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
import exog_config as ec

METRIC_KEY = "PESA Health spend (level)"
METRIC_LABEL = ec.METRIC_NAMES[METRIC_KEY]
MODEL_CONFIG = ec.MODEL_CONFIG[METRIC_KEY]
EXOG_CONFIG = ec.EXOG_CONFIG.get(METRIC_KEY, [])

DIAGNOSTIC_ALPHA = 0.05
INSTABILITY_MARGIN = 1.05

# Annual horizons (years ahead)
DEFAULT_HORIZONS = [1, 2, 3, 4, 5, 8, 10]
DEFAULT_MIN_TRAIN_SIZE = 6


# ---------------------------------------------------------------------------
# Tee logger
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Data loading (annual)
# ---------------------------------------------------------------------------
def load_annual_series():
    """Load PESA annual data from combined_quarterly.csv.
    Annual data is forward-filled across quarters, so we take the first
    quarter of each year (April) as the annual observation."""
    df = pd.read_csv(INPUT_FILE)
    df["quarter"] = pd.to_datetime(df["quarter"])
    
    # Filter to PESA metric
    sub = df[df["metric"] == METRIC_LABEL].dropna(subset=["value"]).sort_values("quarter")
    
    if sub.empty:
        raise ValueError(f"No data found for {METRIC_LABEL}")
    
    # Extract annual data - take the first quarter of each year (April)
    sub["year"] = sub["quarter"].dt.year
    annual = sub.groupby("year").first().reset_index(drop=False)
    annual = annual.rename(columns={"quarter": "date"})
    
    print(f"Loaded {len(annual)} annual observations.")
    print(f"Range: {annual['date'].iloc[0].date()} to {annual['date'].iloc[-1].date()}")
    
    return annual


# ---------------------------------------------------------------------------
# Candidate generation for annual data (NO seasonality)
# ---------------------------------------------------------------------------
def build_candidates():
    """Build candidate models for annual data - NO seasonal components."""
    cfg = MODEL_CONFIG
    order = tuple(cfg["order"])
    trend = cfg["trend"]
    exog_cols = list(EXOG_CONFIG)
    p, d, q = order

    candidates = []
    candidates.append((
        f"PRODUCTION {order} {trend}, exog={exog_cols or 'none'}",
        order, trend, list(exog_cols),
    ))

    # --- trend flip ---
    alt_trend = "n" if trend == "c" else "c"
    candidates.append((f"ALT: trend flip -> {alt_trend!r}", order, alt_trend, list(exog_cols)))

    # --- exog checks ---
    if exog_cols:
        candidates.append((f"ALT: no exog (drop {exog_cols})", order, trend, []))
        if len(exog_cols) > 1:
            for col in exog_cols:
                candidates.append((f"EXOG ISOLATION: {col} only", order, trend, [col]))

    # --- AR/MA/ARMA structural swaps ---
    if (p, q) != (1, 0):
        candidates.append((f"ALT: AR(1) only ({1},{d},{0})", (1, d, 0), trend, list(exog_cols)))
    if (p, q) != (0, 1):
        candidates.append((f"ALT: MA(1) only ({0},{d},{1})", (0, d, 1), trend, list(exog_cols)))
    if (p, q) != (1, 1):
        candidates.append((f"ALT: ARMA(1,1) ({1},{d},{1})", (1, d, 1), trend, list(exog_cols)))

    # --- simplest baseline ---
    candidates.append((f"BASELINE: white noise (0,{d},0)", (0, d, 0), trend, []))

    # de-dupe
    seen = set()
    deduped = []
    for c in candidates:
        key = (c[1], c[2], tuple(c[3]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Diagnostic sequence
# ---------------------------------------------------------------------------
def diagnostic_significance(y, exog, exog_cols):
    print("\n--- [1/5] Significance screen (OLS, exog vs. level) ---")
    if not exog_cols:
        print("  No exog columns configured for this metric - skipping.")
        return
    X = sm.add_constant(exog)
    model = sm.OLS(y, X).fit()
    for name, coef, pval in zip(exog_cols, model.params[1:], model.pvalues[1:]):
        flag = "OK" if pval < DIAGNOSTIC_ALPHA else "FLAG: not significant"
        print(f"  {name:<25} coef={coef:>14.4f}  p={pval:.4f}  [{flag}]")


def diagnostic_quadratic_trend(y):
    print("\n--- [2/5] Quadratic trend check ---")
    t = np.arange(len(y), dtype=float)
    X = sm.add_constant(np.column_stack([t, t ** 2]))
    model = sm.OLS(y, X).fit()
    coef_t2, pval_t2 = model.params[2], model.pvalues[2]
    flag = "FLAG: significant curvature" if pval_t2 < DIAGNOSTIC_ALPHA else "OK: no significant curvature"
    print(f"  t^2 coef={coef_t2:.6f}  p={pval_t2:.4f}  [{flag}]")


def diagnostic_stationarity(y):
    print("\n--- [3/5] Stationarity checks (ADF + KPSS) ---")

    def _run(label, series):
        if len(series) < 10:
            print(f"  {label:<20} (too few observations after differencing - skipped)")
            return
        adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
        adf_verdict = "stationary" if adf_p < DIAGNOSTIC_ALPHA else "FLAG: non-stationary (ADF)"
        kpss_verdict = "stationary" if kpss_p >= DIAGNOSTIC_ALPHA else "FLAG: non-stationary (KPSS)"
        print(f"  {label:<20} ADF p={adf_p:.4f} [{adf_verdict}]   KPSS p={kpss_p:.4f} [{kpss_verdict}]")

    _run("level", y)
    _run("1st diff", np.diff(y, n=1))


def diagnostic_order_selection(y, exog):
    print("\n--- [4/5] Order selection (AICc grid search, suggestions only) ---")
    scale = np.nanmax(np.abs(y))
    y_scaled = y / scale

    # Annual grid - no seasonality
    p_grid = (0, 1, 2)
    d_grid = (0, 1)
    q_grid = (0, 1, 2)

    scored = []
    n_tried, n_failed = 0, 0
    for p in p_grid:
        for d in d_grid:
            for q in q_grid:
                order = (p, d, q)
                trend = "c" if d == 0 else None
                n_tried += 1
                try:
                    model = sm.tsa.arima.ARIMA(
                        y_scaled,
                        exog=exog,
                        order=order,
                        trend=trend,
                    )
                    res = model.fit()
                    if np.isfinite(res.aicc):
                        scored.append((res.aicc, order, trend))
                except Exception:
                    n_failed += 1
                    continue

    scored.sort(key=lambda row: row[0])
    print(f"  Fitted {n_tried - n_failed}/{n_tried} grid combinations successfully.")
    for aicc, order, trend in scored[:5]:
        print(f"  AICc={aicc:>10.2f}  order={order}  trend={trend!r}")
    if not scored:
        print("  No grid combination converged.")


def diagnostic_instability(df, candidates):
    print("\n--- [5/5] Instability check (AR/MA roots, full-history fit per candidate) ---")
    y_full = df["value"].values.astype(float)
    scale = np.nanmax(np.abs(y_full))
    y_scaled = y_full / scale

    for label, order, trend, exog_cols in candidates:
        exog_full = df[list(exog_cols)].values.astype(float) if exog_cols else None
        try:
            model = sm.tsa.arima.ARIMA(
                y_scaled,
                exog=exog_full,
                order=order,
                trend=trend,
            )
            res = model.fit()
        except Exception as exc:
            print(f"  {label:<55} FIT FAILED ({type(exc).__name__})")
            continue

        # Check roots
        ar_roots = np.abs(res.arroots) if hasattr(res, 'arroots') and len(res.arroots) else np.array([])
        ma_roots = np.abs(res.maroots) if hasattr(res, 'maroots') and len(res.maroots) else np.array([])

        flags = []
        if ar_roots.size and ar_roots.min() < INSTABILITY_MARGIN:
            flags.append(f"near-unit-root AR (min|root|={ar_roots.min():.3f})")
        if ma_roots.size and ma_roots.min() < INSTABILITY_MARGIN:
            flags.append(f"near-non-invertible MA (min|root|={ma_roots.min():.3f})")

        status = "; ".join(flags) if flags else "OK"
        print(f"  {label:<55} [{status}]")


# ---------------------------------------------------------------------------
# Backtest logic
# ---------------------------------------------------------------------------
def fit_and_forecast(y_scaled, exog, order, trend, horizon):
    try:
        model = sm.tsa.arima.ARIMA(
            y_scaled,
            exog=exog,
            order=order,
            trend=trend,
        )
        res = model.fit()

        if exog is not None:
            last_val = exog[-1, :]
            last_diff = exog[-1, :] - exog[-2, :] if exog.shape[0] > 1 else np.zeros_like(last_val)
            steps = np.arange(1, horizon + 1).reshape(-1, 1)
            future_exog = last_val + steps * last_diff
        else:
            future_exog = None

        fc = res.get_forecast(steps=horizon, exog=future_exog)
        mean = fc.predicted_mean
        mean = mean.to_numpy() if hasattr(mean, "to_numpy") else np.asarray(mean)
        ci = fc.conf_int(alpha=0.05)
        ci = ci.to_numpy() if hasattr(ci, "to_numpy") else np.asarray(ci)
        return mean[-1], ci[-1, 0], ci[-1, 1]
    except Exception:
        return None


def run_backtest(df, candidates, horizons, min_train_size):
    y_full = df["value"].values.astype(float)
    n = len(y_full)
    scale = np.nanmax(np.abs(y_full))

    exog_cache = {}
    for label, order, trend, exog_cols in candidates:
        key = tuple(exog_cols)
        if key not in exog_cache:
            exog_cache[key] = df[list(exog_cols)].values.astype(float) if exog_cols else None

    results = {label: {h: [] for h in horizons} for label, *_ in candidates}

    for t in range(min_train_size, n - 1):
        y_train = y_full[: t + 1] / scale
        last_known = y_full[t]

        for h in horizons:
            target_idx = t + h
            if target_idx >= n:
                continue
            actual = y_full[target_idx]

            for label, order, trend, exog_cols in candidates:
                exog_full = exog_cache[tuple(exog_cols)]
                exog_train = exog_full[: t + 1] if exog_full is not None else None

                out = fit_and_forecast(y_train, exog_train, order, trend, h)
                if out is None:
                    continue
                fc_mean_scaled, ci_lo_scaled, ci_hi_scaled = out
                results[label][h].append({
                    "forecast": fc_mean_scaled * scale,
                    "ci_lo": ci_lo_scaled * scale,
                    "ci_hi": ci_hi_scaled * scale,
                    "actual": actual,
                    "last_known": last_known,
                })

    return results


def _compute_metrics(rows):
    if not rows:
        return None

    forecasts = np.array([r["forecast"] for r in rows])
    actuals = np.array([r["actual"] for r in rows])
    ci_lo = np.array([r["ci_lo"] for r in rows])
    ci_hi = np.array([r["ci_hi"] for r in rows])
    last_known = np.array([r["last_known"] for r in rows])

    errors = forecasts - actuals
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)

    actual_dir = np.sign(actuals - last_known)
    forecast_dir = np.sign(forecasts - last_known)
    valid = actual_dir != 0
    dir_acc = np.mean(actual_dir[valid] == forecast_dir[valid]) if valid.sum() > 0 else float("nan")

    ci_coverage = np.mean((actuals >= ci_lo) & (actuals <= ci_hi))

    return {"n": len(rows), "rmse": rmse, "mae": mae, "bias": bias,
            "dir_acc": dir_acc, "ci_coverage": ci_coverage}


def score_results(results, horizons, candidates):
    print("\n" + "=" * 90)
    print("BACKTEST RESULTS (rolling-origin, expanding window)")
    print("=" * 90)

    for h in horizons:
        print(f"\n--- Horizon = {h} year(s) ahead ---")
        print(f"{'Spec':<55}{'n':>4}{'RMSE':>9}{'MAE':>9}{'Bias':>9}{'DirAcc':>8}{'CICov':>8}")
        for label, *_ in candidates:
            rows = results[label][h]
            metrics = _compute_metrics(rows)
            if metrics is None:
                print(f"{label:<55}  (no folds completed)")
                continue
            print(
                f"{label:<55}{metrics['n']:>4}{metrics['rmse']:>9.4f}{metrics['mae']:>9.4f}"
                f"{metrics['bias']:>+9.4f}{metrics['dir_acc']:>8.2f}{metrics['ci_coverage']:>8.2f}"
            )


def residual_diagnostics(df):
    print("\n" + "=" * 90)
    print("RESIDUAL DIAGNOSTICS (production spec, full-history fit)")
    print("=" * 90)

    y = df["value"].values.astype(float)
    scale = np.nanmax(np.abs(y))
    y_scaled = y / scale
    exog_cols = list(EXOG_CONFIG)
    exog = df[exog_cols].values.astype(float) if exog_cols else None

    verdict = {"ljung_box": "N/A", "jarque_bera": "N/A", "arch": "N/A", "fit_status": "OK"}

    try:
        model = sm.tsa.arima.ARIMA(
            y_scaled,
            exog=exog,
            order=MODEL_CONFIG["order"],
            trend=MODEL_CONFIG["trend"],
        )
        res = model.fit()
    except Exception as exc:
        print(f"  FIT FAILED on production spec: {type(exc).__name__}: {exc}")
        verdict["fit_status"] = "FIT FAILED"
        return verdict

    resid = np.asarray(res.resid, dtype=float)
    resid = resid[~np.isnan(resid)]
    n = len(resid)

    # Ljung-Box
    lb_lags = [lag for lag in (2, 4, 6) if lag < n]
    if lb_lags:
        lb = sm.stats.acorr_ljungbox(resid, lags=lb_lags, return_df=True)
        print("\n  Ljung-Box (H0: no autocorrelation left in residuals)")
        any_flag = False
        for lag, row in lb.iterrows():
            p = row["lb_pvalue"]
            flag = "OK" if p >= DIAGNOSTIC_ALPHA else "FLAG: residual autocorrelation remains"
            if p < DIAGNOSTIC_ALPHA:
                any_flag = True
            print(f"    lag={lag:<3} stat={row['lb_stat']:.3f}  p={p:.4f}  [{flag}]")
        verdict["ljung_box"] = "FLAG" if any_flag else "OK"
    else:
        print("\n  Ljung-Box: too few residuals to test - skipped.")

    # Jarque-Bera
    if n >= 8:
        jb_stat, jb_p, skew, kurt = sm.stats.stattools.jarque_bera(resid)
        flag = "OK" if jb_p >= DIAGNOSTIC_ALPHA else "FLAG: residuals non-normal"
        print(f"\n  Jarque-Bera (H0: residuals normal): stat={jb_stat:.3f}  p={jb_p:.4f}  "
              f"skew={skew:.3f}  kurtosis={kurt:.3f}  [{flag}]")
        verdict["jarque_bera"] = "OK" if jb_p >= DIAGNOSTIC_ALPHA else "FLAG"
    else:
        print("\n  Jarque-Bera: too few residuals to test - skipped.")

    # ARCH LM
    if n >= 12:
        try:
            nlags = max(1, min(3, n // 3))
            arch_stat, arch_p, f_stat, f_p = het_arch(resid, nlags=nlags)
            flag = "OK" if arch_p >= DIAGNOSTIC_ALPHA else "FLAG: heteroskedasticity (ARCH effects)"
            print(f"\n  ARCH LM (H0: no heteroskedasticity), nlags={nlags}: "
                  f"stat={arch_stat:.3f}  p={arch_p:.4f}  [{flag}]")
            verdict["arch"] = "OK" if arch_p >= DIAGNOSTIC_ALPHA else "FLAG"
        except Exception as exc:
            print(f"\n  ARCH LM: test failed ({type(exc).__name__}) - skipped.")
    else:
        print("\n  ARCH LM: too few residuals to test - skipped.")

    return verdict


def main():
    print("=" * 90)
    print("PESA ANNUAL DIAGNOSTICS")
    print("=" * 90)
    print(f"Run started: {datetime.now().isoformat()}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Metric: {METRIC_KEY}")
    print("=" * 90)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(REPORT_DIR, f"pesa_annual_diagnostics_log_{timestamp}.txt")
    summary_path = os.path.join(REPORT_DIR, f"pesa_annual_diagnostics_summary_{timestamp}.csv")
    sys.stdout = Tee(log_path)

    # Load data
    df = load_annual_series()
    y = df["value"].values.astype(float)

    # Build candidates
    candidates = build_candidates()
    print(f"\nCandidates ({len(candidates)}): {[c[0] for c in candidates]}")

    # Diagnostic sequence
    print("\n" + "=" * 90)
    print("DIAGNOSTIC SEQUENCE")
    print("=" * 90)

    exog_cols = list(EXOG_CONFIG)
    exog = df[exog_cols].values.astype(float) if exog_cols else None

    diagnostic_significance(y, exog, exog_cols)
    diagnostic_quadratic_trend(y)
    diagnostic_stationarity(y)
    diagnostic_order_selection(y, exog)
    diagnostic_instability(df, candidates)

    # Backtest
    horizons = DEFAULT_HORIZONS
    min_train = DEFAULT_MIN_TRAIN_SIZE

    # Adjust for short series
    n = len(df)
    while n - min_train - max(horizons) < 3 and min_train > 4:
        min_train -= 1
    if n - min_train - max(horizons) < 3:
        horizons = [h for h in horizons if n - min_train - h >= 3]
        if not horizons:
            horizons = [1]

    print(f"\nHorizons tested: {horizons} | Min training window: {min_train}")

    results = run_backtest(df, candidates, horizons, min_train)
    score_results(results, horizons, candidates)

    # Residual diagnostics
    resid_verdict = residual_diagnostics(df)

    # Summary
    prod_label = candidates[0][0]
    longest_h = max(horizons)
    prod_metrics = _compute_metrics(results[prod_label][longest_h])

    summary_row = {
        "metric_key": METRIC_KEY,
        "n_obs": n,
        "longest_horizon_tested": longest_h,
        "production_spec": prod_label,
        "n_folds": prod_metrics["n"] if prod_metrics else 0,
        "rmse": prod_metrics["rmse"] if prod_metrics else np.nan,
        "mae": prod_metrics["mae"] if prod_metrics else np.nan,
        "bias": prod_metrics["bias"] if prod_metrics else np.nan,
        "dir_acc": prod_metrics["dir_acc"] if prod_metrics else np.nan,
        "ci_coverage": prod_metrics["ci_coverage"] if prod_metrics else np.nan,
        "ljung_box": resid_verdict["ljung_box"],
        "jarque_bera": resid_verdict["jarque_bera"],
        "arch_heteroskedasticity": resid_verdict["arch"],
        "residual_fit_status": resid_verdict["fit_status"],
    }

    print("\n" + "=" * 90)
    print("COMPACT SUMMARY")
    print("=" * 90)
    for key, val in summary_row.items():
        if key not in ["production_spec", "metric_key"]:
            print(f"{key}: {val}")

    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Full log saved to: {log_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
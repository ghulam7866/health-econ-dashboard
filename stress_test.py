"""
stress_test.py
---------------
Runs diagnostic, backtest, bias summary, and residual diagnostics for a single metric.

This script is used to test and validate individual metrics from the forecasting pipeline.
It runs the full diagnostic sequence including stationarity tests, order selection,
backtesting with multiple candidate models, and residual diagnostics.

Usage:
    python stress_test.py "A&E attendances (flow)"
    python stress_test.py "RTT waiting list (level)"
    python stress_test.py "GP total appointments (flow)"
    python stress_test.py --list

Outputs:
    - Full console log  -> reports/backtest_[metric]_log_[timestamp].txt
    - Summary results   -> reports/backtest_[metric]_summary_[timestamp].csv

Last updated: 2026-07-02
"""

import sys
import os
import itertools
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
# Path configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")

INPUT_FILE = os.path.join(DATA_DIR, "combined_quarterly.csv")

sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
import exog_config as ec

SEASONAL_PERIOD = 4
DIAGNOSTIC_ALPHA = 0.05
INSTABILITY_MARGIN = 1.05

DEFAULT_HORIZONS = [1, 2, 3, 4, 8, 12]
DEFAULT_MIN_TRAIN_SIZE = 6

ORDER_GRID_P = (0, 1, 2)
ORDER_GRID_Q = (0, 1, 2)
ORDER_GRID_D = (0, 1)
ORDER_GRID_SEASONAL_P = (0, 1)
ORDER_GRID_SEASONAL_Q = (0, 1)
ORDER_GRID_SEASONAL_D = (0, 1)
ORDER_SELECTION_TOP_N = 5

BIAS_SUMMARY_LABEL_MATCHES = ["PRODUCTION", "ALT", "REGRESSION CHECK", "BASELINE", "EXOG ISOLATION"]


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

    Raises
    ------
    ValueError
        If no data is found for the metric
    """
    df = pd.read_csv(INPUT_FILE)
    df["quarter"] = pd.to_datetime(df["quarter"])
    metric_label = ec.METRIC_NAMES[metric_key]

    sub = df[df["metric"] == metric_label].dropna(subset=["value"]).sort_values("quarter")

    if sub.empty:
        raise ValueError(f"No data found for metric '{metric_key}' ({metric_label})")

    if metric_key in ec.FIT_START_OVERRIDES:
        cutoff = pd.to_datetime(ec.FIT_START_OVERRIDES[metric_key])
        before = len(sub)
        sub = sub[sub["quarter"] >= cutoff]
        print(f"  FIT_START_OVERRIDE applied: restricting to >= {cutoff.date()} "
              f"({before} -> {len(sub)} observations)")

    return sub.reset_index(drop=True)


def resolve_horizons_and_min_train(metric_key, sub):
    """
    Determine appropriate horizons and minimum training window for a series.

    Adjusts horizons based on series length and model configuration.

    Parameters
    ----------
    metric_key : str
        The metric key
    sub : pd.DataFrame
        The series data

    Returns
    -------
    tuple
        (horizons, min_train_size)
    """
    cfg = ec.MODEL_CONFIG[metric_key]
    max_h = cfg.get("horizons", DEFAULT_HORIZONS[-1])
    horizons = [h for h in DEFAULT_HORIZONS if h <= max_h]

    if not horizons:
        horizons = [max_h]

    n = len(sub)
    min_train = DEFAULT_MIN_TRAIN_SIZE

    while n - min_train - max(horizons) < 3 and min_train > 4:
        min_train -= 1

    if n - min_train - max(horizons) < 3:
        horizons = [h for h in horizons if n - min_train - h >= 3]
        if not horizons:
            horizons = [1]

    if min_train != DEFAULT_MIN_TRAIN_SIZE or horizons != [h for h in DEFAULT_HORIZONS if h <= max_h]:
        print(f"  Adjusted for series length (n={n}): MIN_TRAIN_SIZE={min_train}, HORIZONS={horizons}")

    return horizons, min_train


def build_candidates(metric_key):
    """
    Build a list of candidate models to test for a given metric.

    Includes production spec and various alternatives (trend flip, exog variations,
    AR/MA/ARMA swaps, and white noise baseline).

    Parameters
    ----------
    metric_key : str
        The metric key

    Returns
    -------
    list
        List of candidate model specifications
    """
    cfg = ec.MODEL_CONFIG[metric_key]
    order = tuple(cfg["order"])
    seasonal_order = tuple(cfg["seasonal_order"])
    trend = cfg["trend"]
    exog_cols = list(ec.EXOG_CONFIG.get(metric_key, []))
    p, d, q = order
    P, D, Q, s = seasonal_order

    candidates = []

    candidates.append((
        f"PRODUCTION {order}x{seasonal_order} {trend}, exog={exog_cols or 'none'}",
        order, seasonal_order, trend, list(exog_cols),
    ))

    alt_trend = "n" if trend == "c" else "c"
    candidates.append((f"ALT: trend flip -> {alt_trend!r}", order, seasonal_order, alt_trend, list(exog_cols)))

    if exog_cols:
        candidates.append((f"ALT: no exog (drop {exog_cols})", order, seasonal_order, trend, []))
        if len(exog_cols) > 1:
            for col in exog_cols:
                candidates.append((f"EXOG ISOLATION: {col} only", order, seasonal_order, trend, [col]))

    if D == 1:
        candidates.append((
            "ALT: remove seasonal differencing (D=1->0)",
            order, (P, 0, Q, s), trend, list(exog_cols),
        ))
    elif D == 0 and P == 0 and Q == 0:
        candidates.append((
            "ALT: add seasonal MA (D=0,Q=0 -> Q=1)",
            order, (0, 0, 1, s), trend, list(exog_cols),
        ))

    if d == 0:
        candidates.append((
            "REGRESSION CHECK: d=0->1",
            (p, 1, q), seasonal_order, "n", list(exog_cols),
        ))
    elif d == 1 and D == 1:
        candidates.append((
            "ALT: remove regular differencing (d=1->0)",
            (p, 0, q), seasonal_order, "c", list(exog_cols),
        ))

    if (p, q) != (1, 0):
        candidates.append((f"ALT: AR(1) only ({1},{d},{0})", (1, d, 0), seasonal_order, trend, list(exog_cols)))
    if (p, q) != (0, 1):
        candidates.append((f"ALT: MA(1) only ({0},{d},{1})", (0, d, 1), seasonal_order, trend, list(exog_cols)))
    if (p, q) != (1, 1):
        candidates.append((f"ALT: ARMA(1,1) ({1},{d},{1})", (1, d, 1), seasonal_order, trend, list(exog_cols)))

    candidates.append((f"BASELINE: white noise (0,{d},0)", (0, d, 0), (0, 0, 0, s), trend, []))

    seen = set()
    deduped = []
    for c in candidates:
        key = (c[1], c[2], c[3], tuple(c[4]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    return deduped


def diagnostic_significance(y, exog, exog_cols):
    """Run OLS significance test for exogenous variables."""
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
    """Check for significant quadratic curvature in the series."""
    print("\n--- [2/5] Quadratic trend check ---")
    t = np.arange(len(y), dtype=float)
    X = sm.add_constant(np.column_stack([t, t ** 2]))
    model = sm.OLS(y, X).fit()

    coef_t2, pval_t2 = model.params[2], model.pvalues[2]
    flag = "FLAG: significant curvature" if pval_t2 < DIAGNOSTIC_ALPHA else "OK: no significant curvature"
    print(f"  t^2 coef={coef_t2:.6f}  p={pval_t2:.4f}  [{flag}]")


def diagnostic_stationarity(y):
    """Run stationarity tests (ADF and KPSS) on levels and differences."""
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

    if len(y) > SEASONAL_PERIOD:
        _run(f"seasonal diff ({SEASONAL_PERIOD})", y[SEASONAL_PERIOD:] - y[:-SEASONAL_PERIOD])


def diagnostic_order_selection(y, exog, seasonal_period):
    """Run AICc grid search to suggest optimal ARIMA orders."""
    print("\n--- [4/5] Order selection (AICc grid search, suggestions only) ---")

    scale = np.nanmax(np.abs(y))
    y_scaled = y / scale

    combos = itertools.product(
        ORDER_GRID_P, ORDER_GRID_D, ORDER_GRID_Q,
        ORDER_GRID_SEASONAL_P, ORDER_GRID_SEASONAL_D, ORDER_GRID_SEASONAL_Q,
    )

    scored = []
    n_tried, n_failed = 0, 0

    for p, d, q, P, D, Q in combos:
        order = (p, d, q)
        seasonal_order = (P, D, Q, seasonal_period)
        trend = "c" if (d == 0 and D == 0) else None
        n_tried += 1

        try:
            model = sm.tsa.statespace.SARIMAX(
                y_scaled, exog=exog, order=order, seasonal_order=seasonal_order,
                trend=trend, enforce_stationarity=True, enforce_invertibility=True,
            )
            res = model.fit(disp=False)

            if np.isfinite(res.aicc):
                scored.append((res.aicc, order, seasonal_order, trend))
        except Exception:
            n_failed += 1
            continue

    scored.sort(key=lambda row: row[0])

    print(f"  Fitted {n_tried - n_failed}/{n_tried} grid combinations successfully.")

    for aicc, order, seasonal_order, trend in scored[:ORDER_SELECTION_TOP_N]:
        print(f"  AICc={aicc:>10.2f}  order={order}  seasonal_order={seasonal_order}  trend={trend!r}")

    if not scored:
        print("  No grid combination converged.")


def diagnostic_instability(sub, candidates):
    """Check AR/MA roots for each candidate model to detect instability."""
    print("\n--- [5/5] Instability check (AR/MA roots, full-history fit per candidate) ---")

    y_full = sub["value"].values.astype(float)
    scale = np.nanmax(np.abs(y_full))
    y_scaled = y_full / scale

    for label, order, seasonal_order, trend, exog_cols in candidates:
        exog_full = sub[list(exog_cols)].values.astype(float) if exog_cols else None

        try:
            model = sm.tsa.statespace.SARIMAX(
                y_scaled, exog=exog_full, order=order, seasonal_order=seasonal_order,
                trend=trend, enforce_stationarity=True, enforce_invertibility=True,
            )
            res = model.fit(disp=False)
        except Exception as exc:
            print(f"  {label:<55} FIT FAILED ({type(exc).__name__})")
            continue

        ar_roots = np.abs(res.arroots) if len(res.arroots) else np.array([])
        ma_roots = np.abs(res.maroots) if len(res.maroots) else np.array([])

        flags = []
        if ar_roots.size and ar_roots.min() < INSTABILITY_MARGIN:
            flags.append(f"near-unit-root AR (min|root|={ar_roots.min():.3f})")
        if ma_roots.size and ma_roots.min() < INSTABILITY_MARGIN:
            flags.append(f"near-non-invertible MA (min|root|={ma_roots.min():.3f})")

        status = "; ".join(flags) if flags else "OK"
        print(f"  {label:<55} [{status}]")


def run_diagnostic_sequence(sub, candidates, diagnostic_exog_cols):
    """Run the full diagnostic sequence for a metric."""
    print("\n" + "=" * 90)
    print("DIAGNOSTIC SEQUENCE (screening report - nothing here is auto-applied)")
    print("=" * 90)

    y_full = sub["value"].values.astype(float)
    exog_full = sub[list(diagnostic_exog_cols)].values.astype(float) if diagnostic_exog_cols else None

    diagnostic_significance(y_full, exog_full, diagnostic_exog_cols)
    diagnostic_quadratic_trend(y_full)
    diagnostic_stationarity(y_full)
    diagnostic_order_selection(y_full, exog_full, SEASONAL_PERIOD)
    diagnostic_instability(sub, candidates)

    print("\nReminder: this report is informational. Per the flag-don't-override rule,")
    print("nothing above is auto-applied to exog_config.py - edit it by hand if warranted.")


def fit_and_forecast(y_scaled, exog, order, seasonal_order, trend, horizon):
    """Fit SARIMAX model and generate a one-step forecast."""
    try:
        model = sm.tsa.statespace.SARIMAX(
            y_scaled, exog=exog, order=order, seasonal_order=seasonal_order,
            trend=trend, enforce_stationarity=True, enforce_invertibility=True,
        )
        res = model.fit(disp=False)

        if exog is not None:
            last_val = exog[-1, :]
            last_diff = exog[-1, :] - exog[-2, :]
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


def run_backtest(sub, candidates, horizons, min_train_size):
    """Run rolling-origin backtest for all candidates."""
    y_full = sub["value"].values.astype(float)
    n = len(y_full)
    scale = np.nanmax(np.abs(y_full))

    exog_cache = {}
    for label, order, seasonal_order, trend, exog_cols in candidates:
        key = tuple(exog_cols)
        if key not in exog_cache:
            exog_cache[key] = sub[list(exog_cols)].values.astype(float) if exog_cols else None

    results = {label: {h: [] for h in horizons} for label, *_ in candidates}

    for t in range(min_train_size, n - 1):
        y_train = y_full[:t + 1] / scale
        last_known = y_full[t]

        for h in horizons:
            target_idx = t + h
            if target_idx >= n:
                continue

            actual = y_full[target_idx]

            for label, order, seasonal_order, trend, exog_cols in candidates:
                exog_full = exog_cache[tuple(exog_cols)]
                exog_train = exog_full[:t + 1] if exog_full is not None else None

                out = fit_and_forecast(y_train, exog_train, order, seasonal_order, trend, h)
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


def compute_metrics(rows):
    """
    Compute performance metrics from backtest rows.

    Returns
    -------
    dict or None
        Dictionary with n, rmse, mae, bias, dir_acc, ci_coverage
    """
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

    return {
        "n": len(rows),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "dir_acc": dir_acc,
        "ci_coverage": ci_coverage,
    }


def score_results(results, horizons, candidates):
    """Print backtest results table."""
    print("\n" + "=" * 90)
    print("BACKTEST RESULTS (rolling-origin, expanding window)")
    print("=" * 90)

    for h in horizons:
        print(f"\n--- Horizon = {h} quarter(s) ahead ---")
        print(f"{'Spec':<55}{'n':>4}{'RMSE':>9}{'MAE':>9}{'Bias':>9}{'DirAcc':>8}{'CICov':>8}")

        for label, *_ in candidates:
            rows = results[label][h]
            metrics = compute_metrics(rows)

            if metrics is None:
                print(f"{label:<55}  (no folds completed)")
                continue

            print(
                f"{label:<55}{metrics['n']:>4}{metrics['rmse']:>9.4f}{metrics['mae']:>9.4f}"
                f"{metrics['bias']:>+9.4f}{metrics['dir_acc']:>8.2f}{metrics['ci_coverage']:>8.2f}"
            )

    print("\nNotes:")
    print("  Bias > 0 = over-forecasting; < 0 = under-forecasting.")
    print("  DirAcc = fraction of folds where forecast correctly called up/down vs last known value.")
    print("  CICov  = fraction of folds where actual fell inside the model's own 95% CI (target ~0.95).")


def print_bias_summary(results, candidates):
    """Print bias summary for production and near-neighbour variants."""
    matched = [label for label, *_ in candidates if any(m in label for m in BIAS_SUMMARY_LABEL_MATCHES)]

    if not matched:
        return

    horizons_present = sorted({h for label in matched for h in results[label].keys()})
    bias_horizons = horizons_present[-2:] if len(horizons_present) >= 2 else horizons_present

    print("\n" + "=" * 90)
    print("BIAS SUMMARY - production + near-neighbour variants, long horizons")
    print(f"(horizons shown: {bias_horizons})")
    print("=" * 90)

    header = f"{'Spec':<55}" + "".join(f"{('h=' + str(h)):>16}" for h in bias_horizons)
    print(header)

    for label in matched:
        row_str = f"{label:<55}"

        for h in bias_horizons:
            if h not in results[label]:
                row_str += f"{'not tested':>16}"
                continue

            metrics = compute_metrics(results[label][h])
            if metrics is None:
                row_str += f"{'no folds':>16}"
                continue

            direction = "OVER" if metrics["bias"] > 0 else ("UNDER" if metrics["bias"] < 0 else "FLAT")
            cell = f"{metrics['bias']:+.4f}({direction})"
            row_str += f"{cell:>16}"

        print(row_str)


def residual_diagnostics(sub, metric_key):
    """Run residual diagnostics on the production model."""
    print("\n" + "=" * 90)
    print("RESIDUAL DIAGNOSTICS (production spec, full-history fit)")
    print("=" * 90)

    cfg = ec.MODEL_CONFIG[metric_key]
    exog_cols = list(ec.EXOG_CONFIG.get(metric_key, []))

    y = sub["value"].values.astype(float)
    scale = np.nanmax(np.abs(y))
    y_scaled = y / scale
    exog = sub[exog_cols].values.astype(float) if exog_cols else None

    verdict = {"ljung_box": "N/A", "jarque_bera": "N/A", "arch": "N/A", "fit_status": "OK"}

    try:
        model = sm.tsa.statespace.SARIMAX(
            y_scaled, exog=exog, order=cfg["order"], seasonal_order=cfg["seasonal_order"],
            trend=cfg["trend"], enforce_stationarity=True, enforce_invertibility=True,
        )
        res = model.fit(disp=False)
    except Exception as exc:
        print(f"  FIT FAILED on production spec: {type(exc).__name__}: {exc}")
        verdict["fit_status"] = "FIT FAILED"
        return verdict

    resid = np.asarray(res.resid, dtype=float)
    resid = resid[~np.isnan(resid)]
    n = len(resid)

    lb_lags = [lag for lag in (4, 8) if lag < n]
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

    if n >= 8:
        jb_stat, jb_p, skew, kurt = sm.stats.stattools.jarque_bera(resid)
        flag = "OK" if jb_p >= DIAGNOSTIC_ALPHA else "FLAG: residuals non-normal"

        print(f"\n  Jarque-Bera (H0: residuals normal): stat={jb_stat:.3f}  p={jb_p:.4f}  "
              f"skew={skew:.3f}  kurtosis={kurt:.3f}  [{flag}]")

        verdict["jarque_bera"] = "OK" if jb_p >= DIAGNOSTIC_ALPHA else "FLAG"
    else:
        print("\n  Jarque-Bera: too few residuals to test - skipped.")

    if n >= 12:
        try:
            nlags = max(1, min(4, n // 3))
            arch_stat, arch_p, f_stat, f_p = het_arch(resid, nlags=nlags)
            flag = "OK" if arch_p >= DIAGNOSTIC_ALPHA else "FLAG: heteroskedasticity (ARCH effects)"

            print(f"\n  ARCH LM (H0: no heteroskedasticity), nlags={nlags}: "
                  f"stat={arch_stat:.3f}  p={arch_p:.4f}  [{flag}]")

            verdict["arch"] = "OK" if arch_p >= DIAGNOSTIC_ALPHA else "FLAG"
        except Exception:
            print(f"\n  ARCH LM: test failed - skipped.")
    else:
        print("\n  ARCH LM: too few residuals to test - skipped.")

    print("\n  Note: with quarterly NHS series this short, low power on all three tests")
    print("  is expected - treat a single borderline p-value as a flag to note, not a verdict.")

    return verdict


def process_metric(metric_key):
    """Run full diagnostic and backtest sequence for a single metric."""
    print("\n\n" + "#" * 90)
    print(f"# METRIC: {metric_key}")
    print(f"# Series label: {ec.METRIC_NAMES[metric_key]}")
    print("#" * 90)

    sub = load_series(metric_key)
    print(f"Loaded {len(sub)} observations.")
    print(f"Range: {sub['quarter'].iloc[0].date()} to {sub['quarter'].iloc[-1].date()}")

    horizons, min_train_size = resolve_horizons_and_min_train(metric_key, sub)
    candidates = build_candidates(metric_key)
    diagnostic_exog_cols = list(ec.EXOG_CONFIG.get(metric_key, []))

    print(f"Horizons tested: {horizons} | Min training window: {min_train_size}")
    print(f"Candidates ({len(candidates)}): {[c[0] for c in candidates]}")

    run_diagnostic_sequence(sub, candidates, diagnostic_exog_cols)

    results = run_backtest(sub, candidates, horizons, min_train_size)
    score_results(results, horizons, candidates)
    print_bias_summary(results, candidates)

    resid_verdict = residual_diagnostics(sub, metric_key)

    prod_label = candidates[0][0]
    longest_h = max(horizons)
    prod_metrics = compute_metrics(results[prod_label][longest_h])

    summary_row = {
        "metric_key": metric_key,
        "n_obs": len(sub),
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

    return summary_row, results, candidates, horizons


def main():
    """Main entry point - parses arguments and runs the diagnostics."""
    parser = argparse.ArgumentParser(
        description="Run backtest diagnostics for a specific metric.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stress_test.py "A&E attendances (flow)"
  python stress_test.py "RTT waiting list (level)"
  python stress_test.py "GP total appointments (flow)"
  python stress_test.py --list
        """
    )

    parser.add_argument(
        "metric",
        nargs="?",
        default="A&E attendances (flow)",
        help='Name of the metric to test (default: "A&E attendances (flow)")'
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available metrics and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable metrics:")
        for key in ec.METRIC_NAMES.keys():
            print(f"  - {key}")
        print(f"\nTotal: {len(ec.METRIC_NAMES)} metrics")
        return

    if args.metric not in ec.METRIC_NAMES:
        print(f"\nERROR: Metric '{args.metric}' not found in METRIC_NAMES.")
        print("\nAvailable metrics:")
        for key in ec.METRIC_NAMES.keys():
            print(f"  - {key}")
        print(f"\nTotal: {len(ec.METRIC_NAMES)} metrics")
        sys.exit(1)

    metric_key = args.metric
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    safe_name = metric_key.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    log_path = os.path.join(REPORT_DIR, f"backtest_{safe_name}_log_{timestamp}.txt")
    summary_path = os.path.join(REPORT_DIR, f"backtest_{safe_name}_summary_{timestamp}.csv")

    sys.stdout = Tee(log_path)

    print("=" * 90)
    print("INDIVIDUAL METRIC BACKTEST")
    print("=" * 90)
    print(f"Run started: {datetime.now().isoformat()}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Metric: {metric_key}")
    print("=" * 90)

    try:
        summary_row, results, candidates, horizons = process_metric(metric_key)

        print("\n\n" + "=" * 90)
        print("COMPACT SUMMARY")
        print("=" * 90)
        print(f"Metric: {metric_key}")
        print(f"Observations: {summary_row['n_obs']}")
        print(f"Longest horizon tested: {summary_row['longest_horizon_tested']}")
        print(f"RMSE at longest horizon: {summary_row['rmse']:.4f}")
        print(f"MAE at longest horizon: {summary_row['mae']:.4f}")
        print(f"Bias at longest horizon: {summary_row['bias']:.4f}")
        print(f"Direction Accuracy: {summary_row['dir_acc']:.2f}")
        print(f"CI Coverage: {summary_row['ci_coverage']:.2f}")
        print(f"Ljung-Box: {summary_row['ljung_box']}")
        print(f"Jarque-Bera: {summary_row['jarque_bera']}")
        print(f"ARCH: {summary_row['arch_heteroskedasticity']}")

        pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

    except Exception as exc:
        print(f"\n!!! METRIC '{metric_key}' FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

    print(f"\nFull log saved to: {log_path}")
    print("=" * 90)
    print("Run completed.")


if __name__ == "__main__":
    main()
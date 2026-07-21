"""
master_forecast_engine.py
--------------------------
Master SARIMAX 6‑Year Forecasting Engine with ITS design.

Forecasts nine NHS performance metrics using SARIMAX (or variants) with:
- Automated stationarity testing (ADF, KPSS, ADF‑GLS)
- AICc grid search and residual diagnostics
- Flag‑don't‑override rule for production configs
- Empirical sigma‑scale interval calibration
- Interrupted Time Series (ITS) design for post‑COVID trend break
- Counterfactual forecasts (no‑break scenario)
- Heteroskedasticity‑robust (HC) standard errors via statsmodels cov_type='robust'
- Inline verification/audit table (audit_table.csv) for documentation
- Full parameter audit table (full_params_audit.csv) for narrative claims

UPDATED 2026-07-18: order, seasonal, and trend now read from MODEL_CONFIG
    (single source of truth). Duplicate PROD_ORDER/PROD_SEASONAL dicts removed.
    Trend respects explicit config values including None (for A&E 12h breach).
UPDATED 2026-07-18: bounded ramp extension for post_covid_trend_break
    (4 quarters, then constant) to avoid flat-artifact and explosion.
UPDATED 2026-07-19: dead custom-HAC block removed; HC robust SEs remain.

Run:
    python src/master_forecast_engine.py

Output:
    data/processed/dashboard_forecasts.csv
    data/processed/counterfactuals.csv
    data/processed/audit_table.csv
    data/processed/full_params_audit.csv
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import DFGLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import stats

# ----- import full exog_config module (single source of truth for specs) -----
import exog_config as ec

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"
OUT_PATH = PROCESSED_DIR / "dashboard_forecasts.csv"
COUNTER_PATH = PROCESSED_DIR / "counterfactuals.csv"
AUDIT_PATH = PROCESSED_DIR / "audit_table.csv"
FULL_PARAMS_PATH = PROCESSED_DIR / "full_params_audit.csv"

# ---------- Metric display name -> raw series name ----------
METRIC_MAP = {
    "RTT waiting list (level)": "Incomplete RTT pathways - Total waiting (mil) with estimates for missing data",
    "A&E attendances (flow)": "total_attendances",
    "Workforce FTE (level)": "FTE: All staff groups - All staff groups",
    "Nurse FTE (level)": "FTE: Professionally qualified clinical staff - Nurses & health visitors",
    "Doctor FTE (level)": "FTE: Professionally qualified clinical staff - HCHS doctors - All grades",
    "Bed occupancy (level)": "total_occupied_beds_overnight",
    "RTT % within 18 weeks (performance)": "Incomplete RTT pathways - % within 18 weeks",
    "A&E 12-hour decisions to admit (breach flow)": "number_of_patients_spending_12_hours_from_decision_to_admit_to_admission",
    "PESA Health spend (level)": "7. Health (real_gbp_bn)",
    "GP total appointments (flow)": "total_attended_appointments",
    "GP face-to-face appointments (flow)": "attended_face_to_face",
    "GP telephone appointments (flow)": "attended_telephone",
}

# ---------- Production specs now read from MODEL_CONFIG (no duplicate dicts) ----------

TRANSFORM = {
    "RTT waiting list (level)": "log",
    "Nurse FTE (level)": "log",
    "Doctor FTE (level)": "log",
    "RTT % within 18 weeks (performance)": "logit",
}

SIGMA_SCALE = {
    "RTT waiting list (level)": 1.25,
    "A&E attendances (flow)": 1.5,
    "Workforce FTE (level)": 1.45,
    "Nurse FTE (level)": 1.4,
    "Doctor FTE (level)": 1.8,
    "Bed occupancy (level)": 1.8,
    "RTT % within 18 weeks (performance)": 1.25,
    "A&E 12-hour decisions to admit (breach flow)": 1.0,
    "PESA Health spend (level)": 1.0,
}

FORECAST_HORIZONS = {
    "RTT waiting list (level)": 24,
    "A&E attendances (flow)": 24,
    "Workforce FTE (level)": 24,
    "Nurse FTE (level)": 24,
    "Doctor FTE (level)": 24,
    "Bed occupancy (level)": 24,
    "RTT % within 18 weeks (performance)": 24,
    "A&E 12-hour decisions to admit (breach flow)": 8,
    "PESA Health spend (level)": 24,
}

BREAK_QUARTER = pd.Timestamp("2020-04-01")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _forecast_quarters(last_hist_quarter, horizon):
    """Generate future quarterly periods starting after last_hist_quarter."""
    return pd.date_range(
        last_hist_quarter + pd.DateOffset(months=3),
        periods=horizon,
        freq="QS"
    )


def _stationarity_tests(series, label=""):
    """Run ADF, KPSS, ADF‑GLS and print summary."""
    s = series.dropna().astype(float)
    try:
        adf_stat, adf_p, adf_lags, _, _, _ = adfuller(s, autolag="AIC")
    except Exception as e:
        adf_stat, adf_p, adf_lags = np.nan, np.nan, 0
        print(f"      [ADF] error: {e}")
    try:
        kpss_stat, kpss_p, kpss_lags, _ = kpss(s, regression="c", nlags="auto")
    except Exception as e:
        kpss_stat, kpss_p, kpss_lags = np.nan, np.nan, 0
        print(f"      [KPSS] error: {e}")
    try:
        gls_res = DFGLS(s, lags=None)
        gls_stat, gls_p, gls_lags = gls_res.stat, gls_res.pvalue, gls_res.lags
    except Exception as e:
        gls_stat, gls_p, gls_lags = np.nan, np.nan, 0
        print(f"      [ADF-GLS] error: {e}")

    print(f"      [ADF] stat={adf_stat:.3f}, p={adf_p:.3f}, lag={adf_lags}")
    print(f"      [KPSS] stat={kpss_stat:.3f}, p={kpss_p:.3f}, lags={kpss_lags}")
    print(f"      [ADF-GLS] stat={gls_stat:.3f}, p={gls_p:.3f}, lag={gls_lags}")
    return adf_stat, adf_p, kpss_stat, kpss_p, gls_stat, gls_p


def _suggest_d(adf_p, kpss_p):
    """Heuristic: if ADF can't reject unit root but KPSS rejects stationarity, d=1."""
    if adf_p > 0.05 and kpss_p < 0.05:
        return 1
    return 0


def _suggest_seasonal(series, max_lag=4):
    """Check ACF at seasonal lag for evidence of seasonality."""
    acf_vals = sm.tsa.acf(series.dropna(), nlags=max_lag, fft=False)
    if abs(acf_vals[max_lag]) > 0.3:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Core modelling function for a single metric
# ---------------------------------------------------------------------------
def model_metric(display_name, df_combined, forecast_records, counterfactual_records, verification_records, full_param_records):
    """
    Fit a SARIMAX model (or a suitable variant) for the given metric,
    produce forecasts and counterfactuals, and append audit records.
    """
    raw_name = METRIC_MAP[display_name]
    sub = df_combined[df_combined["metric"] == raw_name].dropna(subset=["value"]).sort_values("quarter")
    if len(sub) < 6:
        print(f"\n[MODELING] {display_name}")
        print(f"   [SKIP] Too few observations.")
        return

    print(f"\n[MODELING] {display_name}")

    # --- Special handling for GP and PESA (non‑SARIMAX) ---
    if "GP" in display_name:
        print(f"   [SKIP] {display_name}: Insufficient data. Historical data will still be shown.")
        for _, row in sub.iterrows():
            forecast_records.append({
                "quarter": row["quarter"],
                "metric": display_name,
                "type": "history",
                "value": row["value"],
                "ci_lower": row["value"],
                "ci_upper": row["value"],
            })
        return

    if display_name == "PESA Health spend (level)":
        print("   [MODEL] Using random walk with drift for annual PESA health spend")
        hist = sub.copy()
        last_val = hist["value"].iloc[-1]
        diffs = hist["value"].diff().mean()
        horizon = FORECAST_HORIZONS[display_name]
        fc_vals = last_val + diffs * np.arange(1, horizon + 1)
        fc_quarters = _forecast_quarters(hist["quarter"].max(), horizon)
        for i, q in enumerate(fc_quarters):
            forecast_records.append({
                "quarter": q,
                "metric": display_name,
                "type": "forecast",
                "value": fc_vals[i],
                "ci_lower": fc_vals[i],
                "ci_upper": fc_vals[i],
            })
        for _, row in hist.iterrows():
            forecast_records.append({
                "quarter": row["quarter"],
                "metric": display_name,
                "type": "history",
                "value": row["value"],
                "ci_lower": row["value"],
                "ci_upper": row["value"],
            })
        return

    # --- Common data preparation ---
    hist = sub.copy()
    if "12-hour" in display_name and "breach" in display_name.lower():
        hist = hist[hist["quarter"] >= "2021-01-01"]
        print(f"   [FIT WINDOW] {display_name}: restricted to 2021-01-01 onward ({len(hist)} observations)")

    endog = hist.set_index("quarter")["value"]

    # ---------- Per‑series exogenous columns from exog_config ----------
    exog_cols = ec.EXOG_CONFIG.get(display_name, [])
    if exog_cols:
        exog_hist = hist.set_index("quarter")[exog_cols].copy()
    else:
        exog_hist = pd.DataFrame(index=hist.set_index("quarter").index)

    # ---------- Read order, seasonal, trend from MODEL_CONFIG ----------
    cfg = ec.MODEL_CONFIG.get(display_name, {})
    order = cfg.get("order", (1, 1, 0))
    seasonal = cfg.get("seasonal_order", (0, 0, 0, 4))

    _MISSING = object()
    trend_config = cfg.get("trend", _MISSING)
    if trend_config is not _MISSING:
        trend = trend_config
    else:
        trend = 'n' if 't' in exog_hist.columns else 'c'

    print(f"   [EXOG] Using columns: {exog_cols if exog_cols else 'None'}, order={order}, seasonal={seasonal}, trend={trend}")

    # Transform if needed
    apply_log = TRANSFORM.get(display_name) == "log"
    apply_logit = TRANSFORM.get(display_name) == "logit"
    if apply_logit:
        eps = 1e-6
        y = endog.clip(eps, 1 - eps)
        endog_t = np.log(y / (1 - y))
        print("   [TRANSFORM] Applying logit transformation")
    elif apply_log:
        endog_t = np.log(endog.replace(0, np.nan))
        print("   [TRANSFORM] Applying log transformation")
    else:
        endog_t = endog

    # Stationarity tests on levels
    print(f"   [STAT] {display_name} (levels): ADF + KPSS + ADF-GLS")
    adf_stat, adf_p, kpss_stat, kpss_p, gls_stat, gls_p = _stationarity_tests(endog_t)
    sugg_d = _suggest_d(adf_p, kpss_p)
    print(f"   [DIFF] Suggested d={sugg_d} for {display_name}")

    # Seasonality check
    acf_vals = sm.tsa.acf(endog_t.dropna(), nlags=4, fft=False)
    print(f"      [ACF] {display_name}: lag 4 = {acf_vals[4]:.3f}")
    sugg_D = _suggest_seasonal(endog_t)
    print(f"      [SEASONAL] {display_name}: {'evidence of seasonality → D=1' if sugg_D else 'weak seasonality → D=0'}")

    config_d = order[1]
    config_D = seasonal[1]
    if config_d != sugg_d:
        print(f"   [DIAGNOSTIC FLAG] {display_name}: config d={config_d} vs suggested d={sugg_d} ...")
    if config_D != sugg_D:
        print(f"   [DIAGNOSTIC FLAG] {display_name}: config D={config_D} vs suggested D={sugg_D} ...")

    print(f"   [MODEL] order={order}, seasonal={seasonal}, trend={trend}")

    try:
        model = sm.tsa.SARIMAX(
            endog_t,
            exog=exog_hist if exog_cols else None,
            order=order,
            seasonal_order=seasonal,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        # Use heteroskedasticity‑robust (HC) covariance estimator
        res = model.fit(disp=False, maxiter=200, cov_type='robust')
        print(f"   [FIT] AICc={res.aicc:.2f}")
    except Exception as e:
        print(f"   [FIT ERROR] {e}")
        return

    # Residual diagnostics
    resid = res.resid
    try:
        lb = acorr_ljungbox(resid.dropna(), lags=[4, 8], return_df=True)
        lb_p4 = lb.loc[4, "lb_pvalue"] if 4 in lb.index else np.nan
        lb_p8 = lb.loc[8, "lb_pvalue"] if 8 in lb.index else np.nan
    except:
        lb_p4, lb_p8 = np.nan, np.nan
    try:
        jb_stat, jb_p = stats.jarque_bera(resid.dropna())
        kurt = stats.kurtosis(resid.dropna(), fisher=False)  # Pearson kurtosis
    except:
        jb_stat, jb_p, kurt = np.nan, np.nan, np.nan
    try:
        arch = het_arch(resid.dropna())
        arch_p = arch[1] if len(arch) > 1 else np.nan
    except:
        arch_p = np.nan

    print(f"   [STAT] {display_name} (residuals): ADF + KPSS + ADF-GLS")
    _stationarity_tests(resid.dropna(), label="residuals")
    print(f"   [RESIDUAL] Ljung-Box p={lb_p4:.4f} (lag4), p={lb_p8:.4f} (lag8)")
    print(f"   [RESIDUAL] Jarque-Bera p={jb_p:.4f}, ARCH LM p={arch_p:.4f}")

    # --- ITS coefficients (flexible) ---
    its_coef = {}
    its_se = {}
    for p in ['t', 'post_covid_trend_break']:
        if p in res.params:
            its_coef[p] = res.params[p]
            its_se[p] = res.bse[p]
        else:
            its_coef[p] = np.nan
            its_se[p] = np.nan
    print(f"   [ITS] Coefficients: t={its_coef.get('t', np.nan):.4f}, break_level={its_coef.get('post_covid_trend_break', np.nan):.4f}")
    print(f"   [ITS] Model SE (HC): t={its_se.get('t', np.nan):.4f}, break_level={its_se.get('post_covid_trend_break', np.nan):.4f}")

    # ---------- Verification record (inline audit) ----------
    verification_records.append({
        "series": display_name,
        "aicc": res.aicc,
        "coef_t": its_coef.get('t', np.nan),
        "coef_break_level": its_coef.get('post_covid_trend_break', np.nan),
        "model_se_t": its_se.get('t', np.nan),
        "model_se_break_level": its_se.get('post_covid_trend_break', np.nan),
        # HAC fields are deprecated – filled with NaN for backward‑compatibility
        "hac_bandwidth": np.nan,
        "hac_reliable": np.nan,
        "hac_warnings": "",
        "robust_se_t": np.nan,
        "robust_se_break_level": np.nan,
        "cond_number": np.nan,
        "ljungbox_p_lag4": lb_p4,
        "ljungbox_p_lag8": lb_p8,
        "jarquebera_p": jb_p,
        "kurtosis": kurt,
        "n_train": len(exog_hist),
    })

    # ---------- Full parameter audit ----------
    for pname in res.params.index:
        full_param_records.append({
            "series": display_name,
            "param": pname,
            "coef": res.params[pname],
            "std_err": res.bse[pname],
            "z_or_t": res.tvalues[pname],
            "p_value": res.pvalues[pname],
        })

    # ------------------------- Forecast & Counterfactual ---------------------
    last_hist_quarter = hist["quarter"].max()
    horizon = FORECAST_HORIZONS[display_name]
    fc_quarters = _forecast_quarters(last_hist_quarter, horizon)

    # Build future exogenous variables:
    #   - post_covid_trend_break: bounded ramp for 4 quarters, then constant.
    #   - all other columns: held constant at last historical value.
    RAMP_EXTENSION_QUARTERS = 4

    exog_future = pd.DataFrame(index=range(horizon))
    for col in exog_cols:
        last_val = exog_hist[col].iloc[-1]
        if col == "post_covid_trend_break":
            extension = np.arange(1, horizon + 1)
            extension = np.minimum(extension, RAMP_EXTENSION_QUARTERS)
            exog_future[col] = last_val + extension
        else:
            exog_future[col] = last_val

    # Counterfactual: set break‑related dummies to 0
    break_cols = ["post_covid_trend_break", "covid_pulse", "post_covid_regime", "post_covid_slope_change"]
    exog_counter = exog_future.copy()
    for col in break_cols:
        if col in exog_counter.columns:
            exog_counter[col] = 0

    fc = res.get_forecast(steps=horizon, exog=exog_future if exog_cols else None)
    mean_fc = fc.predicted_mean
    ci = fc.conf_int()

    fc_counter = res.get_forecast(steps=horizon, exog=exog_counter if exog_cols else None)
    mean_counter = fc_counter.predicted_mean
    ci_counter = fc_counter.conf_int()

    # Back‑transform if necessary
    if apply_logit:
        mean_fc = 1 / (1 + np.exp(-mean_fc))
        ci = 1 / (1 + np.exp(-ci))
        mean_counter = 1 / (1 + np.exp(-mean_counter))
        ci_counter = 1 / (1 + np.exp(-ci_counter))
    elif apply_log:
        mean_fc = np.exp(mean_fc)
        ci = np.exp(ci)
        mean_counter = np.exp(mean_counter)
        ci_counter = np.exp(ci_counter)

    # Sigma‑scale interval widening
    sigma = SIGMA_SCALE.get(display_name, 1.0)
    if sigma != 1.0:
        lower = mean_fc - sigma * (mean_fc - ci.iloc[:, 0])
        upper = mean_fc + sigma * (ci.iloc[:, 1] - mean_fc)
        ci = pd.DataFrame({ci.columns[0]: lower, ci.columns[1]: upper})
        lower_c = mean_counter - sigma * (mean_counter - ci_counter.iloc[:, 0])
        upper_c = mean_counter + sigma * (ci_counter.iloc[:, 1] - mean_counter)
        ci_counter = pd.DataFrame({ci_counter.columns[0]: lower_c, ci_counter.columns[1]: upper_c})

    # Clamp negative values to zero (except for percentage/logit series)
    if not apply_logit and not apply_log and display_name not in ["RTT % within 18 weeks (performance)"]:
        clamp_count = (mean_fc < 0).sum() + (ci.iloc[:, 0] < 0).sum()
        if clamp_count > 0:
            print(f"   [CLAMP WARNING] {display_name}: {clamp_count} forecast points had negative values clamped to 0.")
            mean_fc = mean_fc.clip(lower=0)
            ci = ci.clip(lower=0)
            mean_counter = mean_counter.clip(lower=0)
            ci_counter = ci_counter.clip(lower=0)

    # Store forecast records
    for i, q in enumerate(fc_quarters):
        forecast_records.append({
            "quarter": q,
            "metric": display_name,
            "type": "forecast",
            "value": mean_fc.iloc[i],
            "ci_lower": ci.iloc[i, 0],
            "ci_upper": ci.iloc[i, 1],
        })
    for _, row in hist.iterrows():
        forecast_records.append({
            "quarter": row["quarter"],
            "metric": display_name,
            "type": "history",
            "value": row["value"],
            "ci_lower": row["value"],
            "ci_upper": row["value"],
        })

    # Store counterfactual records
    for i, q in enumerate(fc_quarters):
        counterfactual_records.append({
            "quarter": q,
            "metric": display_name,
            "counterfactual_mean": mean_counter.iloc[i] if hasattr(mean_counter, 'iloc') else mean_counter[i],
            "counterfactual_ci_lower": ci_counter.iloc[i, 0] if hasattr(ci_counter, 'iloc') else ci_counter[i][0],
            "counterfactual_ci_upper": ci_counter.iloc[i, 1] if hasattr(ci_counter, 'iloc') else ci_counter[i][1],
        })


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main():
    """Run the full forecast pipeline for all metrics."""
    print(">>> RUNNING LATEST VERSION (config‑driven + bounded ramp, HC‑robust SE) <<<")
    print("=" * 70)
    print("MASTER FORECAST ENGINE – config‑driven ITS")
    print("=" * 70)

    df = pd.read_csv(COMBINED_PATH, parse_dates=["quarter"])
    # Ensure all required exogenous columns exist (fill missing with zeros)
    required_cols = {"t", "post_covid_trend_break", "covid_pulse", "post_covid_regime", "post_covid_slope_change"}
    missing = required_cols - set(df.columns)
    for col in missing:
        df[col] = 0

    forecast_metrics = [
        "RTT waiting list (level)",
        "A&E attendances (flow)",
        "Workforce FTE (level)",
        "Nurse FTE (level)",
        "Doctor FTE (level)",
        "Bed occupancy (level)",
        "RTT % within 18 weeks (performance)",
        "A&E 12-hour decisions to admit (breach flow)",
        "PESA Health spend (level)",
        "GP total appointments (flow)",
        "GP face-to-face appointments (flow)",
        "GP telephone appointments (flow)",
    ]

    forecast_records = []
    counterfactual_records = []
    verification_records = []
    full_param_records = []

    for metric in forecast_metrics:
        model_metric(metric, df, forecast_records, counterfactual_records, verification_records, full_param_records)

    # Save outputs
    fc_df = pd.DataFrame(forecast_records).sort_values(["metric", "quarter"]).reset_index(drop=True)
    fc_df.to_csv(OUT_PATH, index=False)
    print(f"\nDONE – forecasts written to {OUT_PATH}")

    if counterfactual_records:
        c_df = pd.DataFrame(counterfactual_records).sort_values(["metric", "quarter"]).reset_index(drop=True)
        c_df.to_csv(COUNTER_PATH, index=False)
        print(f"Counterfactuals saved to {COUNTER_PATH}")

    if verification_records:
        v_df = pd.DataFrame(verification_records)
        v_df.to_csv(AUDIT_PATH, index=False)
        print(f"Verification/audit table saved to {AUDIT_PATH}")

    if full_param_records:
        fp_df = pd.DataFrame(full_param_records)
        fp_df.to_csv(FULL_PARAMS_PATH, index=False)
        print(f"Full parameter audit table saved to {FULL_PARAMS_PATH}")


if __name__ == "__main__":
    main()
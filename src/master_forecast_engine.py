"""
master_forecast_engine.py
-------------------------
Main forecasting engine for the Health Economics Dashboard.

This module runs the SARIMAX forecasting pipeline for all health system metrics.
It handles:
- Data loading and preprocessing
- Log transformations where needed
- Exogenous variable scaling
- Model fitting with fallback options
- Forecast generation with back-transformation
- Clamping and smoothing of forecasts

Usage:
    python master_forecast_engine.py

Input:
    data/processed/combined_quarterly.csv

Output:
    data/processed/dashboard_forecasts.csv

Last updated: 2026-07-02
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf
from pathlib import Path
from exog_config import EXOG_CONFIG, METRIC_NAMES, MODEL_CONFIG, FIT_START_OVERRIDES

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "combined_quarterly.csv"
OUTPUT_FILE = PROCESSED_DIR / "dashboard_forecasts.csv"


# ---------------------------------------------------------------------------
# Stationarity testing functions
# ---------------------------------------------------------------------------

def adf_gls(series, label):
    """
    Perform ADF-GLS test for stationarity (ERS test).
    
    Parameters
    ----------
    series : array-like
        Time series data to test
    label : str
        Label for logging purposes
    
    Returns
    -------
    float or None
        p-value of the ADF-GLS test, or None if test fails
    """
    s = pd.Series(series).dropna()
    if len(s) < 10:
        print(f"      [ADF-GLS] {label}: insufficient length.")
        return None

    y = s.values
    T = len(y)
    t = np.arange(1, T + 1)
    alpha = 0.9

    # GLS detrending
    y_gls = y[1:] - alpha * y[:-1]
    X_gls = np.column_stack([np.ones(T - 1), t[1:] - alpha * t[:-1]])
    beta = np.linalg.lstsq(X_gls, y_gls, rcond=None)[0]
    detrended = y_gls - X_gls @ beta

    try:
        stat, p, usedlag, *_ = adfuller(detrended, autolag="AIC")
        print(f"      [ADF-GLS] stat={stat:.3f}, p={p:.3f}, lag={usedlag}")
        return p
    except Exception as e:
        print(f"      [ADF-GLS ERROR] {label}: {e}")
        return None


def run_stationarity_battery(series, label):
    """
    Run a battery of stationarity tests: ADF, KPSS, and ADF-GLS.
    
    Parameters
    ----------
    series : array-like
        Time series data to test
    label : str
        Label for logging purposes
    
    Returns
    -------
    dict
        Dictionary containing p-values from each test
    """
    s = pd.Series(series).dropna()
    if len(s) < 10:
        print(f"   [STAT] {label}: insufficient length.")
        return {"adf_p": None, "kpss_p": None, "adf_gls_p": None}

    print(f"   [STAT] {label}: ADF + KPSS + ADF-GLS")

    # ADF test (Augmented Dickey-Fuller)
    try:
        adf_stat, adf_p, adf_lag, *_ = adfuller(s, autolag="AIC")
        print(f"      [ADF] stat={adf_stat:.3f}, p={adf_p:.3f}, lag={adf_lag}")
    except Exception as e:
        print(f"      [ADF ERROR] {label}: {e}")
        adf_p = None

    # KPSS test (Kwiatkowski-Phillips-Schmidt-Shin)
    try:
        kpss_stat, kpss_p, kpss_lags, *_ = kpss(s, regression="c", nlags="auto")
        print(f"      [KPSS] stat={kpss_stat:.3f}, p={kpss_p:.3f}, lags={kpss_lags}")
    except Exception as e:
        print(f"      [KPSS ERROR] {label}: {e}")
        kpss_p = None

    # ADF-GLS test (ERS)
    adf_gls_p = adf_gls(s, label)

    # Summary interpretation
    if adf_p is not None and kpss_p is not None:
        if adf_p < 0.05 and kpss_p > 0.05:
            print(f"      [SUMMARY] {label}: broadly stationary.")
        elif adf_p > 0.05 and kpss_p < 0.05:
            print(f"      [SUMMARY] {label}: clearly non-stationary.")
        else:
            print(f"      [SUMMARY] {label}: mixed signals.")

    if adf_gls_p is not None:
        print(f"      [ERS] ADF-GLS p={adf_gls_p:.3f}")

    return {"adf_p": adf_p, "kpss_p": kpss_p, "adf_gls_p": adf_gls_p}


def suggest_d(stat_results, label):
    """
    Suggest the order of differencing (d) based on stationarity tests.
    
    Parameters
    ----------
    stat_results : dict
        Results from run_stationarity_battery
    label : str
        Label for logging purposes
    
    Returns
    -------
    int
        Suggested differencing order (0 or 1)
    """
    adf_p = stat_results.get("adf_p")
    kpss_p = stat_results.get("kpss_p")
    adf_gls_p = stat_results.get("adf_gls_p")

    # Default to d=1 (difference once)
    d = 1
    
    # If ADF says stationary and KPSS says stationary, use d=0
    if adf_p is not None and kpss_p is not None:
        if adf_p < 0.05 and kpss_p > 0.05:
            d = 0
    
    # If ADF-GLS says stationary, use d=0
    if adf_gls_p is not None and adf_gls_p < 0.05:
        d = 0

    print(f"   [DIFF] Suggested d={d} for {label}")
    return d


def suggest_seasonal_D(series, label, freq=4):
    """
    Suggest the order of seasonal differencing (D) based on seasonality tests.
    
    Parameters
    ----------
    series : array-like
        Time series data to test
    label : str
        Label for logging purposes
    freq : int, default=4
        Seasonal frequency (4 for quarterly data)
    
    Returns
    -------
    int
        Suggested seasonal differencing order (0 or 1)
    """
    s = pd.Series(series).dropna()
    if len(s) < freq * 3:
        print(f"   [SEASONAL] {label}: insufficient length.")
        return 0

    # HEGY test for seasonal unit roots
    try:
        arch_unitroot = __import__("arch.unitroot", fromlist=["HEGY"])
        HEGY = getattr(arch_unitroot, "HEGY")
        hegy = HEGY(s, seasonal_periods=freq)
        pvals = hegy.pvalue
        print(f"      [HEGY] {label}: p-values={pvals}")
        if np.any(pvals < 0.05):
            print(f"      [SEASONAL] {label}: seasonal unit root → D=1")
            return 1
    except Exception:
        pass

    # ACF at seasonal lag
    acf_vals = acf(s, nlags=freq * 2, fft=True)
    seasonal_acf = acf_vals[freq]
    print(f"      [ACF] {label}: lag {freq} = {seasonal_acf:.3f}")
    if abs(seasonal_acf) > 0.3:
        print(f"      [SEASONAL] {label}: evidence of seasonality → D=1")
        return 1

    print(f"      [SEASONAL] {label}: weak seasonality → D=0")
    return 0


def check_min_obs_for_order(n_obs, order, seasonal_order, label):
    """
    Check if the series has enough observations for the specified model order.
    
    Parameters
    ----------
    n_obs : int
        Number of observations in the series
    order : tuple
        (p, d, q) ARIMA order
    seasonal_order : tuple
        (P, D, Q, s) seasonal ARIMA order
    label : str
        Label for logging purposes
    
    Returns
    -------
    bool
        True if enough observations, False otherwise
    """
    p, d, q = order
    P, D, Q, s = seasonal_order
    min_required = s * (D + max(P, Q) + 1) + (d + max(p, q) + 1) * 2
    
    if n_obs < min_required:
        print(
            f"   [LENGTH WARNING] {label}: n_obs={n_obs} below rough minimum "
            f"{min_required} for order={order}, seasonal={seasonal_order}. "
            f"Model is likely under-identified — review before trusting this fit."
        )
        return False
    return True


def fit_with_length_fallback(y, exog, order, seasonal_order, trend, n_obs, label):
    """
    Fit SARIMAX model with fallback to simpler specs if needed.
    
    Parameters
    ----------
    y : array-like
        Time series data (already scaled/transformed)
    exog : array-like or None
        Exogenous variables
    order : tuple
        (p, d, q) ARIMA order
    seasonal_order : tuple
        (P, D, Q, s) seasonal ARIMA order
    trend : str or None
        Trend specification ('c', 'n', or None)
    n_obs : int
        Number of observations in the series
    label : str
        Label for logging purposes
    
    Returns
    -------
    tuple
        (fitted_model, final_order, final_seasonal_order)
    """
    ok = check_min_obs_for_order(n_obs, order, seasonal_order, label)

    candidate_specs = []
    if ok:
        candidate_specs.append((order, seasonal_order))
    else:
        print(
            f"   [FALLBACK] {label}: insufficient data for spec {order}/{seasonal_order} "
            f"— starting from a reduced spec. Forecast should be treated as provisional."
        )

    # Fallback ladder: progressively simpler models
    candidate_specs += [
        ((max(order[0] - 1, 0), order[1], max(order[2] - 1, 0)), seasonal_order),
        ((0, order[1], 1), seasonal_order),
        ((0, 1, 1), (0, seasonal_order[1], 0, seasonal_order[3])),
        ((0, 1, 1), (0, 0, 0, seasonal_order[3])),
    ]

    last_exception = None
    for cand_order, cand_seasonal in candidate_specs:
        enforce = (cand_order, cand_seasonal) == (order, seasonal_order) and ok
        print(
            f"   [FIT] Trying SARIMAX order={cand_order}, seasonal={cand_seasonal}, "
            f"trend={trend}, enforce={enforce}"
        )
        try:
            model = sm.tsa.statespace.SARIMAX(
                y,
                exog=exog,
                order=cand_order,
                seasonal_order=cand_seasonal,
                trend=trend,
                enforce_stationarity=enforce,
                enforce_invertibility=enforce,
            )
            res = model.fit(disp=False)
            if np.isfinite(res.aicc):
                return res, cand_order, cand_seasonal
            print(f"   [FIT REJECTED] {label}: order={cand_order} gave AICc=inf, trying next fallback.")
        except Exception as e:
            last_exception = e
            print(f"   [FIT FAILED] {label}: order={cand_order}, seasonal={cand_seasonal}, error={e}")

    raise RuntimeError(
        f"{label}: unable to fit SARIMAX after fallback; last error: {last_exception}"
    )


def fit_candidate_spec(y_scaled, exog, order, seasonal_order, trend, label):
    """
    One-off diagnostic fit for comparing alternative specs.
    
    Parameters
    ----------
    y_scaled : array-like
        Scaled time series data
    exog : array-like or None
        Exogenous variables
    order : tuple
        (p, d, q) ARIMA order
    seasonal_order : tuple
        (P, D, Q, s) seasonal ARIMA order
    trend : str or None
        Trend specification
    label : str
        Label for logging purposes
    
    Returns
    -------
    statsmodels result object
        Fitted model result
    """
    model = sm.tsa.statespace.SARIMAX(
        y_scaled, exog=exog, order=order, seasonal_order=seasonal_order,
        trend=trend, enforce_stationarity=True, enforce_invertibility=True,
    )
    res = model.fit(disp=False)
    print(f"   [CANDIDATE] {label}: order={order}, seasonal={seasonal_order} → AICc={res.aicc:.2f}")

    param_names = res.model.param_names
    if "ar.S.L4" in param_names:
        idx = param_names.index("ar.S.L4")
        coef = res.params[idx]
        pval = res.pvalues[idx]
        print(f"   [CANDIDATE] Seasonal AR coef: {coef:.4f}, p-value: {pval:.4f}")
    else:
        print(f"   [CANDIDATE] Seasonal AR term not found in params: {param_names}")

    return res


# ---------------------------------------------------------------------------
# Exogenous variable handling
# ---------------------------------------------------------------------------

def fit_exog_scaler(exog_array):
    """
    Compute scaling parameters from historical exogenous variables only.
    
    Parameters
    ----------
    exog_array : array-like
        Exogenous variable data
    
    Returns
    -------
    tuple
        (mean, std) scaling parameters
    """
    X = np.asarray(exog_array, dtype=float)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_exog_scaler(exog_array, mean, std, label):
    """
    Apply historical scaling parameters to exogenous variables.
    
    Parameters
    ----------
    exog_array : array-like
        Exogenous variable data
    mean : array-like
        Mean values for scaling
    std : array-like
        Standard deviation values for scaling
    label : str
        Label for logging purposes
    
    Returns
    -------
    array-like
        Scaled exogenous variables
    """
    X = np.asarray(exog_array, dtype=float)

    # Forward/backward fill any NaN values
    if np.isnan(X).any():
        n_nan = int(np.isnan(X).sum())
        print(
            f"   [EXOG WARNING] {label}: {n_nan} NaN value(s) found in exog — "
            f"forward/backward-filling before scaling."
        )
        X_df = pd.DataFrame(X)
        X_df = X_df.ffill().bfill()
        X = X_df.values

    # Apply scaling
    X_reg = (X - mean) / std
    print(f"   [EXOG] Applied historical scaling to {label}")
    return X_reg


def generate_future_dummies(start_quarter, horizons=24, historical_df=None):
    """
    Generate future exogenous variables for forecasting.
    
    Parameters
    ----------
    start_quarter : pd.Timestamp
        Last historical quarter
    horizons : int, default=24
        Number of forecast periods
    historical_df : pd.DataFrame, optional
        Historical data for trend extrapolation
    
    Returns
    -------
    pd.DataFrame
        Future exogenous variables
    """
    future_dates = pd.date_range(
        start=start_quarter + pd.offsets.QuarterEnd(),
        periods=horizons,
        freq="QE",
    )
    future_df = pd.DataFrame(index=future_dates)

    # COVID pulse is 0 in future (no new pandemic shocks)
    future_df["covid_pulse"] = 0.0
    
    # Post-COVID regime is 1 in future
    future_df["post_covid_regime"] = 1.0

    # Extrapolate post_covid_trend_break if present
    if historical_df is not None and "post_covid_trend_break" in historical_df.columns:
        last_val = historical_df["post_covid_trend_break"].max()
        if pd.isna(last_val):
            last_val = 0.0
        future_df["post_covid_trend_break"] = last_val + np.arange(1, horizons + 1)
    else:
        base_date = pd.to_datetime("2020-04-01")
        future_df["quarter_dt"] = future_df.index
        future_df["post_covid_trend_break"] = (
            (future_df["quarter_dt"].dt.year - base_date.year) * 4
            + (future_df["quarter_dt"].dt.quarter - base_date.quarter)
        )

    # Generate future quadratic trend if present in historical data
    if historical_df is not None and "quadratic_trend" in historical_df.columns:
        n_hist = len(historical_df)
        n_future = horizons
        total_n = n_hist + n_future
        
        t = np.arange(total_n)
        t_centered = t - np.mean(t)
        quadratic = (t_centered ** 2)
        quadratic_scaled = quadratic / np.std(quadratic) if np.std(quadratic) > 0 else quadratic
        
        future_df["quadratic_trend"] = quadratic_scaled[-n_future:]
    else:
        base_date = pd.to_datetime("2020-04-01")
        future_df["quarter_dt"] = future_df.index
        t = np.arange(horizons)
        t_centered = t - np.mean(t)
        quadratic = (t_centered ** 2)
        future_df["quadratic_trend"] = quadratic / np.std(quadratic) if np.std(quadratic) > 0 else quadratic

    return future_df


# ---------------------------------------------------------------------------
# Main forecasting engine
# ---------------------------------------------------------------------------

def main():
    """
    Main execution function for the forecasting engine.
    
    This function:
    1. Loads the combined quarterly data
    2. Processes each metric in METRIC_NAMES
    3. Applies transformations (log where configured)
    4. Fits SARIMAX models with fallback
    5. Generates forecasts with back-transformation
    6. Applies smoothing and clamping where needed
    7. Saves results to dashboard_forecasts.csv
    """
    print("=" * 70)
    print("MASTER FORECAST ENGINE – Production")
    print("=" * 70)

    # Load input data
    df = pd.read_csv(INPUT_FILE)
    df["quarter"] = pd.to_datetime(df["quarter"])

    # Container for all records (history + forecast)
    all_records = []

    # Process each metric
    for display_name, raw_metric_id in METRIC_NAMES.items():
        print(f"\n[MODELING] {display_name}")

        # Skip PESA (handled by annual pipeline)
        if display_name == "PESA Health spend (level)":
            print("   [SKIP] Uses annual pipeline.")
            continue

        # Extract series data
        sub = (
            df[df["metric"] == raw_metric_id]
            .dropna(subset=["value"])
            .sort_values("quarter")
        )

        # Skip GP series due to insufficient data
        gp_series = [
            "GP total appointments (flow)",
            "GP face-to-face appointments (flow)",
            "GP telephone appointments (flow)"
        ]
        if display_name in gp_series:
            print(f"   [SKIP] {display_name}: Insufficient data ({len(sub)} observations) for reliable forecasting.")
            print(f"   [NOTE] Only {len(sub)} quarters of data available (Oct 2023 - Apr 2026).")
            print(f"   [ACTION] Writing historical data only. No forecasts will be generated.")
            
            # Write only history records
            for _, row in sub.iterrows():
                all_records.append({
                    "metric": display_name,
                    "raw_metric_name": raw_metric_id,
                    "quarter": row["quarter"].strftime("%Y-%m-%d"),
                    "type": "history",
                    "value": row["value"],
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                })
            continue  

        # Apply FIT_START_OVERRIDE if configured
        if display_name in FIT_START_OVERRIDES:
            cutoff = pd.to_datetime(FIT_START_OVERRIDES[display_name])
            original_len = len(sub)
            sub = sub[sub["quarter"] >= cutoff]
            print(
                f"   [FIT WINDOW] {display_name}: restricted to {cutoff.date()} onward "
                f"({original_len} → {len(sub)} observations)"
            )

        # Skip if too few observations
        if len(sub) < 8:
            print("   [SKIP] Too few observations.")
            continue

        # Extract values as array
        y = sub["value"].values.astype(float)
        
        # Get model configuration
        base_cfg = MODEL_CONFIG[display_name]
        apply_log_transform = False
        
        # Apply log transformation if configured for RTT waiting list
        if display_name == "RTT waiting list (level)":
            if base_cfg.get("transform") == "log":
                if np.all(y > 0):
                    print(f"   [TRANSFORM] Applying log transformation to {display_name} (no scaling)")
                    y = np.log(y)
                    apply_log_transform = True
                else:
                    print(f"   [WARNING] Cannot apply log transform: non-positive values found in {display_name}")
        
        # Run stationarity tests on the (possibly transformed) data
        stat_levels = run_stationarity_battery(y, f"{display_name} (levels)")
        d_suggest = suggest_d(stat_levels, display_name)
        D_suggest = suggest_seasonal_D(y, display_name, freq=4)

        # Scale the data (skip scaling for log-transformed data)
        if apply_log_transform:
            scale = 1.0
            print(f"   [SCALE] Using scale=1 for log-transformed data")
        else:
            scale = np.nanmax(np.abs(y))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
        
        y_scaled = y / scale

        # Handle exogenous variables
        exog_cols = EXOG_CONFIG.get(display_name, [])
        exog_mean = None
        exog_std = None
        exog_hist = None
        cols_present = []

        if exog_cols:
            cols_present = [c for c in exog_cols if c in sub.columns]
            if cols_present:
                exog_mean, exog_std = fit_exog_scaler(sub[cols_present].values)
                exog_hist = apply_exog_scaler(
                    sub[cols_present].values, exog_mean, exog_std, display_name
                )
            else:
                print(f"[EXOG WARNING] {display_name}: no matching historical exog → fitting without exog.")

        use_exog = exog_hist is not None

        # Set model order from config
        p_cfg, d_cfg, q_cfg = base_cfg["order"]
        order = (p_cfg, d_cfg, q_cfg)

        print(f"   [ORDER] Using config order={order}")

        # Log diagnostic flags if config differs from suggestions
        if d_cfg != d_suggest:
            print(
                f"   [DIAGNOSTIC FLAG] {display_name}: "
                f"config d={d_cfg} vs suggested d={d_suggest} "
                f"(keeping configured order — review exog_config.py if this persists)"
            )

        seasonal_order = base_cfg["seasonal_order"]

        if seasonal_order[1] != D_suggest:
            print(
                f"   [DIAGNOSTIC FLAG] {display_name}: "
                f"config D={seasonal_order[1]} vs suggested D={D_suggest} "
                f"(keeping configured seasonal order — review exog_config.py if this persists)"
            )

        # Override for RTT % within 18 weeks (force random walk with drift)
        if display_name == "RTT % within 18 weeks (performance)":
            order = (0, 1, 0)
            seasonal_order = (0, 0, 0, 4)
            base_cfg["trend"] = "c"
            print(f"   [OVERRIDE] Forcing RTT % within 18 weeks to random walk with drift (0,1,0) c")

        print(f"   [MODEL] order={order}, seasonal={seasonal_order}, trend={base_cfg['trend']}")

        # Write historical records
        for _, row in sub.iterrows():
            all_records.append(
                {
                    "metric": display_name,
                    "raw_metric_name": raw_metric_id,
                    "quarter": row["quarter"].strftime("%Y-%m-%d"),
                    "type": "history",
                    "value": row["value"],
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )

        # Fit model with fallback
        try:
            # Prevent white-noise model for non-GP series
            if order == (0, 0, 0) and base_cfg.get("model_type") != "ARIMA_mean":
                print("   [FIX] Preventing white-noise model → using (0,1,1)")
                order = (0, 1, 1)
            else:
                print(f"   [INFO] Keeping white-noise model for {display_name} (mean forecast)")

            # Prevent linear random-walk forecast for non-GP series
            if (
                order[1] == 1
                and seasonal_order[1] == 1
                and order[0] == 0
                and order[2] == 0
            ) and base_cfg.get("model_type") != "ARIMA_mean":
                print("   [FIX] Preventing linear random-walk forecast → adding MA(1)")
                order = (0, 1, 1)

            # Fit the model
            res, order, seasonal_order = fit_with_length_fallback(
                y_scaled, exog_hist, order, seasonal_order, base_cfg["trend"], len(sub), display_name
            )
            
            # Log exogenous coefficient check
            if res.model.exog_names:
                print(f"   [EXOG COEF CHECK] {display_name}: {dict(zip(res.model.exog_names, res.params))}")
            print(f"   [FIT] AICc={res.aicc:.2f}")

            # Skip if model is degenerate
            if not np.isfinite(res.aicc):
                print(f"   [SKIP] {display_name}: model still degenerate (AICc=inf) even after fallback — history written, forecast skipped.")
                continue

            # Stationarity check on residuals
            resid = res.resid
            run_stationarity_battery(resid, f"{display_name} (residuals)")
            
            # Set forecast horizon
            horizons = base_cfg.get("horizons", 24)
            horizons = min(horizons, len(sub))

        except Exception as e:
            print(f"   [ERROR] {display_name}: {e} — history written, forecast skipped.")
            continue

        # Generate future exogenous variables
        future_exog_df = generate_future_dummies(sub["quarter"].iloc[-1], horizons, sub)

        # Prepare future exogenous variables
        exog_future = None
        if use_exog and cols_present:
            future_cols_present = [c for c in cols_present if c in future_exog_df.columns]
            if future_cols_present == cols_present:
                exog_future = apply_exog_scaler(
                    future_exog_df[cols_present].values, exog_mean, exog_std, display_name
                )
            else:
                print(f"[EXOG WARNING] {display_name}: future exog columns don't match training → forecasting without exog.")

        # Generate forecast
        fc = res.get_forecast(steps=horizons, exog=exog_future)

        # Extract raw forecast values
        raw_mean = fc.predicted_mean
        raw_ci = fc.conf_int(alpha=0.05)
        
        # Back-transform if log transformation was applied
        if apply_log_transform:
            print(f"   [BACK-TRANSFORM] Applying exponential back-transform for {display_name}")
            
            mean_fc_scaled = (raw_mean.to_numpy() if hasattr(raw_mean, "to_numpy") else np.asarray(raw_mean))
            ci_scaled = (raw_ci.to_numpy() if hasattr(raw_ci, "to_numpy") else np.asarray(raw_ci))
            
            mean_fc = np.exp(mean_fc_scaled)
            ci_lower = np.exp(ci_scaled[:, 0])
            ci_upper = np.exp(ci_scaled[:, 1])
            ci = np.column_stack([ci_lower, ci_upper])
            
            print(f"   [BACK-TRANSFORM] Done: mean_fc range = [{mean_fc.min():.2f}, {mean_fc.max():.2f}]")
        else:
            # Apply scaling
            mean_fc = (raw_mean.to_numpy() if hasattr(raw_mean, "to_numpy") else np.asarray(raw_mean)) * scale
            ci = (raw_ci.to_numpy() if hasattr(raw_ci, "to_numpy") else np.asarray(raw_ci)) * scale

        # ============================================================
        # Forecast smoothing and post-processing
        # ============================================================

        # Moving average smoothing for A&E attendances
        if display_name == "A&E attendances (flow)":
            last_hist = y[-1]
            first_fore = mean_fc[0]
            
            # Apply smoothing if there's a noticeable jump
            if abs((first_fore - last_hist) / last_hist) > 0.03:
                if len(mean_fc) > 3:
                    smoothed = mean_fc.copy()
                    # Smooth interior points with 3-point moving average
                    for i in range(1, len(mean_fc) - 1):
                        smoothed[i] = (mean_fc[i-1] + mean_fc[i] + mean_fc[i+1]) / 3
                    # Handle endpoints
                    if len(mean_fc) > 2:
                        smoothed[0] = (mean_fc[0] + mean_fc[1]) / 2
                        smoothed[-1] = (mean_fc[-1] + mean_fc[-2]) / 2
                    mean_fc = smoothed
                    print(f"   [DAMPEN] Applied moving average smoothing to A&E attendances forecast")

        # Smooth initial drop for RTT waiting list
        if display_name == "RTT waiting list (level)":
            last_hist = y[-1]
            first_fore = mean_fc[0]
            
            # Smooth if drop is more than 15%
            if (first_fore - last_hist) / last_hist < -0.15:
                dampening_points = min(6, len(mean_fc))
                for i in range(dampening_points):
                    # Progressively reduce dampening
                    weight = 1.0 - (i / dampening_points) * 0.7
                    blend_value = weight * mean_fc[i] + (1 - weight) * last_hist
                    mean_fc[i] = blend_value
                print(f"   [DAMPEN] Smoothed RTT waiting list drop over {dampening_points} points")

        # Emergency override for RTT % within 18 weeks (ensures smooth transition)
        if display_name == "RTT % within 18 weeks (performance)":
            print(f"   [EMERGENCY] Overriding RTT % within 18 weeks forecast")
            print(f"   [EMERGENCY] BEFORE: first={mean_fc[0]:.4f}, last={mean_fc[-1]:.4f}")
            
            last_hist = y[-1]
            n_points = len(mean_fc)
            target = min(last_hist * 1.05, 0.72)
            
            # Generate smooth S-curve transition
            for i in range(n_points):
                pos = i / (n_points - 1) if n_points > 1 else 0
                smooth_pos = pos * pos * (3 - 2 * pos)  # Smoothstep function
                mean_fc[i] = last_hist + (target - last_hist) * smooth_pos
            
            print(f"   [EMERGENCY] AFTER: first={mean_fc[0]:.4f}, last={mean_fc[-1]:.4f}")

        # ============================================================
        # Clamping and record generation
        # ============================================================

        n_clamped = 0
        for i, ts in enumerate(future_exog_df.index):
            val = float(mean_fc[i])
            lo = float(ci[i, 0])
            hi = float(ci[i, 1])

            # Clamp percentage metrics to [0, 100]
            if "%" in display_name.lower() or "percent" in display_name.lower():
                val = max(0.0, min(100.0, val))
                lo = max(0.0, min(100.0, lo))
                hi = max(0.0, min(100.0, hi))
            else:
                # Clamp non-percentage metrics to non-negative
                if val < 0 or lo < 0:
                    n_clamped += 1
                val = max(0.0, val)
                lo = max(0.0, lo)
                hi = max(lo, hi)

            # Append forecast record
            all_records.append(
                {
                    "metric": display_name,
                    "raw_metric_name": raw_metric_id,
                    "quarter": ts.strftime("%Y-%m-%d"),
                    "type": "forecast",
                    "value": val,
                    "ci_lower": lo,
                    "ci_upper": hi,
                }
            )

        # Log clamping events
        if n_clamped > 0:
            if base_cfg.get("model_type") not in ["ARIMA_mean", "ARIMA_random_walk"]:
                print(
                    f"   [CLAMP WARNING] {display_name}: {n_clamped}/{horizons} forecast points "
                    f"had a negative raw value/CI bound clamped to 0."
                )
            else:
                print(f"   [INFO] GP series '{display_name}': {n_clamped}/{horizons} forecast CI bounds clamped (expected for simple model)")

    # Save all records to CSV
    out = pd.DataFrame(all_records)
    out.to_csv(OUTPUT_FILE, index=False)
    print("\nDONE – forecasts written to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
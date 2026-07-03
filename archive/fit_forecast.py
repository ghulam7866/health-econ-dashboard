"""
fit_forecast.py
-----------------
Fits the final SARIMAX model for each series using the orders confirmed
by select_arima_orders.py, then produces a 6-year (24-quarter) forecast
with confidence intervals.

Exog handling for the forecast horizon:
    - post_covid_regime = 1 for ALL future quarters (it's now the
      permanent post-2020 regime, not a temporary window) - NOTE: no
      longer used in any of the 3 final specs as of this version, since
      it was degenerate for Workforce (see exog_config.py NOTES) and was
      never used for RTT/A&E. Left in build_future_exog in case it's
      reintroduced later (e.g. with a proper trend interaction).
    - covid_pulse = 0 for ALL future quarters (the acute disruption
      window has passed and won't recur in the forecast horizon)

Trend handling:
    - RTT and Workforce both show a sustained climb with no sign of
      levelling off as of the latest data. A (p,1,q) model with no trend
      term forecasts roughly flat from the last observed value, which
      would misrepresent both series. trend='c' under d=1 differencing
      adds a constant drift (equivalent to a linear trend in levels) -
      this is what makes the forecast continue climbing rather than
      flattening.
    - A&E's trend setting is left as None (OPEN QUESTION - see
      exog_config.py NOTES for A&E: unconfirmed whether trend='c' was
      ever included in select_arima_orders.py's grid search for this
      series).

Run:
    python src/fit_forecast.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exog_config import EXOG_CONFIG, METRIC_NAMES

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"
FORECAST_HORIZON = 24  # 6 years of quarters

# Confirmed orders from select_arima_orders.py (full grid search), plus
# trend - see module docstring for why RTT/Workforce now carry trend='c'
# while A&E is left as an open question.
MODEL_SPECS = {
    "RTT waiting list (level)": {
        "order": (0, 1, 3),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "c",
    },
    "A&E attendances (flow)": {
        "order": (0, 1, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": None,  # OPEN QUESTION - confirm against select_arima_orders.py
    },
    "Workforce FTE (level)": {
        "order": (0, 1, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "c",
    },
}

# Series -> exog column to test dropping, purely for AIC comparison.
# Add entries here whenever a coefficient looks non-significant but isn't
# flagged as numerically unstable (the SE/coef ratio check only catches
# instability, not "this term may not be pulling its weight" - that's a
# parsimony question, decided here by direct AIC comparison instead).
# RTT's covid_pulse check (run previously) confirmed AIC improves by 6.70
# points dropping it - already applied in EXOG_CONFIG, so nothing pending here.
PARSIMONY_CHECKS = {}


def compare_exog_drop(df: pd.DataFrame, label: str, drop_col: str):
    """Fits the same series with and without one exog column, holding
    order/seasonal_order/trend fixed, and prints the AIC difference."""
    metric = METRIC_NAMES[label]
    exog_cols = EXOG_CONFIG[label]
    spec = MODEL_SPECS[label]
    sub = df[df["metric"] == metric].sort_values("quarter").reset_index(drop=True)
    y = sub["value"]

    print(f"\n{'-'*70}")
    print(f"PARSIMONY CHECK: {label} — with vs without '{drop_col}'")
    print(f"{'-'*70}")

    results = {}
    for variant_name, cols in [("with", exog_cols), ("without", [c for c in exog_cols if c != drop_col])]:
        X = sub[cols]
        model = SARIMAX(
            y, exog=X,
            order=spec["order"], seasonal_order=spec["seasonal_order"], trend=spec["trend"],
            enforce_stationarity=False, enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        results[variant_name] = fitted.aic
        print(f"  {variant_name:<8} exog={cols}  AIC={fitted.aic:.2f}")

    delta = results["without"] - results["with"]
    if delta < -2:
        verdict = f"DROP '{drop_col}' - AIC improves by {-delta:.2f} points"
    elif delta > 2:
        verdict = f"KEEP '{drop_col}' - AIC worsens by {delta:.2f} points without it"
    else:
        verdict = f"MARGINAL (ΔAIC={delta:.2f}) - either choice is roughly AIC-neutral, use judgement"
    print(f"  → {verdict}")


def build_future_exog(exog_cols: list, n_periods: int, last_row: pd.Series) -> pd.DataFrame:
    """
    Builds the exog matrix for the forecast horizon.
    covid_pulse = 0 throughout (acute window has passed).
    post_covid_regime = 1 throughout - kept here for reuse if ever
      reintroduced; not used by any current spec (dropped from all three
      series - see exog_config.py NOTES for why an 'on forever' step
      dummy is structurally unsafe under d>=1 differencing regardless of
      its OLS significance).
    post_covid_trend_break = continues counting up from its last observed
      value in the data (NOT reset to 0 or a fixed value) - it represents
      "quarters since the break", which keeps incrementing into the future
      exactly like the historical column does.
    """
    future = pd.DataFrame(index=range(n_periods))
    for col in exog_cols:
        if col == "post_covid_regime":
            future[col] = 1
        elif col == "covid_pulse":
            future[col] = 0
        elif col == "post_covid_trend_break":
            last_value = last_row[col]
            future[col] = last_value + np.arange(1, n_periods + 1)
        else:
            raise ValueError(f"Unknown exog column '{col}' — no forward-fill rule defined.")
    return future


def fit_and_forecast(df: pd.DataFrame, label: str) -> pd.DataFrame:
    metric = METRIC_NAMES[label]
    exog_cols = EXOG_CONFIG[label]
    spec = MODEL_SPECS[label]

    sub = df[df["metric"] == metric].sort_values("quarter").reset_index(drop=True)
    y = sub["value"]
    X = sub[exog_cols]
    last_quarter = sub["quarter"].max()

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"  Fitting SARIMAX{spec['order']}{spec['seasonal_order']} "
          f"trend={spec['trend']!r} with exog={exog_cols}")
    print(f"{'='*70}")

    model = SARIMAX(
        y,
        exog=X,
        order=spec["order"],
        seasonal_order=spec["seasonal_order"],
        trend=spec["trend"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    print(fitted.summary().tables[1])  # coefficient table only, not full summary

    # Build future quarters and exog
    future_quarters = pd.date_range(
        start=last_quarter + pd.offsets.QuarterBegin(startingMonth=1),
        periods=FORECAST_HORIZON,
        freq="QS"
    )
    future_X = build_future_exog(exog_cols, FORECAST_HORIZON, sub.iloc[-1])

    forecast = fitted.get_forecast(steps=FORECAST_HORIZON, exog=future_X)
    mean_fc = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.05)  # 95% CI

    out = pd.DataFrame({
        "quarter": future_quarters,
        "metric": label,
        "forecast": mean_fc.values,
        "ci_lower": ci.iloc[:, 0].values,
        "ci_upper": ci.iloc[:, 1].values,
    })

    # Basic residual diagnostic — Ljung-Box on residuals (quick check only)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(fitted.resid, lags=[4], return_df=True)
    lb_p = lb["lb_pvalue"].iloc[0]
    print(f"\n  Ljung-Box (lag 4) p-value: {lb_p:.4f}  "
          f"{'OK — residuals look like white noise' if lb_p > 0.05 else 'WARNING — residual autocorrelation remains'}")
    print(f"  AIC: {fitted.aic:.2f}")

    # Sanity check on exog coefficients - flag if SE looks degenerate in
    # either direction (too large -> instability like A&E's old pulse term;
    # too small -> near-singular, like Workforce's old regime term)
    if exog_cols:
        params = fitted.params
        bse = fitted.bse
        for col in exog_cols:
            if col in params.index and params[col] != 0:
                ratio = bse[col] / abs(params[col])
                if ratio > 1.0:
                    print(f"  ⚠ {col}: SE/coef ratio {ratio:.2f} — looks unstable (too uncertain)")
                elif ratio < 1e-6:
                    print(f"  ⚠ {col}: SE/coef ratio {ratio:.2e} — looks degenerate (too certain)")

    return out


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    for label, drop_col in PARSIMONY_CHECKS.items():
        compare_exog_drop(df, label, drop_col)

    all_forecasts = []
    for label in EXOG_CONFIG.keys():
        fc = fit_and_forecast(df, label)
        all_forecasts.append(fc)

    combined_forecast = pd.concat(all_forecasts, ignore_index=True)
    out_path = PROCESSED_DIR / "forecast_6yr.csv"
    combined_forecast.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"✓ Saved 6-year forecast → {out_path}")
    print(f"  {len(combined_forecast)} rows ({FORECAST_HORIZON} quarters x {len(EXOG_CONFIG)} series)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
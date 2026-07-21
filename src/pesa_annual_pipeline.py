"""
pesa_annual_pipeline.py
--------------------------
An automated pipeline for generating PESA Health spend forecasts.

The script ingests the forward‑filled quarterly representation of PESA
data from `combined_quarterly.csv`, extracts the annual series, fits a
structural time‑series model (Unobserved Components with a local linear
trend), and produces a 6‑year forecast.  The forecast is expanded back to
quarterly frequency and merged into `dashboard_forecasts.csv`.

Usage:
    python src/pesa_annual_pipeline.py

Requirements:
    - combined_quarterly.csv must exist in data/processed/.
    - dashboard_forecasts.csv may already exist; this script will update
      or create it.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import sys
import statsmodels.api as sm

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"
FORECAST_PATH = PROCESSED_DIR / "dashboard_forecasts.csv"

# Allow importing from src/ for configuration
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from exog_config import METRIC_NAMES, EXOG_CONFIG
except ImportError:
    METRIC_NAMES = {"PESA Health spend (level)": "mock_internal_name"}
    EXOG_CONFIG = {"PESA Health spend (level)": []}

LABEL = "PESA Health spend (level)"
FORECAST_YEARS = 6
TARGET_COL = "value"


def extract_annual_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the PESA health spend series and collapse it to annual frequency.

    Because the combined quarterly data forward‑fills the same annual value
    across all four quarters, we simply take the first occurrence in each
    year to obtain a single annual observation.
    """
    metric = METRIC_NAMES[LABEL]
    sub = df[df["metric"] == metric].sort_values("quarter").reset_index(drop=True)
    sub["year"] = sub["quarter"].dt.year
    annual = sub.groupby("year").first().reset_index()
    annual["date"] = pd.to_datetime(annual["year"].astype(str) + "-01-01")
    return annual


def fit_ucm_and_forecast(annual: pd.DataFrame, exog_cols: list):
    """
    Fit an Unobserved Components Model with a local linear trend and
    (optionally) exogenous structural variables, then produce a 6‑year
    forecast.

    Returns
    -------
    annual_forecast : pd.DataFrame
        Contains 'year', 'forecast', 'ci_lower', 'ci_upper'.
    fitted : statsmodels UnobservedComponentsResults
        The fitted model object (useful for summary inspection).
    """
    y = annual[TARGET_COL]
    X = annual[exog_cols] if exog_cols else None

    model = sm.tsa.UnobservedComponents(
        y,
        level='local linear trend',
        exog=X
    )
    fitted = model.fit(disp=False)

    # Build future exogenous: for PESA, covid_pulse stays 0, post_covid_regime stays 1.
    future_X = None
    if exog_cols:
        future_X = pd.DataFrame(index=range(FORECAST_YEARS))
        for col in exog_cols:
            if col == "covid_pulse":
                future_X[col] = 0
            elif col == "post_covid_regime":
                future_X[col] = 1

    forecast = fitted.get_forecast(steps=FORECAST_YEARS, exog=future_X)
    mean_fc = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.05)

    last_year = annual["date"].dt.year.max()
    future_years = range(last_year + 1, last_year + 1 + FORECAST_YEARS)

    annual_forecast = pd.DataFrame({
        "year": future_years,
        "forecast": mean_fc.values,
        "ci_lower": ci.iloc[:, 0].values,
        "ci_upper": ci.iloc[:, 1].values,
    })
    return annual_forecast, fitted


def expand_to_quarters(annual_forecast: pd.DataFrame, annual_history: pd.DataFrame) -> pd.DataFrame:
    """
    Convert annual forecast/history rows into quarterly rows by repeating
    the same value for every quarter of the year (Jan, Apr, Jul, Oct).
    """
    quarterly_rows = []

    # Historical data
    for _, row in annual_history.iterrows():
        for month in (1, 4, 7, 10):
            quarterly_rows.append({
                "metric": LABEL,
                "raw_metric_name": METRIC_NAMES[LABEL],
                "quarter": pd.Timestamp(year=int(row["year"]), month=month, day=1).strftime("%Y-%m-%d"),
                "type": "history",
                "value": row["value"],
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            })

    # Forecast data
    for _, row in annual_forecast.iterrows():
        for month in (1, 4, 7, 10):
            quarterly_rows.append({
                "metric": LABEL,
                "raw_metric_name": METRIC_NAMES[LABEL],
                "quarter": pd.Timestamp(year=int(row["year"]), month=month, day=1).strftime("%Y-%m-%d"),
                "type": "forecast",
                "value": row["forecast"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
            })

    return pd.DataFrame(quarterly_rows)


def main():
    if not COMBINED_PATH.exists():
        print(f"Error: Path not found {COMBINED_PATH}")
        return

    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    annual = extract_annual_series(df)

    FINAL_EXOG_COLS = EXOG_CONFIG.get(LABEL, [])

    print("\n" + "=" * 70)
    print("FITTING STATE SPACE STRUCTURAL TIME SERIES (LOCAL LINEAR TREND)")
    print(f"Specification: Level + Dynamic Slope | Exogenous={FINAL_EXOG_COLS}")
    print("=" * 70)

    annual_forecast, fitted = fit_ucm_and_forecast(annual, FINAL_EXOG_COLS)

    # Print the coefficient table from the fitted model for diagnostic purposes
    print(fitted.summary().tables[1])

    quarterly_forecast = expand_to_quarters(annual_forecast, annual)

    # Merge into the existing forecast file, replacing any old PESA rows
    if FORECAST_PATH.exists():
        existing = pd.read_csv(FORECAST_PATH)
        existing = existing[existing["metric"] != LABEL]
        combined = pd.concat([existing, quarterly_forecast], ignore_index=True)
    else:
        combined = quarterly_forecast

    combined.to_csv(FORECAST_PATH, index=False)
    print(f"\n[SUCCESS] Process complete. Saved PESA forecast to {FORECAST_PATH}")


if __name__ == "__main__":
    main()
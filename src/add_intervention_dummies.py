"""
add_intervention_dummies.py
----------------------------
Adds COVID intervention dummy variables to combined_quarterly.csv, for use
as exogenous regressors in forecasting/econometric models (e.g. SARIMAX
exog, or as controls in any regression/VAR) rather than deleting the
disrupted period outright.

UPDATED: 2026-07-01 - Preserves quadratic_trend column.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

PULSE_START = "2020-01-01"
PULSE_END = "2021-01-01"
REGIME_START = "2020-04-01"


def add_dummies(df: pd.DataFrame) -> pd.DataFrame:
    quarters = pd.to_datetime(df["quarter"])
    df["covid_pulse"] = ((quarters >= PULSE_START) & (quarters <= PULSE_END)).astype(int)
    df["post_covid_regime"] = (quarters >= REGIME_START).astype(int)

    regime_start_ts = pd.Timestamp(REGIME_START)
    quarters_since_break = (
        (quarters.dt.year - regime_start_ts.year) * 4
        + (quarters.dt.quarter - regime_start_ts.quarter)
    )
    df["post_covid_trend_break"] = quarters_since_break.clip(lower=0)
    df.loc[quarters < regime_start_ts, "post_covid_trend_break"] = 0

    return df


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    print(f"Loaded combined data: {len(df)} rows")
    print(f"Columns before: {df.columns.tolist()}")

    # Check if quadratic_trend exists
    if 'quadratic_trend' not in df.columns:
        print("⚠ WARNING: quadratic_trend not found - adding it now as fallback")
        unique_quarters = sorted(df["quarter"].unique())
        n = len(unique_quarters)
        t = np.arange(n)
        t_centered = t - np.mean(t)
        quadratic = (t_centered ** 2)
        quadratic_scaled = quadratic / np.std(quadratic) if np.std(quadratic) > 0 else quadratic
        quarter_to_quadratic = {q: val for q, val in zip(unique_quarters, quadratic_scaled)}
        df['quadratic_trend'] = df["quarter"].map(quarter_to_quadratic)
    else:
        print(f"✓ quadratic_trend preserved (range: {df['quadratic_trend'].min():.3f} → {df['quadratic_trend'].max():.3f})")

    df = add_dummies(df)
    
    print(f"Columns after: {df.columns.tolist()}")
    
    df.to_csv(COMBINED_PATH, index=False)
    print(f"✓ Saved with dummies → {COMBINED_PATH}")


if __name__ == "__main__":
    main()
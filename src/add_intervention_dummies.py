"""
add_intervention_dummies.py
----------------------------
Adds COVID‑19 intervention dummy variables to `combined_quarterly.csv`.

These dummies serve as exogenous regressors in the SARIMAX forecasting
models and in any subsequent econometric analysis.  Rather than discarding
the pandemic‑disrupted period, we encode it with three variables:

    • covid_pulse          – acute COVID period (2020‑Q1 to 2021‑Q1)
    • post_covid_regime     – permanent step from 2020‑Q2 onward
    • post_covid_trend_break – quarters elapsed since 2020‑Q2 (a ramp)

The script is designed to be run after `align_merge.py` and before the
master forecast engine.  It preserves the `quadratic_trend` column that
was already added by `align_merge.py`.

Usage:
    python src/add_intervention_dummies.py

Last updated: 2026‑07‑01
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

# Dates that define the COVID‑era interventions
PULSE_START = "2020-01-01"
PULSE_END   = "2021-01-01"
REGIME_START = "2020-04-01"


def add_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the three intervention dummies and add them to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Contains a column 'quarter' (datetime).

    Returns
    -------
    pd.DataFrame
        The same DataFrame with three new columns:
        'covid_pulse', 'post_covid_regime', 'post_covid_trend_break'.
    """
    quarters = pd.to_datetime(df["quarter"])

    # 1‑quarter acute COVID pulse (Jan 2020 – Jan 2021)
    df["covid_pulse"] = (
        (quarters >= PULSE_START) & (quarters <= PULSE_END)
    ).astype(int)

    # Permanent regime change from April 2020 onward
    df["post_covid_regime"] = (quarters >= REGIME_START).astype(int)

    # Ramp: number of quarters elapsed since the regime start.
    # This creates a linear trend that begins at the break and can be
    # used to model post‑COVID trend changes.
    regime_start_ts = pd.Timestamp(REGIME_START)
    quarters_since_break = (
        (quarters.dt.year - regime_start_ts.year) * 4
        + (quarters.dt.quarter - regime_start_ts.quarter)
    )
    df["post_covid_trend_break"] = quarters_since_break.clip(lower=0)

    # Explicitly zero out any pre‑break values (belt‑and‑braces)
    df.loc[quarters < regime_start_ts, "post_covid_trend_break"] = 0

    return df


def main():
    """Load combined data, inject dummies, and save."""
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    print(f"Loaded combined data: {len(df)} rows")
    print(f"Columns before: {df.columns.tolist()}")

    # If `quadratic_trend` is missing (should have been added by align_merge),
    # create it here as a fallback so downstream code doesn't break.
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
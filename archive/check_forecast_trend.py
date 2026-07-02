"""
check_forecast_trend.py
------------------------
Quick sanity check on forecast_6yr.csv: prints the first and last forecast
values per series, plus the implied average per-quarter change, so you can
confirm RTT and Workforce are actually climbing across the 24-quarter
horizon rather than flattening out near the last observed value.

Run:
    python check_forecast_trend.py
"""

import pandas as pd
from pathlib import Path
import sys

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
FORECAST_PATH = PROCESSED_DIR / "forecast_6yr.csv"
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from exog_config import METRIC_NAMES  # maps friendly label -> raw metric name


def main():
    fc = pd.read_csv(FORECAST_PATH)
    fc["quarter"] = pd.to_datetime(fc["quarter"])

    combined = pd.read_csv(COMBINED_PATH)
    combined["quarter"] = pd.to_datetime(combined["quarter"])

    for label in fc["metric"].unique():
        sub = fc[fc["metric"] == label].sort_values("quarter").reset_index(drop=True)
        print("=" * 70)
        print(label)
        print("=" * 70)

        # Translate friendly label -> raw metric name used in combined_quarterly.csv
        raw_metric = METRIC_NAMES.get(label)
        if raw_metric is None:
            print(f"  ⚠ No mapping found in METRIC_NAMES for '{label}' — skipping actual comparison")
            actual_sub = pd.DataFrame()
        else:
            actual_sub = combined[combined["metric"] == raw_metric].sort_values("quarter")

        if actual_sub.empty:
            print(f"  ⚠ No rows found in combined_quarterly.csv for raw metric '{raw_metric}'")
        else:
            last_actual_row = actual_sub.iloc[-1]
            print(f"  Last observed ({last_actual_row['quarter'].date()}): {last_actual_row['value']:,.1f}")

        print(f"  Forecast Q1  ({sub['quarter'].iloc[0].date()}): {sub['forecast'].iloc[0]:,.1f}")
        print(f"  Forecast Q12 ({sub['quarter'].iloc[11].date()}): {sub['forecast'].iloc[11]:,.1f}")
        print(f"  Forecast Q24 ({sub['quarter'].iloc[-1].date()}): {sub['forecast'].iloc[-1]:,.1f}")

        total_change = sub["forecast"].iloc[-1] - sub["forecast"].iloc[0]
        avg_per_quarter = total_change / (len(sub) - 1)
        pct_change = (sub["forecast"].iloc[-1] / sub["forecast"].iloc[0] - 1) * 100

        print(f"  Change Q1 -> Q24: {total_change:,.1f}  ({pct_change:+.1f}%)")
        print(f"  Avg change per quarter: {avg_per_quarter:,.1f}")

        if abs(pct_change) < 1:
            print("  ⚠ Forecast looks essentially FLAT across the horizon")
        print()


if __name__ == "__main__":
    main()
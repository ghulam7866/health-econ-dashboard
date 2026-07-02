"""
check_pesa_beds_shape.py
----------------------------
Quick look at PESA and Beds series shape before running the full
diagnostic sequence - specifically checking PESA's forward-fill pattern
(repeated values within each year) since that will distort stationarity
tests and ARIMA order selection if not handled.

Run:
    python check_pesa_beds_shape.py
"""

import pandas as pd
from pathlib import Path
import sys

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from exog_config import METRIC_NAMES


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    for label in ["Bed occupancy (level)", "PESA Health spend (level)"]:
        metric = METRIC_NAMES[label]
        sub = df[df["metric"] == metric].sort_values("quarter").reset_index(drop=True)

        print("=" * 70)
        print(label)
        print("=" * 70)
        print(f"  n obs: {len(sub)}")
        print(f"  date range: {sub['quarter'].min().date()} → {sub['quarter'].max().date()}")

        # Check for repeated values (forward-fill signature)
        sub["is_repeat_of_prev"] = sub["value"] == sub["value"].shift(1)
        n_repeats = sub["is_repeat_of_prev"].sum()
        pct_repeats = n_repeats / len(sub) * 100
        print(f"  Repeated consecutive values: {n_repeats}/{len(sub)} ({pct_repeats:.1f}%)")

        print(f"\n  First 12 values:")
        print(sub[["quarter", "value"]].head(12).to_string(index=False))
        print()


if __name__ == "__main__":
    main()

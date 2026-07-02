"""
inspect_new_series.py
-----------------------
Lists available metrics for PESA, Beds, and GP Appointments in
combined_quarterly.csv, so we can pick the right headline series for
each before running them through the forecasting pipeline.

Run:
    python inspect_new_series.py
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    for source in ["PESA", "Beds", "GP Appointments"]:
        print("=" * 70)
        print(source)
        print("=" * 70)
        sub = df[df["source"] == source]
        if sub.empty:
            print(f"  ⚠ No rows found with source == '{source}'")
            print(f"  Available sources: {df['source'].unique().tolist()}")
            continue

        metrics = sub["metric"].unique()
        print(f"  {len(metrics)} metric(s) found:\n")
        for m in metrics:
            m_sub = sub[sub["metric"] == m].sort_values("quarter")
            n = len(m_sub)
            date_range = f"{m_sub['quarter'].min().date()} → {m_sub['quarter'].max().date()}"
            last_val = m_sub["value"].iloc[-1]
            print(f"    [{n:>3} obs, {date_range}]  {m}")
            print(f"        last value: {last_val:,.1f}")
        print()


if __name__ == "__main__":
    main()

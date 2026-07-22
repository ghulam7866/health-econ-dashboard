"""
capacity_gap.py
----------------
Compute the gap between the actual forecast and the ITS counterfactual
(what the series would have been without the COVID break).

This script reads the dashboard forecast file and the counterfactuals file
(both produced by master_forecast_engine.py), merges them on quarter and
metric, and computes the absolute and percentage gap for four key series.
Results are printed to the console and saved to capacity_gap_table.csv.

The gap is defined as:
    gap_abs = actual_forecast_value - counterfactual_mean
    gap_pct = (gap_abs / counterfactual_mean) * 100

Usage:
    python capacity_gap.py

Input:
    data/processed/dashboard_forecasts.csv
    data/processed/counterfactuals.csv

Output:
    Console summary for RTT waiting list, A&E attendances, Bed occupancy,
    and RTT % within 18 weeks.
    data/processed/capacity_gap_table.csv (full quarter‑by‑quarter table)
"""
import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
FORECAST_PATH = PROCESSED / "dashboard_forecasts.csv"
COUNTER_PATH = PROCESSED / "counterfactuals.csv"

# Series for which the capacity‑gap is most relevant to capacity planning
SERIES = [
    "RTT waiting list (level)",
    "A&E attendances (flow)",
    "Bed occupancy (level)",
    "RTT % within 18 weeks (performance)",
]

# ---- Load data ----
fc = pd.read_csv(FORECAST_PATH, parse_dates=["quarter"])
fc = fc[fc["type"] == "forecast"]          # keep only forecast rows (exclude history)

cf = pd.read_csv(COUNTER_PATH, parse_dates=["quarter"])

# Merge actual forecasts with counterfactuals on quarter and metric name
merged = pd.merge(
    fc, cf,
    on=["quarter", "metric"],
    how="inner",
    suffixes=("_actual", "_counter")
)

# Compute gaps
merged["gap_abs"] = merged["value"] - merged["counterfactual_mean"]
merged["gap_pct"] = (merged["gap_abs"] / merged["counterfactual_mean"].replace(0, pd.NA)) * 100

print("=" * 100)
print("CAPACITY‑GAP ANALYSIS (actual forecast vs. no‑COVID counterfactual)")
print("=" * 100)

for series in SERIES:
    sub = merged[merged["metric"] == series].sort_values("quarter")
    if sub.empty:
        print(f"\n{series}: no data")
        continue

    last = sub.iloc[-1]                    # latest forecast quarter
    avg8 = sub.head(8)                     # first 8 forecast quarters
    avg_gap_abs = avg8["gap_abs"].mean()
    avg_gap_pct = avg8["gap_pct"].mean()

    # ---- Formatting helpers ----
    def fmt_val(v, is_pct_metric=False):
        """Format a value with appropriate decimal places based on scale."""
        if abs(v) < 10 and is_pct_metric:
            return f"{v:.4f}"              # proportion – show 4 decimal places
        elif abs(v) < 100:
            return f"{v:.2f}"
        elif abs(v) < 1e6:
            return f"{v:,.1f}"
        else:
            return f"{v:,.0f}"

    def fmt_pct(v):
        """Format a percentage gap with appropriate precision."""
        if abs(v) < 1:
            return f"{v:+.4f}%"
        elif abs(v) < 10:
            return f"{v:+.2f}%"
        else:
            return f"{v:+.1f}%"

    # RTT % is a proportion (0–1); format accordingly
    is_proportion = "RTT %" in series or "performance" in series

    print(f"\n{series}")
    print(f"  Latest quarter ({last['quarter'].strftime('%Y-%m-%d')}):")
    print(f"    Actual forecast:      {fmt_val(last['value'], is_proportion)}")
    print(f"    Counterfactual:       {fmt_val(last['counterfactual_mean'], is_proportion)}")
    print(f"    Absolute gap:         {fmt_val(last['gap_abs'], False)}")
    print(f"    Percentage gap:       {fmt_pct(last['gap_pct'])}")
    print(f"  Average over h=1‑8:")
    print(f"    Absolute gap:         {fmt_val(avg_gap_abs, False)}")
    print(f"    Percentage gap:       {fmt_pct(avg_gap_pct)}")

# Save the full merged table for further inspection
merged.to_csv(PROCESSED / "capacity_gap_table.csv", index=False)
print("\nFull gap table saved to data/processed/capacity_gap_table.csv")
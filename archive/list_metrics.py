"""
list_metrics.py
----------------
Lists every distinct metric in each processed CSV, with its value range
and most recent value, so you can pick which ones become the dashboard's
headline series.

Run:
    python list_metrics.py
"""
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")

FILES = [
    "rtt_clean.csv",
    "ae_clean.csv",
    "beds_clean.csv",
    "population_clean.csv",
    "gp_appointments_clean.csv",
    "workforce_clean.csv",
    "pesa_clean.csv",
    "nice_clean.csv",
]


def list_metrics(filename):
    path = PROCESSED_DIR / filename
    print("=" * 70)
    print(filename)
    print("=" * 70)
    if not path.exists():
        print(f"  NOT FOUND: {path}")
        return

    df = pd.read_csv(path)
    if "metric" not in df.columns:
        print(f"  (no 'metric' column — single series, {len(df)} rows)")
        print()
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    summary = (
        df.groupby("metric")
        .apply(lambda g: pd.Series({
            "n_points": len(g),
            "min": g["value"].min(),
            "max": g["value"].max(),
            "latest_date": g["date"].max(),
            "latest_value": g.sort_values("date")["value"].iloc[-1],
        }))
        .sort_values("latest_value", ascending=False)
    )
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(summary.to_string())
    print()


def main():
    for f in FILES:
        list_metrics(f)


if __name__ == "__main__":
    main()

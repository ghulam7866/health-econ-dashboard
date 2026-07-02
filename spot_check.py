"""
spot_check.py
-------------
Loads each processed CSV and prints summary stats + the most recent rows
for a few headline metrics per source, ensuring data look completely sane.
"""
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

if not PROCESSED_DIR.exists():
    PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")

CHECKS = {
    "rtt_clean.csv": ["% within 18 weeks", "Incomplete RTT pathways - Total waiting"],
    "ae_clean.csv": ["total_attendances"],
    "beds_clean.csv": [],
    "population_clean.csv": [],
    "gp_appointments_clean.csv": ["total_attended_appointments"],
    "workforce_clean.csv": ["FTE: All staff groups - All staff groups", "Nurses & health visitors"],
    "pesa_clean.csv": ["Health"],
    "nice_clean.csv": [],
}

def spot_check(filename, metric_filters):
    path = PROCESSED_DIR / filename
    print("=" * 75)
    print(f"[INSPECTING] {filename}")
    print("=" * 75)
    if not path.exists():
        print(f"  [NOT FOUND] {path}")
        return

    df = pd.read_csv(path)
    print(f"   Shape Matrix: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    date_col = "date" if "date" in df.columns else ("quarter" if "quarter" in df.columns else None)
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        print(f"   Temporal Horizon: {df[date_col].min()} -> {df[date_col].max()}")

    if "value" in df.columns:
        print(f"   Value boundaries: {df['value'].min():,.2f} -> {df['value'].max():,.2f}")

    if "metric" in df.columns and metric_filters:
        for f in metric_filters:
            sub = df[df["metric"].str.contains(f, case=False, na=False)]
            if sub.empty:
                print(f"\n   [WARNING] No rows matched metric filter target: '{f}'")
                continue
            print(f"\n   --- Metric subset contains '{f}' ({sub['metric'].nunique()} unique keys) ---")
            print(f"   Subset boundaries: {sub['value'].min():,.2f} -> {sub['value'].max():,.2f}")
            print("   Most recent 5 historical rows:")
            sort_key = date_col if date_col else df.columns[0]
            print(sub.sort_values(sort_key).tail(5).to_string(index=False))
    else:
        print("\n   Most recent 5 historical rows:")
        sort_key = date_col if date_col else df.columns[0]
        print(df.sort_values(sort_key).tail(5).to_string(index=False))
    print()

def main():
    for filename, filters in CHECKS.items():
        spot_check(filename, filters)

if __name__ == "__main__":
    main()
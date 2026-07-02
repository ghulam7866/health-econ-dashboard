"""
check_beds_gap_status.py
---------------------------
Diagnoses two issues from the last run:
  1. Q2 2024-25 (2024-09-30) appears to be missing from the extend step's
     success output - check if the file exists and why it might have
     failed silently or been skipped.
  2. beds_clean.csv showed date range ending 2026-03-31 after extend_beds_data.py,
     but combined_quarterly.csv shows Beds ending 2026-01-01 after align_merge.py
     - check what's actually in beds_clean.csv right now.

Run:
    python check_beds_gap_status.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\raw")
PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")

# Check 1: does the Q2 2024-25 file exist?
q2_file = RAW_DIR / "Beds-Open-Overnight-Web_File-Q2-2024-25.xlsx"
print(f"Q2 2024-25 file exists: {q2_file.exists()}")
if q2_file.exists():
    print(f"  Size: {q2_file.stat().st_size / 1024:.0f} KB")
    try:
        df = pd.read_excel(q2_file, sheet_name="Occupied by Specialty", header=14)
        england_row = df[df["Org Name"] == "England"]
        print(f"  'Occupied by Specialty' sheet readable: yes")
        print(f"  England row found: {len(england_row)}")
        if not england_row.empty:
            print(f"  Period End value: {england_row['Period End'].iloc[0]}")
            print(f"  Year value: {england_row['Year'].iloc[0]}")
    except Exception as e:
        print(f"  ⚠ Error reading file: {e}")

# Check 2: what's actually in beds_clean.csv right now?
print(f"\n{'='*60}")
beds = pd.read_csv(PROCESSED_DIR / "beds_clean.csv")
beds["date"] = pd.to_datetime(beds["date"])
beds = beds.sort_values("date")
print("Current beds_clean.csv — last 10 rows:")
print(beds.tail(10).to_string())

print(f"\nFull date list (last 15 quarters):")
print(beds["date"].tail(15).dt.date.tolist())

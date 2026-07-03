"""
diagnose_beds_gap3.py
------------------------
Confirms the full specialty column list and computes the England-row
national total for the Q1 2024-25 file, to sanity-check it lines up with
the existing beds_clean.csv before writing the full parser/append logic.

Run:
    python diagnose_beds_gap3.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\raw")
PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")

path = RAW_DIR / "Beds-Open-Overnight-Web_File-Q1-2024-25-revised.xlsx"
df = pd.read_excel(path, sheet_name="Occupied by Specialty", header=14)

print(f"Total columns: {len(df.columns)}")
print(f"Column names: {list(df.columns)}\n")

# Find the England row (national total)
england_row = df[df["Org Name"] == "England"]
print(f"England row found: {len(england_row)} row(s)")

if not england_row.empty:
    # Sum all specialty columns (everything after Org Name)
    specialty_cols = [c for c in df.columns if c not in
                       ["Year", "Period End", "Region Code", "Org Code", "Org Name"]]
    total_beds = england_row[specialty_cols].sum(axis=1).iloc[0]
    print(f"Specialty columns: {len(specialty_cols)}")
    print(f"England row total occupied beds (summed across specialties): {total_beds:,.1f}")

# Compare to existing processed data's last value
existing = pd.read_csv(PROCESSED_DIR / "beds_clean.csv")
existing["date"] = pd.to_datetime(existing["date"])
print(f"\nExisting beds_clean.csv last 3 rows:")
print(existing.tail(3).to_string())

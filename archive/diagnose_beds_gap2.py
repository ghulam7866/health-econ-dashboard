"""
diagnose_beds_gap2.py
------------------------
Checks the 'Occupied by Specialty' sheet specifically (the one matching
your existing beds_clean.csv structure) across the oldest and newest gap
files, to confirm the layout is consistent before writing the parser.

Run:
    python diagnose_beds_gap2.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\raw")

BEDS_GAP_FILES = [
    "Beds-Open-Overnight-Web_File-Q1-2024-25-revised.xlsx",
    "Beds-Open-Overnight-Web_File-Q4-2025-26.xlsx",
]

SHEET = "Occupied by Specialty"


def main():
    for fname in BEDS_GAP_FILES:
        path = RAW_DIR / fname
        print("=" * 70)
        print(f"{fname}  —  sheet '{SHEET}'")
        print("=" * 70)

        if not path.exists():
            print(f"  ⚠ NOT FOUND: {path}")
            continue

        df = pd.read_excel(path, sheet_name=SHEET, header=None, nrows=25)
        for i, row in df.iterrows():
            vals = [str(v)[:22] for v in row.values[:8]]
            print(f"  Row {i:>2}: {vals}")
        print()


if __name__ == "__main__":
    main()

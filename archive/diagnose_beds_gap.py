"""
diagnose_beds_gap.py
----------------------
Inspects the structure of the 8 newly-downloaded quarterly KH03 'Web_File'
releases, so we can write the right parsing logic to append them onto the
existing beds_clean.csv.

Run:
    python diagnose_beds_gap.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\raw")

BEDS_GAP_FILES = [
    "Beds-Open-Overnight-Web_File-Q1-2024-25-revised.xlsx",
    "Beds-Open-Overnight-Web_File-Q4-2025-26.xlsx",  # also check the most recent
]


def main():
    for fname in BEDS_GAP_FILES:
        path = RAW_DIR / fname
        print("=" * 70)
        print(fname)
        print("=" * 70)

        if not path.exists():
            print(f"  ⚠ NOT FOUND: {path}")
            continue

        xl = pd.ExcelFile(path)
        print(f"  Sheets: {xl.sheet_names}\n")

        # Preview first sheet's raw structure
        first_sheet = xl.sheet_names[0]
        df = pd.read_excel(path, sheet_name=first_sheet, header=None, nrows=20)
        print(f"  First 20 rows of sheet '{first_sheet}':")
        for i, row in df.iterrows():
            vals = [str(v)[:25] for v in row.values[:6]]
            print(f"    Row {i:>2}: {vals}")
        print()


if __name__ == "__main__":
    main()

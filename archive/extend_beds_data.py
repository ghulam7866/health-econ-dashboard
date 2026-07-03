"""
extend_beds_data.py
----------------------
Closes the gap in beds_clean.csv (which stopped at 2024-04-01) by parsing
the 8 quarterly KH03 'Web_File' releases (Q1 2024-25 through Q4 2025-26)
and appending their England-level national totals.

Verified: the England row's summed specialty columns match the existing
beds_clean.csv methodology almost exactly (117,315.0 vs 117,314.9 for the
same quarter) - confirms this is a safe extension, not a different metric.

Run:
    python src/extend_beds_data.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\raw")
PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
BEDS_CLEAN_PATH = PROCESSED_DIR / "beds_clean.csv"

# (filename, period end date) - period end date is the last day of the
# quarter, matching the existing data's convention (e.g. 2024-06-30)
GAP_FILES = [
    ("Beds-Open-Overnight-Web_File-Q1-2024-25-revised.xlsx", "2024-06-30"),
    ("Beds-Open-Overnight-Web_File-Q2-2024-25.xlsx", "2024-09-30"),
    ("Beds-Open-Overnight-Web_File-Q3-2024-25-revised.xlsx", "2024-12-31"),
    ("Beds-Open-Overnight-Web_File-Q4-2024-25.xlsx", "2025-03-31"),
    ("Beds-Open-Overnight-Web_File-Q1-2025-26-revised.xlsx", "2025-06-30"),
    ("Beds-Open-Overnight-Web_File-Q2-2025-26.xlsx", "2025-09-30"),
    ("Beds-Open-Overnight-Web_File-Q3-2025-26.xlsx", "2025-12-31"),
    ("Beds-Open-Overnight-Web_File-Q4-2025-26.xlsx", "2026-03-31"),
]

SHEET = "Occupied by Specialty"
HEADER_ROW = 14  # confirmed via diagnose_beds_gap2.py / diagnose_beds_gap3.py


def extract_england_total(path: Path, period_end: str) -> dict:
    """Reads one quarterly Web_File and returns the England-level national
    total occupied beds for that quarter."""
    df = pd.read_excel(path, sheet_name=SHEET, header=HEADER_ROW)

    england_row = df[df["Org Name"] == "England"]
    if england_row.empty:
        raise ValueError(f"No 'England' row found in {path.name}")

    # Specialty columns = everything except the identifying columns and
    # any trailing 'Unnamed: N' junk columns from blank cells in the sheet
    specialty_cols = [
        c for c in df.columns
        if c not in ["Year", "Period End", "Region Code", "Org Code", "Org Name"]
        and not str(c).startswith("Unnamed")
    ]

    total = england_row[specialty_cols].sum(axis=1).iloc[0]

    return {
        "date": pd.to_datetime(period_end),
        "value": total,
        "metric": "total_occupied_beds_overnight",
        "source": "NHS England KH03",
    }


def main():
    print("=" * 70)
    print("Extending beds_clean.csv with 8 quarterly gap-fill files")
    print("=" * 70)

    new_rows = []
    for fname, period_end in GAP_FILES:
        path = RAW_DIR / fname
        if not path.exists():
            print(f"  ⚠ Missing file, skipping: {fname}")
            continue
        try:
            row = extract_england_total(path, period_end)
            new_rows.append(row)
            print(f"  ✓ {period_end}: {row['value']:,.1f}  ({fname})")
        except Exception as e:
            print(f"  ✗ {fname}: {e}")

    if not new_rows:
        print("\nNo new rows extracted — nothing to append.")
        return

    new_df = pd.DataFrame(new_rows)

    existing = pd.read_csv(BEDS_CLEAN_PATH)
    existing["date"] = pd.to_datetime(existing["date"])

    # Drop any overlapping dates from existing (in case of re-run) before
    # appending, so we don't get duplicates
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "metric"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    combined.to_csv(BEDS_CLEAN_PATH, index=False)

    print(f"\n✓ Saved extended beds_clean.csv → {BEDS_CLEAN_PATH}")
    print(f"  Total rows: {len(combined)}")
    print(f"  Date range: {combined['date'].min().date()} → {combined['date'].max().date()}")


if __name__ == "__main__":
    main()

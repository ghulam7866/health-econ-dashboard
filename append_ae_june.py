"""
append_ae_june.py
------------------
One‑off script: append June 2026 monthly A&E total to the existing
Monthly A&E Time Series Excel file, producing a new cumulative file
that the cleaner can ingest without modification.

Usage:
    python append_ae_june.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\raw")

# ---- File paths ----
OLD_TS = RAW_DIR / "Monthly-AE-Time-Series-May-2026-wlgnE2.xls"
NEW_MONTH = RAW_DIR / "AE_June_2026.csv"
OUTPUT = RAW_DIR / "Monthly-AE-Time-Series-June-2026-merged.xlsx"

# ---- 1. Load old time series (sheet "Activity", header at row 13) ----
df_old = pd.read_excel(OLD_TS, sheet_name="Activity", header=13)

# Rename the date column (index 1) to 'period'
df_old = df_old.rename(columns={df_old.columns[1]: "period"})
df_old["period"] = pd.to_datetime(df_old["period"], errors="coerce")

# ---- 2. Find the total attendances column (robust search) ----
target_col = None
for col in df_old.columns:
    col_str = str(col).strip().lower()
    if "total" in col_str and "attendances" in col_str:
        target_col = col
        break

if target_col is None:
    print("Available columns:")
    for i, col in enumerate(df_old.columns):
        print(f"  [{i}] {col}")
    raise KeyError(
        "Could not find a column containing both 'total' and 'attendances'. "
        "Check the printed column list and update the script accordingly."
    )

print(f"Found total attendances column: '{target_col}'")

# ---- 3. Extract June 2026 total attendances from the new CSV ----
df_june = pd.read_csv(NEW_MONTH)
total_row = df_june[df_june["Org Code"].str.strip() == "Total"]
if total_row.empty:
    raise ValueError("No 'Total' row found in the June CSV")

attend_cols = [
    "A&E attendances Type 1",
    "A&E attendances Type 2",
    "A&E attendances Other A&E Department",
]
june_total = total_row[attend_cols].sum(axis=1).iloc[0]

# ---- 4. Create new row with the exact same columns as df_old ----
new_row = {col: pd.NA for col in df_old.columns}
new_row["period"] = pd.Timestamp("2026-06-01")
new_row[target_col] = june_total

# ---- 5. Append and save as .xlsx, writing header at row 13 (0‑indexed) ----
new_df = pd.DataFrame([new_row])
df_combined = pd.concat([df_old, new_df], ignore_index=True)
df_combined = df_combined.sort_values("period").reset_index(drop=True)

with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    # startrow=13 → column headers are written at row 14 (1‑indexed),
    # which is where the cleaner's header=13 expects them.
    df_combined.to_excel(writer, sheet_name="Activity", index=False, startrow=13)

print(f"\nMerged A&E time series saved to: {OUTPUT}")
print(f"Data now runs from {df_combined['period'].min().date()} to {df_combined['period'].max().date()}")
print(f"Total rows: {len(df_combined)}")
print("Last 3 rows (total attendances only):")
print(df_combined[["period", target_col]].tail(3))
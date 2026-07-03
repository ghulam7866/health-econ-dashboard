"""
diagnose3.py
------------
Follow-up to diagnose2.py — that one only showed the NHS Workforce
'Title sheet' (a table of contents), not actual data, and only checked
PESA sheet 4_1. This looks at the real Workforce data sheet and the
remaining PESA sheets.

Run:
    python diagnose3.py
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\raw")


def preview_sheet(path, sheet_name, nrows=20, ncols=6):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=nrows)
    for i, row in df.iterrows():
        vals = [str(v)[:25] for v in row.values[:ncols]]
        print(f"  Row {i:>2}: {vals}")


# ── NHS Workforce — sheet '1' (HCHS staff by staff group, monthly) ──────
print("=" * 70)
print("NHS WORKFORCE — sheet '1' (HCHS staff by staff group)")
print("=" * 70)
wf_matches = list(RAW_DIR.glob("*Workforce*"))
if wf_matches:
    wf_path = wf_matches[0]
    print(f"File: {wf_path.name}")
    xl = pd.ExcelFile(wf_path)
    if "1" in xl.sheet_names:
        preview_sheet(wf_path, "1")
    else:
        print(f"  No sheet named '1'. Available sheets: {xl.sheet_names}")
else:
    print("NOT FOUND")

# Also preview sheet '4' (HCHS doctors by grade) in case it's more useful
print("\n" + "-" * 70)
print("NHS WORKFORCE — sheet '4' (HCHS doctors by grade) — for comparison")
print("-" * 70)
if wf_matches:
    xl = pd.ExcelFile(wf_path)
    if "4" in xl.sheet_names:
        preview_sheet(wf_path, "4")
    else:
        print(f"  No sheet named '4'.")

# ── HMT PESA — sheets 4_2, 4_3, 4_4 ──────────────────────────────────────
print("\n" + "=" * 70)
print("HMT PESA — sheets 4_2, 4_3, 4_4")
print("=" * 70)
pesa_matches = list(RAW_DIR.glob("PESA*"))
if pesa_matches:
    pesa_path = pesa_matches[0]
    print(f"File: {pesa_path.name}")
    xl = pd.ExcelFile(pesa_path)
    for sheet in ["4_2", "4_3", "4_4"]:
        print(f"\n--- Sheet '{sheet}' ---")
        if sheet in xl.sheet_names:
            preview_sheet(pesa_path, sheet)
        else:
            print(f"  No sheet named '{sheet}'. Available: {xl.sheet_names}")
else:
    print("NOT FOUND")

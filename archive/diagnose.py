"""
diagnose.py
-----------
Prints the first 20 raw rows of the RTT and A&E files so we can
identify the correct header row for the cleaner.

Run:
    python diagnose.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\raw")

# ── RTT ──────────────────────────────────────────────────────────────
rtt_path = next(RAW_DIR.glob("RTT-Overview*"))
print(f"=== RTT: {rtt_path.name} ===")
print(f"Sheets: {pd.ExcelFile(rtt_path).sheet_names}\n")
df = pd.read_excel(rtt_path, header=None, nrows=20, sheet_name=0)
for i, row in df.iterrows():
    vals = [str(v)[:30] for v in row.values[:6]]
    print(f"  Row {i:>2}: {vals}")

# ── A&E ──────────────────────────────────────────────────────────────
ae_path = next(RAW_DIR.glob("Monthly-AE*"))
print(f"\n=== A&E Activity sheet: {ae_path.name} ===")
df2 = pd.read_excel(ae_path, sheet_name="Activity", header=None, nrows=20)
for i, row in df2.iterrows():
    vals = [str(v)[:30] for v in row.values[:6]]
    print(f"  Row {i:>2}: {vals}")

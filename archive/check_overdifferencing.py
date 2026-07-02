"""
check_overdifferencing.py
--------------------------
Fast numerical check (no plotting) for whether d=2 is genuinely needed
for workforce, or whether d=1 was actually sufficient and we'd be
overdifferencing.

Logic: an overdifferenced series typically shows strong NEGATIVE lag-1
autocorrelation (classic signature, often beyond -0.5). We check the
lag-1 ACF of both the first and second difference, plus the variance of
each - if differencing again increases variance, that's also a sign
we've gone too far (proper differencing should reduce variance until
stationary, then differencing further adds noise back in).

Run:
    python src/check_overdifferencing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import acf

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

METRIC = "FTE: All staff groups - All staff groups"
LABEL = "Workforce FTE (level)"


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])
    sub = df[df["metric"] == METRIC].sort_values("quarter")
    level = sub["value"].reset_index(drop=True)

    diff1 = level.diff().dropna()
    diff2 = diff1.diff().dropna()

    print(f"--- {LABEL}: overdifferencing check ---\n")

    for name, series in [("First difference", diff1), ("Second difference", diff2)]:
        lag1_acf = acf(series, nlags=1)[1]
        variance = series.var()
        print(f"{name}:")
        print(f"  Lag-1 ACF: {lag1_acf:+.3f}  "
              f"{'<- OVERDIFFERENCING SIGNAL (strongly negative)' if lag1_acf < -0.5 else '(no overdifferencing signal)'}")
        print(f"  Variance:  {variance:,.1f}")
        print()

    var1 = diff1.var()
    var2 = diff2.var()
    print(f"Variance comparison: diff1={var1:,.1f}  →  diff2={var2:,.1f}")
    if var2 < var1:
        print("  Variance DECREASED with second difference → d=2 likely genuinely needed.")
    else:
        print("  Variance INCREASED with second difference → likely overdifferencing, d=1 may be enough.")


if __name__ == "__main__":
    main()

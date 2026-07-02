"""
check_ae_widening_ci.py
-------------------------
Confirms A&E's confidence interval WIDENS over the 24-quarter horizon
even though the point forecast is flat - this is the expected behaviour
of a random walk (d=1, no AR/MA/exog terms): the point forecast stays at
the last observed value, but uncertainty accumulates with each step ahead.

Run:
    python check_ae_widening_ci.py
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
FORECAST_PATH = PROCESSED_DIR / "forecast_6yr.csv"


def main():
    fc = pd.read_csv(FORECAST_PATH)
    fc["quarter"] = pd.to_datetime(fc["quarter"])

    ae = fc[fc["metric"] == "A&E attendances (flow)"].sort_values("quarter").reset_index(drop=True)
    ae["ci_width"] = ae["ci_upper"] - ae["ci_lower"]

    print("=" * 70)
    print("A&E attendances — confidence interval width by quarter")
    print("=" * 70)
    print(ae[["quarter", "forecast", "ci_lower", "ci_upper", "ci_width"]].to_string(index=False))

    first_width = ae["ci_width"].iloc[0]
    last_width = ae["ci_width"].iloc[-1]
    print(f"\nCI width Q1:  {first_width:,.0f}")
    print(f"CI width Q24: {last_width:,.0f}")
    print(f"Ratio (Q24/Q1): {last_width/first_width:.2f}x")

    if last_width > first_width * 1.5:
        print("\n✓ CI widens substantially — random walk uncertainty behaving as expected.")
    elif last_width > first_width:
        print("\n✓ CI widens, though modestly — still consistent with random walk behaviour.")
    else:
        print("\n⚠ CI is NOT widening — worth investigating, this would be unexpected for a (0,1,0) model.")


if __name__ == "__main__":
    main()

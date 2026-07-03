"""
test_beds_significance.py
----------------------------
Significance test for Bed occupancy's COVID dummies, same logic as
test_covid_significance.py but isolated to this one series since it's
being added after the original three were already done.

Run:
    python src/test_beds_significance.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exog_config import METRIC_NAMES

LABEL = "Bed occupancy (level)"


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])
    metric = METRIC_NAMES[LABEL]
    sub = df[df["metric"] == metric].sort_values("quarter").reset_index(drop=True)

    print(f"--- {LABEL} ---")
    print(f"  n obs: {len(sub)}\n")

    sub["trend"] = np.arange(len(sub))
    sub["trend_sq"] = sub["trend"] ** 2

    # Linear trend
    X_lin = sm.add_constant(sub[["trend", "covid_pulse", "post_covid_regime"]])
    model_lin = sm.OLS(sub["value"], X_lin, missing="drop").fit()

    # Quadratic trend (robustness check, same as workforce got)
    X_quad = sm.add_constant(sub[["trend", "trend_sq", "covid_pulse", "post_covid_regime"]])
    model_quad = sm.OLS(sub["value"], X_quad, missing="drop").fit()

    print(f"  {'Term':<20} {'Linear p':>12} {'Quadratic p':>14}  Verdict")
    for term in ["covid_pulse", "post_covid_regime"]:
        p_lin = model_lin.pvalues.get(term, np.nan)
        p_quad = model_quad.pvalues.get(term, np.nan)
        sig_lin = p_lin < 0.05
        sig_quad = p_quad < 0.05

        if sig_lin and sig_quad:
            verdict = "ROBUST — keep"
        elif sig_lin and not sig_quad:
            verdict = "FRAGILE — likely trend artefact"
        elif not sig_lin and sig_quad:
            verdict = "Emerges with quadratic — investigate"
        else:
            verdict = "Not significant either way — drop"

        print(f"  {term:<20} {p_lin:>12.4f} {p_quad:>14.4f}  {verdict}")

    print(f"\n  R² linear: {model_lin.rsquared:.4f}  |  R² quadratic: {model_quad.rsquared:.4f}")
    print(f"  trend_sq p-value: {model_quad.pvalues.get('trend_sq', np.nan):.4f} "
          f"({'curvature matters' if model_quad.pvalues.get('trend_sq', 1) < 0.05 else 'no meaningful curvature'})")


if __name__ == "__main__":
    main()

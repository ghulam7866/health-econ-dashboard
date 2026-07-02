"""
test_covid_quadratic_check.py
-------------------------------
Robustness check on the COVID significance results: re-runs the OLS with
a quadratic trend term (trend + trend^2) instead of linear-only, to see
if covid_pulse / post_covid_regime significance survives once the model
can capture mild curvature in the underlying trend.

Logic: if a dummy's significance disappears once trend^2 is added, that
dummy was likely just absorbing trend misspecification, not a genuine
COVID effect. If it survives, the effect is more likely real.

Run:
    python src/test_covid_quadratic_check.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

KEY_METRICS = {
    "RTT waiting list (level)": "Incomplete RTT pathways - Total waiting (mil) with estimates for missing data",
    "A&E attendances (flow)": "total_attendances",
    "Workforce FTE (level)": "FTE: All staff groups - All staff groups",
}


def test_series(df: pd.DataFrame, metric_name: str, label: str):
    sub = df[df["metric"] == metric_name].sort_values("quarter").copy()
    if sub.empty:
        print(f"  ⚠ {label}: no data found")
        return None

    sub = sub.reset_index(drop=True)
    sub["trend"] = np.arange(len(sub))
    sub["trend_sq"] = sub["trend"] ** 2

    print(f"\n--- {label} ---")
    print(f"  n obs: {len(sub)}")

    # Linear trend (original model, for comparison)
    X_lin = sm.add_constant(sub[["trend", "covid_pulse", "post_covid_regime"]])
    model_lin = sm.OLS(sub["value"], X_lin, missing="drop").fit()

    # Quadratic trend (robustness check)
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
            verdict = "FRAGILE — likely trend artefact, consider dropping"
        elif not sig_lin and sig_quad:
            verdict = "Emerges with quadratic — investigate"
        else:
            verdict = "Not significant either way — drop"

        print(f"  {term:<20} {p_lin:>12.4f} {p_quad:>14.4f}  {verdict}")

    print(f"  R² linear: {model_lin.rsquared:.4f}  |  R² quadratic: {model_quad.rsquared:.4f}")
    print(f"  trend_sq p-value: {model_quad.pvalues.get('trend_sq', np.nan):.4f} "
          f"({'curvature matters' if model_quad.pvalues.get('trend_sq', 1) < 0.05 else 'no meaningful curvature'})")


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    print("=" * 75)
    print("QUADRATIC TREND ROBUSTNESS CHECK")
    print("=" * 75)

    for label, metric in KEY_METRICS.items():
        test_series(df, metric, label)

    print("\n" + "=" * 75)
    print("Done. Use the 'Verdict' column to decide final exog per series.")
    print("=" * 75)


if __name__ == "__main__":
    main()

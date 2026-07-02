"""
test_covid_significance.py
---------------------------
Tests whether covid_pulse, post_covid_regime, and post_covid_trend_break
are statistically significant regressors for each key metric, using a
simple OLS against a deterministic trend + the three dummies.

This is diagnostic only - it tells you which dummy(ies) to keep in the
SARIMAX exog matrix for each series. It does NOT do the final forecasting
model itself (that's the next step).

Logic per series (piecewise-linear / segmented trend spec):
    value ~ trend + covid_pulse + post_covid_regime + post_covid_trend_break

What each term actually tests:
    - trend:                  baseline (pre-break) slope
    - covid_pulse:             temporary deviation during the acute window
    - post_covid_regime:       permanent LEVEL shift at the break (a step)
    - post_covid_trend_break:  change in SLOPE after the break (a ramp) -
                                 this is what actually tests "did the
                                 growth rate change", which a level dummy
                                 alone cannot test.

If a coefficient's p-value > 0.05, that term is not pulling its weight
for that series and can reasonably be dropped from its SARIMAX exog.

Also prints the correlation between covid_pulse and post_covid_regime per
series - if their date windows overlap, the two can become hard for OLS
to tell apart, making either coefficient unstable to small changes in the
pulse window (this is what happened to A&E's post_covid_regime result
when PULSE_END moved by one quarter).

Run:
    python src/test_covid_significance.py
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

# Add/edit metrics here as your key series list grows
KEY_METRICS = {
    "RTT waiting list (level)": "Incomplete RTT pathways - Total waiting (mil) with estimates for missing data",
    "A&E attendances (flow)": "total_attendances",
    "Workforce FTE (level)": "FTE: All staff groups - All staff groups",
}

DUMMY_COLS = ["covid_pulse", "post_covid_regime", "post_covid_trend_break"]


def test_series(df: pd.DataFrame, metric_name: str, label: str) -> dict:
    sub = df[df["metric"] == metric_name].sort_values("quarter").copy()
    if sub.empty:
        print(f"  ⚠ {label}: no data found for metric '{metric_name}'")
        return None
    sub = sub.reset_index(drop=True)
    sub["trend"] = np.arange(len(sub))

    X = sub[["trend"] + DUMMY_COLS]
    X = sm.add_constant(X)
    y = sub["value"]
    model = sm.OLS(y, X, missing="drop").fit()

    pvals = {col: model.pvalues.get(col, np.nan) for col in DUMMY_COLS}
    coefs = {col: model.params.get(col, np.nan) for col in DUMMY_COLS}

    print(f"\n--- {label} ---")
    print(f"  n obs: {len(sub)}")
    for col in DUMMY_COLS:
        sig = "SIGNIFICANT" if pvals[col] < 0.05 else "not significant"
        print(f"  {col:<23} coef={coefs[col]:>14,.1f}  p={pvals[col]:.4f}  {sig}")
    print(f"  R-squared: {model.rsquared:.4f}")

    # Collinearity check - overlapping pulse/regime windows can make OLS
    # unable to cleanly separate the two, so either coefficient can swing
    # with small changes to the pulse window boundaries.
    corr = sub["covid_pulse"].corr(sub["post_covid_regime"])
    print(f"  corr(covid_pulse, post_covid_regime): {corr:.3f}"
          f"{'  ⚠ high overlap — treat both coefficients as unstable' if abs(corr) > 0.5 else ''}")

    return {
        "metric": label,
        "pulse_pvalue": pvals["covid_pulse"],
        "pulse_significant": pvals["covid_pulse"] < 0.05,
        "regime_pvalue": pvals["post_covid_regime"],
        "regime_significant": pvals["post_covid_regime"] < 0.05,
        "trend_break_pvalue": pvals["post_covid_trend_break"],
        "trend_break_significant": pvals["post_covid_trend_break"] < 0.05,
        "pulse_regime_corr": corr,
        "r_squared": model.rsquared,
    }


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    missing_cols = [c for c in DUMMY_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns {missing_cols}. Run add_intervention_dummies.py first."
        )

    print("=" * 70)
    print("COVID DUMMY SIGNIFICANCE TEST (OLS, piecewise-linear trend spec)")
    print("=" * 70)

    results = []
    for label, metric in KEY_METRICS.items():
        r = test_series(df, metric, label)
        if r:
            results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY — recommended exog per series")
    print("=" * 70)
    for r in results:
        keep = []
        if r["pulse_significant"]:
            keep.append("covid_pulse")
        if r["regime_significant"]:
            keep.append("post_covid_regime")
        if r["trend_break_significant"]:
            keep.append("post_covid_trend_break")
        keep_str = ", ".join(keep) if keep else "(none — drop all)"
        print(f"  {r['metric']}: {keep_str}")
        if abs(r["pulse_regime_corr"]) > 0.5:
            print(f"    ⚠ covid_pulse/post_covid_regime correlation = {r['pulse_regime_corr']:.3f} "
                  f"— recheck this result if the pulse window changes")

    out = pd.DataFrame(results)
    out_path = PROCESSED_DIR / "covid_significance_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\n✓ Saved results → {out_path}")


if __name__ == "__main__":
    main()
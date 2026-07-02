"""
refit_compare_exog.py
-----------------------
Targeted check: refits RTT and A&E WITHOUT their now-insignificant exog
term (found in fit_forecast.py), compares AICc and coefficient stability
against the original spec, and flags numerical instability (suspiciously
tiny standard errors / huge std errs relative to coefficient size).

This resolves whether the OLS-significant dummies genuinely belong in
the SARIMAX exog, or whether the model's own AR/MA dynamics already
captured that information once given the chance.

Run:
    python src/refit_compare_exog.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exog_config import METRIC_NAMES

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

# (label, order, seasonal_order, exog_full, exog_reduced)
COMPARISONS = [
    (
        "RTT waiting list (level)",
        (0, 1, 3), (0, 0, 0, 4),
        ["covid_pulse", "post_covid_regime"],   # full (original)
        ["covid_pulse"],                          # reduced (drop regime)
    ),
    (
        "A&E attendances (flow)",
        (0, 1, 0), (0, 0, 0, 4),
        ["covid_pulse"],                          # full (original)
        [],                                        # reduced (drop pulse entirely)
    ),
]


def fit_one(y, X, order, seasonal_order, label):
    """Fit a SARIMAX, return key diagnostics. X can be None or empty df."""
    exog = X if (X is not None and len(X.columns) > 0) else None
    model = SARIMAX(
        y, exog=exog, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)

    lb = acorr_ljungbox(fitted.resid, lags=[4], return_df=True)
    lb_p = lb["lb_pvalue"].iloc[0]

    # Flag numerically unstable coefficients: std err implausibly small
    # relative to coefficient (ratio < 1e-4) or implausibly large (>10x coef)
    instability_flags = []
    if exog is not None:
        for col in exog.columns:
            coef = fitted.params.get(col, np.nan)
            se = fitted.bse.get(col, np.nan)
            if abs(coef) > 0 and not np.isnan(se):
                ratio = se / abs(coef)
                if ratio < 1e-4:
                    instability_flags.append(f"{col}: SE/coef={ratio:.2e} (suspiciously tiny)")
                elif ratio > 10:
                    instability_flags.append(f"{col}: SE/coef={ratio:.2e} (suspiciously huge)")

    return {
        "aic": fitted.aic,
        "aicc": fitted.aic + (2 * fitted.df_model * (fitted.df_model + 1)) /
                (fitted.nobs - fitted.df_model - 1) if fitted.nobs - fitted.df_model - 1 > 0 else np.nan,
        "ljung_box_p": lb_p,
        "params": fitted.params.to_dict(),
        "pvalues": fitted.pvalues.to_dict(),
        "bse": fitted.bse.to_dict(),
        "instability_flags": instability_flags,
        "fitted": fitted,
    }


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    for label, order, seasonal_order, exog_full, exog_reduced in COMPARISONS:
        metric = METRIC_NAMES[label]
        sub = df[df["metric"] == metric].sort_values("quarter").reset_index(drop=True)
        y = sub["value"]

        print(f"\n{'='*72}")
        print(f"{label}  —  order={order}{seasonal_order}")
        print(f"{'='*72}")

        X_full = sub[exog_full] if exog_full else None
        X_reduced = sub[exog_reduced] if exog_reduced else None

        full = fit_one(y, X_full, order, seasonal_order, label)
        reduced = fit_one(y, X_reduced, order, seasonal_order, label)

        print(f"\n  FULL exog={exog_full}")
        print(f"    AIC={full['aic']:.2f}  AICc={full['aicc']:.2f}  Ljung-Box p={full['ljung_box_p']:.4f}")
        if full["instability_flags"]:
            print(f"    ⚠ Instability flags: {full['instability_flags']}")
        else:
            print(f"    No instability flags")

        print(f"\n  REDUCED exog={exog_reduced or '(none)'}")
        print(f"    AIC={reduced['aic']:.2f}  AICc={reduced['aicc']:.2f}  Ljung-Box p={reduced['ljung_box_p']:.4f}")
        if reduced["instability_flags"]:
            print(f"    ⚠ Instability flags: {reduced['instability_flags']}")
        else:
            print(f"    No instability flags")

        delta_aicc = full["aicc"] - reduced["aicc"]
        print(f"\n  ΔAICc (full - reduced): {delta_aicc:+.2f}")
        if delta_aicc > 2:
            print(f"  → REDUCED model preferred (full model's extra term isn't earning its AICc penalty)")
        elif delta_aicc < -2:
            print(f"  → FULL model preferred (extra term meaningfully improves fit)")
        else:
            print(f"  → Models roughly equivalent (within 2 AICc) — prefer REDUCED for parsimony")

    print(f"\n{'='*72}")
    print("Done. Use ΔAICc + instability flags to decide final exog spec.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

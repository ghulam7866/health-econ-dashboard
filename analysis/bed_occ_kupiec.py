"""
kupiec_bed_occupancy.py
------------------------
Rolling‑origin backtest for bed occupancy (production spec only)
followed by a Kupiec (1995) unconditional coverage (POF) test for
each forecast horizon.

The Kupiec test evaluates whether the empirical proportion of actual
values falling outside the 95% forecast intervals matches the nominal
5% expected under the null hypothesis of correct calibration.

The production specification (order, seasonal, trend, sigma_scale,
t‑based CI) is hard‑coded to match exog_config.py and the master engine.

Usage:
    python kupiec_bed_occupancy.py

Input:
    data/processed/combined_quarterly.csv

Output:
    Console table showing, for each horizon (1–12 quarters):
      - number of backtest folds
      - expected vs actual exceedances
      - empirical coverage
      - Kupiec likelihood‑ratio statistic and p‑value
      - whether H0 (correct coverage) is rejected at α = 0.05
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, t as t_dist, kurtosis
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")

# ── Production specification (matches exog_config.py) ──────
METRIC_KEY = "Bed occupancy (level)"
RAW_NAME   = "total_occupied_beds_overnight"
EXOG_COLS  = ["covid_pulse", "post_covid_regime"]
ORDER      = (1, 1, 1)
SEASONAL   = (1, 0, 1, 4)
TREND      = "c"
SIGMA_SCALE= 1.8                            # empirically calibrated
DF_FLOOR   = 5.0                            # minimum df for t‑distribution
HORIZONS   = [1, 2, 3, 4, 8, 12]
MIN_TRAIN  = 6

# ── Load data ─────────────────────────────────────────────
df = pd.read_csv("data/processed/combined_quarterly.csv", parse_dates=["quarter"])
sub = df[df["metric"] == RAW_NAME].dropna(subset=["value"]).sort_values("quarter")
y_raw = sub["value"].values.astype(float)
exog  = sub[EXOG_COLS].values.astype(float) if EXOG_COLS else None
n = len(y_raw)

# ── Helper: t‑based CI factor (identical to stress_test.py) ──
def get_t_multiplier(resid):
    """
    Estimate t‑distribution degrees of freedom from full‑fit residuals.
    Returns the multiplier for the CI half‑width and the df used.
    Falls back to 1.96 (normal) if estimation fails.
    """
    if len(resid) < 10:
        return 1.96, None
    std_resid = (resid - np.mean(resid)) / np.std(resid, ddof=1)
    try:
        df_t, _, _ = t_dist.fit(std_resid)
        df_use = max(df_t, DF_FLOOR)       # enforce floor
        mult = t_dist.ppf(0.975, df_use) * np.sqrt((df_use - 2) / df_use)
        return mult, df_use
    except:
        return 1.96, None

# Fit full‑history model to obtain the t‑multiplier (needed for CI construction)
if EXOG_COLS:
    exog_full = exog
else:
    exog_full = None
model_full = sm.tsa.SARIMAX(
    y_raw, exog=exog_full, order=ORDER, seasonal_order=SEASONAL,
    trend=TREND, enforce_stationarity=True, enforce_invertibility=True,
    initialization='approximate_diffuse'
)
res_full = model_full.fit(disp=False, method='lbfgs', maxiter=2000)
t_mult, df_used = get_t_multiplier(res_full.resid)
print(f"Estimated t‑distribution df = {df_used:.1f}, multiplier = {t_mult:.4f}")

# ── Rolling‑origin backtest ───────────────────────────────
# For each forecast origin t (from MIN_TRAIN to n‑1) and each horizon h,
# we check whether the actual value at t+h falls inside the 95% CI.
# indicator = 1 if outside, 0 if inside.
exceedances = {h: [] for h in HORIZONS}

for t in range(MIN_TRAIN, n - 1):
    y_train = y_raw[:t+1]
    exog_train = exog_full[:t+1] if exog_full is not None else None
    try:
        model = sm.tsa.SARIMAX(
            y_train, exog=exog_train, order=ORDER, seasonal_order=SEASONAL,
            trend=TREND, enforce_stationarity=True, enforce_invertibility=True,
            initialization='approximate_diffuse'
        )
        res = model.fit(disp=False, method='lbfgs', maxiter=2000)

        # future exogenous: constant extrapolation (same as stress_test)
        if exog_full is not None:
            last_val = exog_full[t]
            last_diff = exog_full[t] - exog_full[t-1] if t > 0 else 0
            steps = np.arange(1, max(HORIZONS)+1).reshape(-1, 1)
            future_exog = last_val + steps * last_diff
        else:
            future_exog = None

        fc = res.get_forecast(steps=max(HORIZONS), exog=future_exog)
        means = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)       # 95% normal‑based CI

        for h in HORIZONS:
            target = t + h
            if target >= n:
                continue
            actual = y_raw[target]
            fc_h = means[h-1]
            # Raw half‑width from normal‑based CI
            se = (ci[h-1, 1] - ci[h-1, 0]) / (2.0 * 1.96)
            # Apply t‑based widening and sigma scale (exactly as stress_test)
            halfwidth = t_mult * se * SIGMA_SCALE
            lo = fc_h - halfwidth
            hi = fc_h + halfwidth
            outside = 1 if (actual < lo or actual > hi) else 0
            exceedances[h].append(outside)
    except Exception as e:
        # If a fold fails (e.g. non‑convergence), skip it
        continue

# ── Kupiec (1995) unconditional coverage test ─────────────
def kupiec_test(n_total, n_exceed, p_expected=0.05):
    """
    Kupiec (1995) unconditional coverage test.

    H0: true exceedance probability = p_expected (usually 0.05 for a 95% CI)
    Under H0, the LR statistic is asymptotically chi‑squared with 1 df.

    Returns (LR_statistic, p_value).
    """
    if n_exceed == 0 or n_exceed == n_total:
        # Edge case: all or none of the actuals fell outside the CI.
        p_hat = n_exceed / n_total
        if p_hat == 0:
            lr = 2 * n_total * np.log(1 - p_expected)   # limit as p_hat → 0
        else:
            lr = 2 * n_total * (np.log(p_expected))      # p_hat = 1
        p_value = 0.0
        return lr, p_value
    p_hat = n_exceed / n_total
    lr = -2 * (n_exceed * np.log(p_expected / p_hat) +
               (n_total - n_exceed) * np.log((1 - p_expected) / (1 - p_hat)))
    p_value = 1 - chi2.cdf(lr, 1)
    return lr, p_value

# ── Print results table ───────────────────────────────────
print("\n" + "=" * 80)
print("HORIZON  N_FOLDS  EXPECTED EXCEED.  ACTUAL EXCEED.  COVERAGE  KUPIEC LR  p‑value  REJECT H0 (α=0.05)?")
print("=" * 80)
for h in HORIZONS:
    ind = exceedances[h]
    n_total = len(ind)
    if n_total == 0:
        print(f"   h={h:<2}   no folds")
        continue
    n_exceed = sum(ind)
    cov = 1 - n_exceed / n_total
    lr, pv = kupiec_test(n_total, n_exceed)
    reject = "YES" if pv < 0.05 else "NO"
    print(f"   h={h:<2}    {n_total:>4}      {n_total*0.05:>5.1f}          {n_exceed:>4}          {cov:>6.4f}   {lr:>8.4f}  {pv:>6.4f}        {reject}")
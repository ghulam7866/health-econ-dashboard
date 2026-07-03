import pandas as pd
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf
from exog_config import METRIC_NAMES

PROCESSED_DIR = Path("data/processed")  # adjust if running from a different cwd
df = pd.read_csv(PROCESSED_DIR / "combined_quarterly.csv")
df["quarter"] = pd.to_datetime(df["quarter"])

ae_display_name = "A&E 12-hour decisions to admit (breach flow)"
ae_raw_id = METRIC_NAMES[ae_display_name]

ae_breach_series = (
    df[df["metric"] == ae_raw_id]
    .dropna(subset=["value"])
    .sort_values("quarter")
    .set_index("quarter")["value"]
)
print(ae_breach_series.head())
print(ae_breach_series.tail())

# --- 1. Isolate the same post-2021 window you're already fitting on ---
ae_post2021 = ae_breach_series.loc["2021-01-01":].dropna()
y = ae_post2021.values.astype(float)
n_obs = len(y)
print(f"A&E post-2021 window: n_obs={n_obs}")

scale = np.nanmax(np.abs(y)) or 1.0
y_scaled = y / scale


# --- 2. Re-run stationarity battery + ACF lag-4 on THIS window specifically ---
def stationarity_report(series, label):
    s = pd.Series(series).dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    print(f"[{label}] n={len(s)}  ADF p={adf_p:.3f}  KPSS p={kpss_p:.3f}")
    return adf_p, kpss_p


print("\n--- Stationarity on fitting window ---")
adf_p, kpss_p = stationarity_report(ae_post2021, "levels")
adf_p_d1, kpss_p_d1 = stationarity_report(ae_post2021.diff().dropna(), "d=1")

acf_vals = acf(ae_post2021, nlags=4, fft=False)
print(f"ACF lag 4 (post-2021): {acf_vals[4]:.3f}")
seasonal_supported = abs(acf_vals[4]) > 0.3
print(f"-> Seasonal signal at this length: {'present' if seasonal_supported else 'weak/absent'}")


# --- 3. Candidate spec comparison: current config vs non-seasonal alternatives ---
def fit_candidate(order, seasonal_order, trend, label):
    try:
        model = sm.tsa.statespace.SARIMAX(
            y_scaled, order=order, seasonal_order=seasonal_order,
            trend=trend, enforce_stationarity=True, enforce_invertibility=True,
        )
        res = model.fit(disp=False)
        resid = res.resid
        resid_adf_p, resid_kpss_p = stationarity_report(pd.Series(resid), f"{label} residuals")
        print(f"[CANDIDATE] {label}: order={order}, seasonal={seasonal_order}, trend={trend} "
              f"-> AICc={res.aicc:.2f}, converged={res.mle_retvals.get('converged', 'n/a')}")
        return {
            "label": label, "order": order, "seasonal_order": seasonal_order,
            "trend": trend, "aicc": res.aicc, "resid_adf_p": resid_adf_p,
            "resid_kpss_p": resid_kpss_p, "res": res,
        }
    except Exception as e:
        print(f"[CANDIDATE FAILED] {label}: {e}")
        return None


print("\n--- Candidate comparison ---")
candidates = [
    fit_candidate((1, 0, 1), (0, 1, 1, 4), "c", "current (seasonal, trend=c)"),
    fit_candidate((1, 0, 1), (0, 0, 0, 4), "c", "no seasonal, trend=c"),
    fit_candidate((1, 0, 1), (0, 0, 0, 4), None, "no seasonal, no trend"),
    fit_candidate((0, 1, 1), (0, 0, 0, 4), None, "diff + MA(1), no seasonal, no trend"),
    fit_candidate((1, 1, 0), (0, 0, 0, 4), None, "diff + AR(1), no seasonal, no trend"),
]

valid = [c for c in candidates if c is not None]
if valid:
    best = min(valid, key=lambda c: c["aicc"])
    print(f"\n[BEST BY AICc] {best['label']} -> AICc={best['aicc']:.2f}")
    print("Compare resid_adf_p across candidates — lower is better (cleaner residuals),")
    print("and check this AICc is in the same ballpark as your other series (RTT -344, Workforce -457, A&E flow -112),")
    print("not the -9.69 you got from the seasonal/trend spec at this window length.")

ar1_candidate = fit_candidate((1, 1, 0), (0, 0, 0, 4), None, "diff + AR(1) final check")
print(ar1_candidate["res"].summary())

# --- 4. Drift candidate: random walk with drift ---
# AR(1) p=0.832 above suggests the level-model structure (AR/MA around a mean)
# isn't capturing what's going on. The post-2021 tail is climbing sharply, so
# a pure random-walk-with-drift spec (which encodes "this series moves by a
# roughly constant amount each quarter" rather than "mean-reverts") is the
# natural next thing to test before deciding anything.
print("\n--- Drift candidate ---")
rw_drift = fit_candidate((0, 1, 0), (0, 0, 0, 4), "c", "random walk with drift")
if rw_drift is not None:
    print(rw_drift["res"].summary())

    # --- 5. Forecast trajectory check ---
    # AICc and residual diagnostics are in-sample; the actual flatline-to-zero
    # problem only shows up in the out-of-sample forecast path. Check it
    # directly rather than inferring it from the fit statistics above.
    print("\n--- Forecast trajectory check (drift candidate) ---")
    horizons = 24
    fc = rw_drift["res"].get_forecast(steps=horizons)
    raw_mean = fc.predicted_mean
    raw_ci = fc.conf_int(alpha=0.05)

    mean_fc = (raw_mean.to_numpy() if hasattr(raw_mean, "to_numpy") else np.asarray(raw_mean)) * scale
    ci = (raw_ci.to_numpy() if hasattr(raw_ci, "to_numpy") else np.asarray(raw_ci)) * scale

    print("Forecast mean (rescaled):")
    print(mean_fc)
    print(f"\nMin of forecast mean: {mean_fc.min():.2f}")
    print(f"Min of lower CI bound: {ci[:, 0].min():.2f}")

    n_would_clamp = int(np.sum((mean_fc < 0) | (ci[:, 0] < 0)))
    if n_would_clamp > 0:
        print(
            f"[STILL UNSTABLE] {n_would_clamp}/{horizons} forecast points would still "
            f"have a negative raw value or CI bound clamped to 0 — drift candidate "
            f"does not fully resolve the flatline issue."
        )
    else:
        print(
            f"[OK] All {horizons} forecast points (mean and lower CI) stay positive — "
            f"drift candidate avoids the clamp-to-zero flatline seen in the current config."
        )
else:
    print("[SKIP] Drift candidate failed to fit — see error above.")

# --- 6. Forecast trajectory check on the CURRENT production config ---
# We've only looked at in-sample diagnostics for this spec so far. Check
# whether the mean forecast itself actually crashes toward zero, or whether
# the original flatline was a CI-band clamping artifact instead.
print("\n--- Forecast trajectory check (CURRENT production config) ---")
current_candidate = candidates[0]  # "current (seasonal, trend=c)" = (1,0,1)/(0,1,1,4)/trend=c
if current_candidate is not None:
    horizons = 24
    fc_current = current_candidate["res"].get_forecast(steps=horizons)
    raw_mean_current = fc_current.predicted_mean
    raw_ci_current = fc_current.conf_int(alpha=0.05)

    mean_current = (raw_mean_current.to_numpy() if hasattr(raw_mean_current, "to_numpy")
                     else np.asarray(raw_mean_current)) * scale
    ci_current = (raw_ci_current.to_numpy() if hasattr(raw_ci_current, "to_numpy")
                   else np.asarray(raw_ci_current)) * scale

    print("Forecast mean (rescaled):")
    print(mean_current)
    print(f"\nMin of forecast mean: {mean_current.min():.2f}")
    print(f"Min of lower CI bound: {ci_current[:, 0].min():.2f}")

    n_would_clamp_current = int(np.sum((mean_current < 0) | (ci_current[:, 0] < 0)))
    if n_would_clamp_current > 0:
        print(
            f"[CLAMP TRIGGERED] {n_would_clamp_current}/{horizons} forecast points "
            f"would have a negative raw mean or CI bound clamped to 0 under the "
            f"current production config."
        )
        print(
            "Check whether it's the MEAN going negative (real flatline problem) "
            "or just the LOWER CI BOUND (variance-growth artifact, different issue)."
        )
    else:
        print("[OK] Current config forecast stays fully positive across the horizon.")
else:
    print("[SKIP] Current candidate failed to fit — see error above.")

# --- 7. Shortened-horizon check on the drift candidate ---
# 24 quarters from 22 observations is a long extrapolation regardless of
# which spec wins. Check how much tighter the CI band gets at horizons
# closer to what you already use for the GP series (horizons=3) — a shorter
# horizon may make the clamping problem disappear entirely without needing
# a different model.
print("\n--- Shortened horizon check (drift candidate) ---")
if rw_drift is not None:
    for h in [3, 6, 8, 12]:
        fc_h = rw_drift["res"].get_forecast(steps=h)
        ci_h_raw = fc_h.conf_int(alpha=0.05)
        mean_h = (fc_h.predicted_mean.to_numpy() if hasattr(fc_h.predicted_mean, "to_numpy")
                  else np.asarray(fc_h.predicted_mean)) * scale
        ci_h = (ci_h_raw.to_numpy() if hasattr(ci_h_raw, "to_numpy")
                else np.asarray(ci_h_raw)) * scale
        n_clamp_h = int(np.sum((mean_h < 0) | (ci_h[:, 0] < 0)))
        print(
            f"horizons={h}: min mean={mean_h.min():.2f}, min lower CI={ci_h[:, 0].min():.2f}, "
            f"would_clamp={n_clamp_h}/{h}"
        )
        print("\n--- Shortened horizon check (CURRENT production config) ---")
    for h in [6, 8, 11, 16, 22]:
        fc_h = current_candidate["res"].get_forecast(steps=h)
        ci_h = fc_h.conf_int(alpha=0.05).to_numpy() * scale if hasattr(fc_h.conf_int(alpha=0.05), "to_numpy") else np.asarray(fc_h.conf_int(alpha=0.05)) * scale
        mean_h = fc_h.predicted_mean.to_numpy() * scale if hasattr(fc_h.predicted_mean, "to_numpy") else np.asarray(fc_h.predicted_mean) * scale
        n_clamp_h = int(np.sum((mean_h < 0) | (ci_h[:, 0] < 0)))
        print(f"horizons={h}: min lower CI={ci_h[:,0].min():.2f}, would_clamp={n_clamp_h}/{h}")


# --- 8. Full-history retest of the CURRENT production spec ---
# Everything above was fit on the post-2021-only window (n=22). This section
# refits the exact same spec (order=(1,0,1), seasonal=(0,1,1,4), trend='c')
# on the FULL historical series instead, to check whether trend='c' alone
# resolves the original flatline-to-zero problem, or whether the post-2021
# restriction is actually doing necessary work independent of the trend fix.

print("\n--- Full-history retest (same spec, full series instead of post-2021) ---")

y_full = ae_breach_series.dropna().values.astype(float)
n_full = len(y_full)
print(f"Full A&E breach-flow series: n_obs={n_full}")

scale_full = np.nanmax(np.abs(y_full)) or 1.0
y_full_scaled = y_full / scale_full

try:
    model_full = sm.tsa.statespace.SARIMAX(
        y_full_scaled,
        order=(1, 0, 1),
        seasonal_order=(0, 1, 1, 4),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    res_full = model_full.fit(disp=False)
    print(f"[FULL-HISTORY FIT] AICc={res_full.aicc:.2f}, converged={res_full.mle_retvals.get('converged', 'n/a')}")

    resid_full = res_full.resid
    stationarity_report(pd.Series(resid_full), "full-history residuals")

    horizons_full = 24
    fc_full = res_full.get_forecast(steps=horizons_full)
    raw_mean_full = fc_full.predicted_mean
    raw_ci_full = fc_full.conf_int(alpha=0.05)

    mean_full = (raw_mean_full.to_numpy() if hasattr(raw_mean_full, "to_numpy")
                 else np.asarray(raw_mean_full)) * scale_full
    ci_full = (raw_ci_full.to_numpy() if hasattr(raw_ci_full, "to_numpy")
               else np.asarray(raw_ci_full)) * scale_full

    print("Forecast mean (rescaled):")
    print(mean_full)
    print(f"\nMin of forecast mean: {mean_full.min():.2f}")
    print(f"Min of lower CI bound: {ci_full[:, 0].min():.2f}")

    n_clamp_full = int(np.sum((mean_full < 0) | (ci_full[:, 0] < 0)))
    if n_clamp_full > 0:
        print(
            f"[STILL FLATLINES ON FULL HISTORY] {n_clamp_full}/{horizons_full} forecast "
            f"points would be clamped to 0 even with trend='c' on the full series — "
            f"confirms the post-2021 restriction is necessary, not just one valid option."
        )
    else:
        print(
            f"[RESOLVED ON FULL HISTORY TOO] trend='c' alone fixes the flatline on the "
            f"full series — post-2021 restriction is not strictly required for this spec, "
            f"though may still be preferable for other reasons (regime relevance, etc.)."
        )
except Exception as e:
    print(f"[FULL-HISTORY FIT FAILED] {e}")
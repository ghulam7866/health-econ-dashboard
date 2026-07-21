"""
exog_config.py
--------------
Single source of truth for exogenous variables and model specifications.

This file is imported by both the forecast engine and the backtesting
framework (`stress_test.py`).  Every production model's ARIMA order,
seasonal order, trend, transformation, CI method, and sigma scale is
defined here, ensuring that the engine and stress test always agree.

Last updated: 2026‑07‑09  (corrected Doctor FTE metric name; Nurse FTE added)
"""

# Which exogenous columns each series uses in its SARIMAX model.
# An empty list means no exogenous regressors.
EXOG_CONFIG = {
    "RTT waiting list (level)": ["post_covid_trend_break"],
    "A&E attendances (flow)": [],
    "Workforce FTE (level)": ["post_covid_trend_break", "covid_pulse"],
    "Nurse FTE (level)": ["post_covid_trend_break"],
    "Doctor FTE (level)": ["post_covid_trend_break"],
    "Bed occupancy (level)": ["covid_pulse", "post_covid_regime"],
    "PESA Health spend (level)": [],
    "RTT % within 18 weeks (performance)": ["post_covid_trend_break"],
    "A&E 12-hour decisions to admit (breach flow)": [],
    "GP total appointments (flow)": [],
    "GP face-to-face appointments (flow)": [],
    "GP telephone appointments (flow)": [],
}

# Full model specification for each metric.
MODEL_CONFIG = {
    "RTT waiting list (level)": {
        "order": (0, 1, 1),
        "seasonal_order": (1, 0, 1, 4),
        "trend": "n",
        "transform": "log",
        "model_type": "ARIMA_MA_no_exog",
        "ci_method": "t",            # t‑based prediction intervals
        "t_df_floor": 5.0,
        "sigma_scale": 1.25,
        "notes": "Log MA(1), no constant. t‑based CI, sigma_scale=1.25. Coverage 0.91–0.94.",
    },

    "A&E attendances (flow)": {
        "order": (1, 1, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "c",
        "model_type": "ARIMA_diff_AR1_no_exog",
        "ci_method": "t",
        "t_df_floor": 5.0,
        "sigma_scale": 1.5,
        "notes": "Differenced AR(1) with constant. t‑based CI, sigma_scale=1.5. Coverage 0.91–0.98.",
    },

    "Workforce FTE (level)": {
        "order": (2, 1, 0),
        "seasonal_order": (1, 0, 1, 4),
        "trend": "n",
        "model_type": "ARIMA_stable_no_exog",
        "ci_method": "t",
        "t_df_floor": 5.0,
        "sigma_scale": 1.45,
        "notes": "AR(2) no exog. t‑based CI, sigma_scale=1.45. Coverage 0.93–0.97.",
    },

   "Nurse FTE (level)": {
        "order": (2, 1, 0),
        "seasonal_order": (1, 0, 1, 4),
        "trend": "n",                # confirmed by trend comparison in backtest
        "transform": "log",
        "model_type": "ARIMA_stable_no_exog",
        "ci_method": "t",
        "t_df_floor": 5.0,
        "sigma_scale": 1.4,
        "notes": "Trend='n' per backtest. Coverage 0.95‑1.00. Bias –9118 at h=12 noted. Near‑unit‑root AR and lag‑4 Ljung‑Box flagged.",
   },

   "Doctor FTE (level)": {
        "order": (2, 1, 0),
        "seasonal_order": (2, 0, 1, 4),
        "trend": "c",                # h=12 coverage 0.94 for both trends, but 'c' has lower bias
        "transform": "log",
        "model_type": "ARIMA_stable_no_exog",
        "ci_method": "t",
        "t_df_floor": 5.0,
        "sigma_scale": 1.8,
        "notes": "Log transform + trend='c'. sigma_scale=1.8. Coverage 0.92‑0.98. Lag‑4 Ljung‑Box flagged.",
    },

    "Bed occupancy (level)": {
        "order": (1, 1, 1),
        "seasonal_order": (1, 0, 1, 4),
        "trend": "n",
        "model_type": "ARIMA_stable_calibrated",
        "ci_method": "t",
        "t_df_floor": 5.0,
        "sigma_scale": 1.8,
        "notes": "t‑based CI, sigma_scale=1.8. Data extended to 2026‑01‑01.",
    },

    "PESA Health spend (level)": {
        "order": (0, 1, 0),
        "seasonal_order": (0, 0, 0, 1),
        "trend": "n",
        "model_type": "ARIMA_random_walk",
    },

    "RTT % within 18 weeks (performance)": {
        "order": (1, 0, 1),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "n",
        "model_type": "ARIMA_logit_mean_reverting",
        "horizons": 24,
        "sigma_scale": 1.25,
        "notes": "Logit ARMA(1,1), no constant. sigma_scale=1.25. Coverage 0.94–0.97.",
    },

    "A&E 12-hour decisions to admit (breach flow)": {
        "order": (2, 1, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": None,              # no constant – trend handled by differencing
        "model_type": "ARIMA_diff_2lag_no_exog",
        "horizons": 8,
        "notes": "Restricted window (2021+). Small sample (22 obs).",
    },

    "GP total appointments (flow)": {
        "order": (0, 0, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "n",
        "model_type": "ARIMA_mean",
        "horizons": 3,
    },
    "GP face-to-face appointments (flow)": {
        "order": (0, 0, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "n",
        "model_type": "ARIMA_mean",
        "horizons": 3,
    },
    "GP telephone appointments (flow)": {
        "order": (0, 0, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "n",
        "model_type": "ARIMA_mean",
        "horizons": 3,
    },
}

# Overrides for the start of the training window (e.g., to exclude pre‑2021
# data for series where earlier data is not comparable).
FIT_START_OVERRIDES = {
    "A&E 12-hour decisions to admit (breach flow)": "2021-01-01",
}

# Mapping from the display name used throughout the codebase to the raw
# metric name as it appears in `combined_quarterly.csv`.
METRIC_NAMES = {
    "RTT waiting list (level)": "Incomplete RTT pathways - Total waiting (mil) with estimates for missing data",
    "A&E attendances (flow)": "total_attendances",
    "Workforce FTE (level)": "FTE: All staff groups - All staff groups",
    "Nurse FTE (level)": "FTE: Professionally qualified clinical staff - Nurses & health visitors",
    "Doctor FTE (level)": "FTE: Professionally qualified clinical staff - HCHS doctors - All grades",
    "Bed occupancy (level)": "total_occupied_beds_overnight",
    "PESA Health spend (level)": "7. Health (real_gbp_bn)",
    "RTT % within 18 weeks (performance)": "Incomplete RTT pathways - % within 18 weeks",
    "A&E 12-hour decisions to admit (breach flow)": "number_of_patients_spending_12_hours_from_decision_to_admit_to_admission",
    "GP total appointments (flow)": "total_attended_appointments",
    "GP face-to-face appointments (flow)": "attended_face_to_face",
    "GP telephone appointments (flow)": "attended_telephone",
}

# Informal notes – not used by the engine, but helpful for human readers.
NOTES = {
    "RTT waiting list (level)": "Log MA(1), no constant. t‑based CI, sigma_scale=1.25. Coverage 0.91–0.94.",
    "A&E attendances (flow)": "Differenced AR(1) with constant. t‑based CI, sigma_scale=1.5. Coverage 0.91–0.98.",
    "Workforce FTE (level)": "AR(2) no exog. t‑based CI, sigma_scale=1.45. Coverage 0.93–0.97.",
    "Nurse FTE (level)": "Nurse FTE. Same spec as total Workforce FTE. sigma_scale=1.0 initially.",
    "Doctor FTE (level)": "Doctor FTE. Coverage currently 0.49 at h=8 – needs spec review and sigma_scale calibration.",
    "Bed occupancy (level)": "AR(1) no exog, t‑based CI, sigma_scale=1.8.",
    "PESA Health spend (level)": "Annual random walk.",
    "RTT % within 18 weeks (performance)": "Logit ARMA(1,1), no constant. sigma_scale=1.25. Coverage 0.94–0.97.",
    "A&E 12-hour decisions to admit (breach flow)": "Diff AR(2), restricted to 2021+. Small‑sample caveat.",
    "GP total appointments (flow)": "Historical only.",
    "GP face-to-face appointments (flow)": "Historical only.",
    "GP telephone appointments (flow)": "Historical only.",
}
"""
exog_config.py
--------------
Single source of truth for exogenous variables and model specifications.

This module contains all configuration for the forecasting pipeline:
- EXOG_CONFIG: Which exogenous variables to use per series
- MODEL_CONFIG: ARIMA/SARIMA orders, trends, and transformations
- METRIC_NAMES: Mapping from display names to raw metric identifiers
- FIT_START_OVERRIDES: Custom fit windows for specific series

Usage:
    from exog_config import EXOG_CONFIG, MODEL_CONFIG, METRIC_NAMES

Last updated: 2026-07-02
"""

EXOG_CONFIG = {
    "RTT waiting list (level)": ["quadratic_trend"],
    "A&E attendances (flow)": [],
    "Workforce FTE (level)": [
        "covid_pulse",
        "post_covid_trend_break",
        "quadratic_trend",
    ],
    "Bed occupancy (level)": [
        "covid_pulse",
        "post_covid_regime",
        "quadratic_trend",
    ],
    "PESA Health spend (level)": [],
    "RTT % within 18 weeks (performance)": [],
    "A&E 12-hour decisions to admit (breach flow)": ["quadratic_trend"],
    "GP total appointments (flow)": [],
    "GP face-to-face appointments (flow)": [],
    "GP telephone appointments (flow)": [],
}

MODEL_CONFIG = {
    "RTT waiting list (level)": {
        "order": (0, 1, 1),
        "seasonal_order": (1, 0, 1, 4),
        "trend": "c",
        "transform": "log",
        "model_type": "ARIMA_MA_with_quadratic_log",
        "notes": "Log transformation improved AICc by 312 points",
    },

    "A&E attendances (flow)": {
        "order": (1, 0, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "c",
        "model_type": "ARIMA_AR1_with_trend",
        "notes": "Simple AR(1) with constant trend - AICc best",
    },

    "Workforce FTE (level)": {
        "order": (2, 1, 0),
        "seasonal_order": (1, 0, 1, 4),
        "trend": "c",
        "model_type": "ARIMA_stable_with_quadratic",
        "notes": "AR(2) with quadratic_trend - all exog significant",
    },

    "Bed occupancy (level)": {
        "order": (0, 1, 1),
        "seasonal_order": (1, 0, 1, 4),
        "trend": "c",
        "model_type": "ARIMA_MA_stable_with_quadratic",
        "notes": "MA(1) with quadratic_trend - stable and well-specified",
    },

    "PESA Health spend (level)": {
        "order": (0, 1, 0),
        "seasonal_order": (0, 0, 0, 1),
        "trend": "n",
        "model_type": "ARIMA_random_walk",
        "notes": "Annual series - random walk with no drift",
    },

    "RTT % within 18 weeks (performance)": {
        "order": (0, 1, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "c",
        "model_type": "ARIMA_random_walk_with_drift",
        "notes": "Random walk with drift - smoothed transition enforced",
    },

    "A&E 12-hour decisions to admit (breach flow)": {
        "order": (0, 1, 0),
        "seasonal_order": (1, 0, 0, 4),
        "trend": None,
        "model_type": "ARIMA_random_walk_with_quadratic",
        "notes": "Random walk with quadratic_trend - post-2021 only",
    },

    "GP total appointments (flow)": {
        "order": (0, 0, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "n",
        "model_type": "ARIMA_mean",
        "horizons": 3,
        "notes": "Insufficient data for forecasting - historical only",
    },

    "GP face-to-face appointments (flow)": {
        "order": (0, 0, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "n",
        "model_type": "ARIMA_mean",
        "horizons": 3,
        "notes": "Insufficient data for forecasting - historical only",
    },

    "GP telephone appointments (flow)": {
        "order": (0, 0, 0),
        "seasonal_order": (0, 0, 0, 4),
        "trend": "n",
        "model_type": "ARIMA_mean",
        "horizons": 3,
        "notes": "Insufficient data for forecasting - historical only",
    },
}

FIT_START_OVERRIDES = {
    "A&E 12-hour decisions to admit (breach flow)": "2021-01-01",
}

METRIC_NAMES = {
    "RTT waiting list (level)": "Incomplete RTT pathways - Total waiting (mil) with estimates for missing data",
    "A&E attendances (flow)": "total_attendances",
    "Workforce FTE (level)": "FTE: All staff groups - All staff groups",
    "Bed occupancy (level)": "total_occupied_beds_overnight",
    "PESA Health spend (level)": "7. Health (real_gbp_bn)",
    "RTT % within 18 weeks (performance)": "Incomplete RTT pathways - % within 18 weeks",
    "A&E 12-hour decisions to admit (breach flow)": "number_of_patients_spending_12_hours_from_decision_to_admit_to_admission",
    "GP total appointments (flow)": "total_attended_appointments",
    "GP face-to-face appointments (flow)": "attended_face_to_face",
    "GP telephone appointments (flow)": "attended_telephone",
}

NOTES = {
    "RTT waiting list (level)": (
        "MA(1) with constant trend and log transformation.\n"
        "Quadratic trend significant (p=0.0000).\n"
        "Log transformation improved AICc from -365 to -677."
    ),
    "A&E attendances (flow)": (
        "AR(1) with constant trend.\n"
        "No curvature detected (p=0.9146).\n"
        "Moving average smoothing applied to forecast."
    ),
    "Workforce FTE (level)": (
        "AR(2) with constant trend + quadratic_trend.\n"
        "All exog variables significant (p<0.05).\n"
        "quadratic_trend negative (-36,608) - slowing growth."
    ),
    "Bed occupancy (level)": (
        "MA(1) with constant trend + quadratic_trend.\n"
        "quadratic_trend positive (+6,607) - accelerating upward.\n"
        "All exog variables significant (p<0.05)."
    ),
    "PESA Health spend (level)": (
        "Random walk with no drift.\n"
        "Annual series with 20 observations.\n"
        "No curvature detected (p=0.2095)."
    ),
    "RTT % within 18 weeks (performance)": (
        "Random walk with drift.\n"
        "Emergency override enforces smooth transition.\n"
        "Forecast: 0.65 to 0.68 (+5% drift)."
    ),
    "A&E 12-hour decisions to admit (breach flow)": (
        "Random walk with quadratic_trend.\n"
        "FIT_START_OVERRIDE restricts to post-2021 (22 obs).\n"
        "quadratic_trend positive (+48,710) - upward acceleration."
    ),
    "GP total appointments (flow)": (
        "No forecasts - insufficient data (11 obs).\n"
        "Historical data only with warning message."
    ),
    "GP face-to-face appointments (flow)": (
        "No forecasts - insufficient data (11 obs).\n"
        "Historical data only with warning message."
    ),
    "GP telephone appointments (flow)": (
        "No forecasts - insufficient data (11 obs).\n"
        "Historical data only with warning message."
    ),
}
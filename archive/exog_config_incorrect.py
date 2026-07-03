"""
exog_config.py
--------------
Single source of truth for which COVID dummy variables to include as
exogenous regressors per series, based on:
    1. test_covid_significance.py (linear trend OLS)
    2. test_covid_quadratic_check.py (quadratic trend robustness check)

Decision rule applied: a dummy is kept only if it remained significant
(p < 0.05) under BOTH linear and quadratic trend specifications. This
guards against dummies that are really just absorbing a non-linear trend
the model would otherwise be forced to fit, which happened with
workforce's covid_pulse term.

Import this wherever a SARIMAX exog matrix is built, e.g.:

    from src.exog_config import EXOG_CONFIG
    exog_cols = EXOG_CONFIG["RTT waiting list (level)"]
    exog = df[exog_cols]
"""

EXOG_CONFIG = {
    "RTT waiting list (level)": ["covid_pulse", "post_covid_regime"],
    "A&E attendances (flow)": ["covid_pulse"],
    "Workforce FTE (level)": ["post_covid_regime"],
}

# Metric name lookup (matches combined_quarterly.csv 'metric' column values)
METRIC_NAMES = {
    "RTT waiting list (level)": "Incomplete RTT pathways - Total waiting (mil) with estimates for missing data",
    "A&E attendances (flow)": "total_attendances",
    "Workforce FTE (level)": "FTE: All staff groups - All staff groups",
}

# Brief rationale, useful for write-up / dashboard tooltips
NOTES = {
    "RTT waiting list (level)": (
        "Both dummies robust to quadratic trend. Genuine dual effect: "
        "acute 2020 dip plus permanent post-COVID backlog regime shift."
    ),
    "A&E attendances (flow)": (
        "Only covid_pulse survives. Series fully reverted to pre-COVID "
        "trend by ~2021 Q2 — no permanent structural change."
    ),
    "Workforce FTE (level)": (
        "covid_pulse was a linear-trend artefact (p=0.77 once quadratic "
        "trend included) — disappears entirely, no genuine pulse effect. "
        "post_covid_regime remains significant: real acceleration in FTE "
        "growth rate from 2020 onward, likely post-pandemic recruitment."
    ),
}

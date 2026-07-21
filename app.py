"""
app.py
------
Frontend Streamlit interactive dashboard application.
Now includes derived workforce‑efficiency metrics, GP‑to‑A&E ratio,
asymmetric CI bounds, horizon cap, NICE threshold overlay, policy‑context notes,
System Strain Overview page, and ITS counterfactual display.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import sys
import html

ROOT_DIR = Path(__file__).parent.resolve()
FORECAST_PATH = ROOT_DIR / "data" / "processed" / "dashboard_forecasts.csv"
COMBINED_PATH = ROOT_DIR / "data" / "processed" / "combined_quarterly.csv"
NICE_PATH = ROOT_DIR / "data" / "processed" / "nice_clean.csv"
POPULATION_PATH = ROOT_DIR / "data" / "processed" / "population_clean.csv"
COUNTERFACTUAL_PATH = ROOT_DIR / "data" / "processed" / "counterfactuals.csv"

sys.path.insert(0, str(ROOT_DIR / "src"))
from exog_config import METRIC_NAMES

# ---------- metric definitions ----------
METRICS = [
    ("RTT waiting list (level)", "RTT waiting list", False),
    ("A&E attendances (flow)", "A&E attendances", False),
    ("Workforce FTE (level)", "Workforce FTE", False),
    ("Nurse FTE (level)", "Nurse FTE", False),
    ("Doctor FTE (level)", "Doctor FTE", False),
    ("Bed occupancy (level)", "Bed occupancy", False),
    ("RTT % within 18 weeks (performance)", "RTT % within 18 weeks", True),
    ("A&E 12-hour decisions to admit (breach flow)", "A&E 12-hour breach", False),
    ("PESA Health spend (level)", "PESA Health spend", False),
]

COVERAGE = {
    "RTT waiting list (level)": "Coverage 0.93 (h=12)",
    "A&E attendances (flow)": "Coverage 0.94 (h=8)",
    "Workforce FTE (level)": "Coverage 0.94 (h=12)",
    "Nurse FTE (level)": "Coverage 0.96 (h=8)",
    "Doctor FTE (level)": "Coverage 0.92 (h=8)",
    "Bed occupancy (level)": "Coverage 0.94 (h=8)",
    "RTT % within 18 weeks (performance)": "Coverage 0.95 (h=8)",
    "A&E 12-hour decisions to admit (breach flow)": "Coverage 1.00 (h=8) – indicative",
    "PESA Health spend (level)": "Not backtested – random‑walk model",
}

INTERPRETATION = {
    "RTT waiting list (level)":
        "Recent decline, but the forecast expects renewed growth, consistent with a dampened trend.",
    "A&E attendances (flow)":
        "Trend restored following completion of the quarterly data; forecast direction remains stable (see footnote 2).",
    "Workforce FTE (level)":
        "Essentially flat recently; the model forecasts modest longer‑run growth.",
    "Nurse FTE (level)":
        "Recent data and forecast agree on direction; historically under‑predicted, so intervals were widened.",
    "Doctor FTE (level)":
        "Largest projected movement in the suite (23–27% depending on method; see footnote 1).",
    "Bed occupancy (level)":
        "Forecast to remain essentially flat, with a modest seasonal oscillation each year.",
    "RTT % within 18 weeks (performance)":
        "Recent improvement but the model expects partial mean reversion, standard for logit‑ARMA (see footnote 3).",
    "A&E 12-hour decisions to admit (breach flow)":
        "Recent decline but the model forecasts a large rise; least reliable row (small sample, high volatility).",
    "PESA Health spend (level)":
        "Quarterly trend not meaningful (annual data); forecast reflects the annual drift model (see footnote 4).",
}

st.set_page_config(
    page_title="Health Econometric Forecasting Dashboard",
    layout="wide"
)

def load_data():
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(f"Forecast file not found at: {FORECAST_PATH}")
    df = pd.read_csv(FORECAST_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    nice_df = None
    if NICE_PATH.exists():
        nice_df = pd.read_csv(NICE_PATH)
        nice_df["date"] = pd.to_datetime(nice_df["date"])

    pop_df = None
    if POPULATION_PATH.exists():
        pop_df = pd.read_csv(POPULATION_PATH)
        pop_df["date"] = pd.to_datetime(pop_df["date"])

    return df, nice_df, pop_df

def calculate_ci_width(fore_df):
    if fore_df.empty or "ci_lower" not in fore_df.columns or "ci_upper" not in fore_df.columns:
        return None
    ci_width = ((fore_df["ci_upper"] - fore_df["ci_lower"]) / fore_df["value"]).mean() * 100
    return ci_width

def is_percentage_metric(metric_name):
    return "%" in metric_name or "percentage" in metric_name.lower() or "within 18 weeks" in metric_name.lower()

def calculate_forecast_quality(fore_df, hist_df=None, metric_name=""):
    if fore_df.empty or len(fore_df) < 3:
        return "Insufficient data", "Unknown", None, None

    values = fore_df["value"].values
    diff = np.diff(values)

    if len(diff) > 1:
        mean_abs_change = np.mean(np.abs(diff))
        mean_value = np.mean(values)
        rel_variation = (mean_abs_change / mean_value) * 100 if mean_value > 0 else 0

        volatile_indicators = ["breach", "12-hour", "A&E 12-hour"]
        is_volatile = any(indicator in metric_name.lower() for indicator in volatile_indicators)

        if rel_variation < 0.5:
            quality = f"Very stable ({rel_variation:.1f}%)"
        elif rel_variation < 2.0:
            quality = f"Stable ({rel_variation:.1f}%)"
        elif rel_variation < 5.0:
            quality = f"Moderate variation ({rel_variation:.1f}%)"
        elif rel_variation < 12.0:
            quality = f"Typical health data variation ({rel_variation:.1f}%)"
        else:
            if is_volatile:
                quality = f"High variation: expected for volatile series ({rel_variation:.1f}%)"
            else:
                quality = f"High variation: possibly due to over-fitting or series volatility ({rel_variation:.1f}%)"
    else:
        rel_variation = None
        quality = "Insufficient data for variation assessment"

    if len(values) > 3:
        t = np.arange(len(values))
        slope = np.polyfit(t, values, 1)[0]
        direction = "Increasing" if slope > 0 else ("Decreasing" if slope < 0 else "Stable")
    else:
        first_val = values[0]
        last_val = values[-1]
        pct_change = ((last_val - first_val) / abs(first_val)) * 100 if first_val != 0 else 0
        if pct_change > 3.0:
            direction = "Increasing"
        elif pct_change < -3.0:
            direction = "Decreasing"
        else:
            direction = "Stable"

    change_pct = None
    if hist_df is not None and not hist_df.empty:
        last_hist = hist_df["value"].iloc[-1]
        last_fore = values[-1]
        if last_hist != 0:
            change_pct = ((last_fore - last_hist) / abs(last_hist)) * 100

    return quality, direction, rel_variation, change_pct

def display_calibration_disclaimer(metric_name):
    if metric_name == "Workforce FTE (level)":
        st.caption(
            "**Note**: This model is the best performer among tested alternatives, "
            "but confidence intervals have not passed full calibration testing. "
            "The forecast should be treated as indicative, not statistical. "
            "This is a data-adequacy limitation rather than a model misspecification issue."
        )
    elif "A&E 12-hour" in metric_name and "breach" in metric_name.lower():
        st.caption(
            "**Note**: This series is highly volatile and the forecast horizon has been shortened to 8 quarters. "
            "Confidence intervals are constrained to historical bounds. Treat as indicative."
        )
    elif metric_name.startswith("GP"):
        st.caption(
            "**Note**: Only historical data is shown due to insufficient observations for reliable forecasting."
        )
    elif "RTT waiting list" in metric_name:
        st.caption(
            "**Wide confidence intervals**: The RTT waiting‑list forecast is produced on a log‑scale, "
            "so uncertainty grows multiplicatively over time. The resulting wide bands are the minimum "
            "required to achieve the validated 95% coverage. "
            "This forecast is conditional on current NHS policy; the NICE threshold rise to £25k–£35k/QALY "
            "(April 2026) is not explicitly modelled but may affect future waiting‑list dynamics. "
            "See the methodology document for details."
        )
    else:
        st.caption(
            "**Model selection**: The displayed forecast is from the best-performing model among a set of alternatives, "
            "selected via rolling-origin backtesting. Confidence intervals are shown at the 95% level."
        )

def format_value(value, metric_name, is_percentage=False):
    if is_percentage or is_percentage_metric(metric_name):
        return f"{value:.1%}" if value < 1 else f"{value:.1f}%"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return f"{value:.1f}"

# ------------------------------------------------------------
# Derived metric helpers (unchanged)
# ------------------------------------------------------------
DERIVED_METRICS = {
    "RTT waiting list per 1,000 population": {
        "base_metric": "RTT waiting list (level)",
        "divisor": "population",
        "scaling": 1000,
        "unit": "Patients per 1,000",
        "note": "Population held constant at latest ONS value after 2025.",
    },
    "Workforce FTE per 100,000 population": {
        "base_metric": "Workforce FTE (level)",
        "divisor": "population",
        "scaling": 100000,
        "unit": "FTE per 100,000",
        "note": "Population held constant at latest ONS value after 2025.",
    },
    "Health spend per capita (£)": {
        "base_metric": "PESA Health spend (level)",
        "divisor": "population",
        "scaling": 1e9,
        "unit": "£ per person",
        "note": "Population held constant at latest ONS value after 2025. Spend in real GBP billions converted to per‑capita.",
    },
    "GP Appointments per A&E Attendance": {
        "base_metric": "GP total appointments (flow)",
        "divisor": "A&E attendances (flow)",
        "scaling": 1,
        "unit": "Ratio",
        "note": "Ratio > 1 indicates more GP activity relative to A&E demand; falling ratio suggests increasing acute pressure.",
    },
    "RTT waiting list per 1,000 FTE": {
        "base_metric": "RTT waiting list (level)",
        "divisor_metric": "Workforce FTE (level)",
        "scaling": 1000,
        "unit": "Patients per 1,000 FTE",
        "note": "RTT waiting list divided by total NHS workforce FTE.",
    },
    "A&E attendances per FTE": {
        "base_metric": "A&E attendances (flow)",
        "divisor_metric": "Workforce FTE (level)",
        "scaling": 1,
        "unit": "Attendances per FTE",
        "note": "A&E attendances per total NHS workforce FTE.",
    },
    "Bed occupancy per nurse FTE": {
        "base_metric": "Bed occupancy (level)",
        "divisor_metric": "Nurse FTE (level)",
        "scaling": 1,
        "unit": "Beds per nurse FTE",
        "note": "Occupied beds per nurse FTE (Nurses & health visitors).",
    },
}

def build_derived_series(selected_derived, forecast_df, pop_df, nice_df=None):
    info = DERIVED_METRICS[selected_derived]

    if info.get("divisor") == "population":
        base_metric = info["base_metric"]
        scaling = info["scaling"]
        metric_df = forecast_df[forecast_df["metric"] == base_metric].sort_values("quarter")
        if metric_df.empty:
            raise ValueError(f"'{base_metric}' is not present in the dashboard forecast file.")
        hist_df = metric_df[metric_df["type"] == "history"].copy()
        fore_df = metric_df[metric_df["type"] == "forecast"].copy()
        pop = pop_df.copy()
        pop["year"] = pop["date"].dt.year
        hist_df["year"] = hist_df["quarter"].dt.year
        fore_df["year"] = fore_df["quarter"].dt.year
        hist_df = hist_df.merge(pop[["year", "population"]], on="year", how="left")
        fore_df = fore_df.merge(pop[["year", "population"]], on="year", how="left")
        last_pop = pop["population"].iloc[-1]
        fore_df["population"] = fore_df["population"].fillna(last_pop)
        hist_df["population"] = hist_df["population"].fillna(last_pop)
        hist_df["value"] = (hist_df["value"] / hist_df["population"]) * scaling
        fore_df["value"] = (fore_df["value"] / fore_df["population"]) * scaling
        if "ci_lower" in fore_df.columns:
            fore_df["ci_lower"] = (fore_df["ci_lower"] / fore_df["population"]) * scaling
            fore_df["ci_upper"] = (fore_df["ci_upper"] / fore_df["population"]) * scaling
        hist_df = hist_df.drop(columns=["year", "population"])
        fore_df = fore_df.drop(columns=["year", "population"])
        return hist_df, fore_df, info["unit"], info["note"]

    elif info.get("divisor") == "A&E attendances (flow)":
        gp_metric = "GP total appointments (flow)"
        ae_metric = "A&E attendances (flow)"
        gp = forecast_df[forecast_df["metric"] == gp_metric].sort_values("quarter")
        ae = forecast_df[forecast_df["metric"] == ae_metric].sort_values("quarter")
        gp_hist = gp[gp["type"] == "history"]
        gp_fore = gp[gp["type"] == "forecast"]
        ae_hist = ae[ae["type"] == "history"]
        ae_fore = ae[ae["type"] == "forecast"]
        hist = gp_hist.merge(ae_hist[["quarter", "value"]], on="quarter", suffixes=("_gp", "_ae"))
        hist["value"] = hist["value_gp"] / hist["value_ae"].replace(0, np.nan)
        hist = hist[["quarter", "value"]].copy()
        hist["type"] = "history"
        if not gp_fore.empty and not ae_fore.empty:
            fore = gp_fore.merge(ae_fore[["quarter", "value"]], on="quarter", suffixes=("_gp", "_ae"))
            fore["value"] = fore["value_gp"] / fore["value_ae"].replace(0, np.nan)
            fore = fore[["quarter", "value"]].copy()
            fore["type"] = "forecast"
        else:
            fore = pd.DataFrame(columns=["quarter", "value", "type"])
        hist["ci_lower"] = np.nan
        hist["ci_upper"] = np.nan
        fore["ci_lower"] = np.nan
        fore["ci_upper"] = np.nan
        return hist, fore, info["unit"], info["note"]

    elif "divisor_metric" in info:
        base_metric = info["base_metric"]
        divisor_metric = info["divisor_metric"]
        scaling = info["scaling"]
        base_df = forecast_df[forecast_df["metric"] == base_metric].sort_values("quarter")
        divisor_df = forecast_df[forecast_df["metric"] == divisor_metric].sort_values("quarter")
        if base_df.empty or divisor_df.empty:
            raise ValueError(f"One of '{base_metric}' or '{divisor_metric}' is missing.")
        base_hist = base_df[base_df["type"] == "history"]
        base_fore = base_df[base_df["type"] == "forecast"]
        div_hist = divisor_df[divisor_df["type"] == "history"]
        div_fore = divisor_df[divisor_df["type"] == "forecast"]
        hist = base_hist.merge(div_hist[["quarter", "value"]], on="quarter", suffixes=("_base", "_div"))
        hist["value"] = (hist["value_base"] / hist["value_div"].replace(0, np.nan)) * scaling
        hist = hist[["quarter", "value"]].copy()
        hist["type"] = "history"
        fore = base_fore.merge(div_fore[["quarter", "value", "ci_lower", "ci_upper"]],
                               on="quarter", suffixes=("_base", "_div"))
        fore["value"] = (fore["value_base"] / fore["value_div"].replace(0, np.nan)) * scaling
        if "ci_lower_base" in fore.columns and "ci_lower_div" in fore.columns:
            fore["ci_lower"] = fore["ci_lower_base"] / fore["value_div"] * scaling
            fore["ci_upper"] = fore["ci_upper_base"] / fore["value_div"] * scaling
        fore = fore[["quarter", "value", "ci_lower", "ci_upper"]].copy()
        fore["type"] = "forecast"
        hist["ci_lower"] = np.nan
        hist["ci_upper"] = np.nan
        return hist, fore, info["unit"], info["note"]
    else:
        raise ValueError("Unknown divisor type")

# ------------------------------------------------------------
# Strain overview computation (unchanged)
# ------------------------------------------------------------
def compute_strain_table():
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(f"Forecast file not found at: {FORECAST_PATH}")
    if not COMBINED_PATH.exists():
        raise FileNotFoundError(f"Combined quarterly file not found at: {COMBINED_PATH}")

    df = pd.read_csv(FORECAST_PATH, parse_dates=["quarter"])
    combined = pd.read_csv(COMBINED_PATH, parse_dates=["quarter"])

    rows = []
    for metric_key, short_name, is_pct in METRICS:
        raw_name = METRIC_NAMES.get(metric_key)
        if raw_name is None:
            continue

        hist_all = combined[combined["metric"] == raw_name].dropna(subset=["value"]).sort_values("quarter")
        if len(hist_all) >= 4:
            recent = hist_all.tail(4).copy()
            x = np.arange(len(recent))
            y = recent["value"].values
            b, _ = np.polyfit(x, y, 1)
            avg = np.mean(y)
            trend_pct = (b * 4) / avg * 100 if avg != 0 else 0.0
        else:
            trend_pct = np.nan

        fore = df[(df["metric"] == metric_key) & (df["type"] == "forecast")].sort_values("quarter")
        if len(fore) >= 3:
            xf = np.arange(len(fore))
            yf = fore["value"].values
            slope, _ = np.polyfit(xf, yf, 1)
            avg_f = np.mean(yf)
            slope_pct = (slope / avg_f) * 100 if avg_f != 0 else 0.0
            total_change = slope_pct * (len(fore) - 1)
            if total_change > 2.0:
                direction = "Increasing"
            elif total_change < -2.0:
                direction = "Decreasing"
            else:
                direction = "Stable"
        else:
            direction = "N/A"
            total_change = np.nan

        if metric_key == "PESA Health spend (level)":
            trend_disp = "Annual series"
        elif np.isnan(trend_pct):
            trend_disp = "Insufficient data"
        else:
            trend_disp = f"{trend_pct:+.2f}%"

        if total_change is not np.nan:
            fore_disp = f"{direction}, {total_change:+.2f}% total"
        else:
            fore_disp = direction

        rows.append([
            short_name,
            trend_disp,
            fore_disp,
            COVERAGE.get(metric_key, ""),
            INTERPRETATION.get(metric_key, ""),
        ])

    table_df = pd.DataFrame(rows, columns=[
        "Metric", "Current trend (last 4 qtrs)", "Forecast direction (full horizon)",
        "Confidence rating", "Interpretation"
    ])

    footnotes = [
        "1. Doctor FTE: The linear‑fit‑implied total change (~23%) understates the endpoint movement (~27%) because the forecast trajectory is curved. The interpretation line gives a range; the detailed backtest log contains both figures.",
        "2. A&E attendances: The trend was temporarily withheld due to a partial quarter in the raw data. June 2026 monthly data was subsequently appended, completing the quarter and allowing a reliable trend to be calculated. The current trend of +1.89% reflects the most recent four complete quarters.",
        "3. RTT %: The logit‑ARMA(1,1) model is mean‑reverting. A period of above‑average performance is followed by a forecasted partial reversal toward the long‑run mean – a deliberate model property, not a forecast failure. Coverage validated at 0.95 (h=8).",
        "4. PESA Health spend: The underlying data is annual; quarterly rows are forward‑filled for pipeline consistency. The quarterly trend is therefore not meaningful.",
        "Method notes:",
        "  - Current trends: annualised linear fit over the last 4 historical quarters.",
        "  - Forecast directions: total change derived from linear‑fit slope over the full forecast horizon. A ±2% total‑change threshold determines Increasing/Stable/Decreasing. This avoids seasonal‑endpoint contamination (e.g., Bed occupancy).",
        "  - Confidence ratings are from backtest‑validated coverage figures in the reports/ folder."
    ]
    return table_df, footnotes

# ------------------------------------------------------------
# Individual Series Analysis Page (with ITS counterfactual)
# ------------------------------------------------------------
def individual_series_page():
    st.title("Health Systems Econometric Forecasting Dashboard")
    st.markdown("---")

    try:
        df, nice_df, pop_df = load_data()
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return

    if df.empty:
        st.error("The forecast file is empty. Please run pipeline again.")
        return

    # Load counterfactuals if available
    counterfactual_df = None
    if COUNTERFACTUAL_PATH.exists():
        counterfactual_df = pd.read_csv(COUNTERFACTUAL_PATH)
        counterfactual_df["quarter"] = pd.to_datetime(counterfactual_df["quarter"])

    st.sidebar.header("Series Selection")

    base_metrics = sorted(df["metric"].dropna().unique())
    derived_labels = list(DERIVED_METRICS.keys())
    all_options = base_metrics + derived_labels
    selected = st.sidebar.selectbox("Choose a system indicator to analyse:", all_options)

    show_ci = st.sidebar.checkbox("Display 95% Forecast Confidence Intervals", value=True)
    show_nice = st.sidebar.checkbox("Overlay NICE QALY Policy Threshold Shifts", value=True)
    show_counterfactual = st.sidebar.checkbox("Show ITS Counterfactual (no break)", value=True)
    zoom_forecast = st.sidebar.checkbox("Zoom Y-Axis to Forecast Range", value=False)

    max_horizons = 24
    horizon_cap = st.sidebar.slider(
        "Max forecast quarters to display",
        min_value=4, max_value=24, value=12, step=4,
        help="Limits the x‑axis to the first N forecast quarters for clarity."
    )

    if selected in DERIVED_METRICS:
        is_derived = True
        derived_info = DERIVED_METRICS[selected]
        if derived_info.get("divisor") == "A&E attendances (flow)":
            st.warning(
                "GP appointments data is not forecasted (insufficient history). "
                "Only historical ratio is shown. "
                "Trend in GP activity vs. A&E attendances can still indicate demand pressure."
            )
        try:
            hist_df, fore_df, unit_label, note = build_derived_series(selected, df, pop_df, nice_df)
        except ValueError as e:
            st.warning(str(e))
            hist_df = pd.DataFrame()
            fore_df = pd.DataFrame()
            unit_label = ""
            note = ""
        selected_metric = selected
        is_pct = False
    else:
        is_derived = False
        selected_metric = selected
        metric_df = df[df["metric"] == selected_metric].sort_values("quarter")
        hist_df = metric_df[metric_df["type"] == "history"]
        fore_df = metric_df[metric_df["type"] == "forecast"]
        is_pct = is_percentage_metric(selected_metric)
        note = None

    if not fore_df.empty and len(fore_df) > horizon_cap:
        fore_df_display = fore_df.head(horizon_cap)
    else:
        fore_df_display = fore_df

    if selected_metric.startswith("GP") and not is_derived:
        st.warning(
            "Data Limitation: GP appointments data is only available from October 2023. "
            "With only 11 quarterly observations, reliable forecasting is not possible. "
            "Only historical data is shown. This is a data scope limitation."
        )

    if "A&E 12-hour" in selected_metric and "breach" in selected_metric.lower():
        st.warning(
            "Note: This series is highly volatile with a short history (22 observations post-2021). "
            "Confidence intervals are wide and should be interpreted with caution."
        )

    fig = go.Figure()

    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df["quarter"], y=hist_df["value"],
            mode="lines+markers", name="Observed History",
            line=dict(color="#1f77b4", width=2.5)
        ))

    if not fore_df_display.empty and not selected_metric.startswith("GP"):
        if not hist_df.empty:
            last_hist = hist_df.iloc[-1:]
            plot_fore = pd.concat([last_hist, fore_df_display], ignore_index=True)
        else:
            plot_fore = fore_df_display

        fig.add_trace(go.Scatter(
            x=plot_fore["quarter"], y=plot_fore["value"],
            mode="lines+markers", name="Forecast Horizon",
            line=dict(color="#ff7f0e", width=2.5, dash="dash")
        ))

        if show_ci and "ci_lower" in fore_df_display.columns and "ci_upper" in fore_df_display.columns:
            if not hist_df.empty:
                plot_fore_ci = pd.concat([last_hist, fore_df_display], ignore_index=True)
                plot_fore_ci["ci_lower"] = plot_fore_ci["ci_lower"].fillna(plot_fore_ci["value"])
                plot_fore_ci["ci_upper"] = plot_fore_ci["ci_upper"].fillna(plot_fore_ci["value"])
            else:
                plot_fore_ci = fore_df_display

            fig.add_trace(go.Scatter(
                x=pd.concat([plot_fore_ci["quarter"], plot_fore_ci["quarter"].iloc[::-1]], ignore_index=True),
                y=pd.concat([plot_fore_ci["ci_upper"], plot_fore_ci["ci_lower"].iloc[::-1]], ignore_index=True),
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.15)',
                line=dict(color='rgba(255,127,14,0)'),
                hoverinfo="skip",
                name="95% Confidence Interval"
            ))

        # ---- ITS counterfactual trace ----
        if show_counterfactual and counterfactual_df is not None:
            c_df = counterfactual_df[counterfactual_df["metric"] == selected_metric]
            if not c_df.empty:
                # restrict to same forecast quarters as displayed
                c_df = c_df[c_df["quarter"].isin(fore_df_display["quarter"])]
                if not c_df.empty:
                    fig.add_trace(go.Scatter(
                        x=c_df["quarter"],
                        y=c_df["counterfactual_mean"],
                        mode="lines",
                        name="Counterfactual (no break)",
                        line=dict(color="green", width=2, dash="dot")
                    ))
                    if "counterfactual_ci_lower" in c_df.columns and "counterfactual_ci_upper" in c_df.columns:
                        fig.add_trace(go.Scatter(
                            x=pd.concat([c_df["quarter"], c_df["quarter"].iloc[::-1]], ignore_index=True),
                            y=pd.concat([c_df["counterfactual_ci_upper"], c_df["counterfactual_ci_lower"].iloc[::-1]], ignore_index=True),
                            fill='toself',
                            fillcolor='rgba(0,128,0,0.1)',
                            line=dict(color='rgba(0,128,0,0)'),
                            hoverinfo="skip",
                            name="Counterfactual CI"
                        ))
    elif selected_metric.startswith("GP") and not hist_df.empty and not is_derived:
        last_date = hist_df["quarter"].iloc[-1]
        last_value = hist_df["value"].iloc[-1]
        fig.add_annotation(
            x=last_date, y=last_value * 0.9,
            text="Insufficient data for forecasting",
            showarrow=True, arrowhead=1, ax=50, ay=-30,
            font=dict(size=12, color="orange"), arrowcolor="orange"
        )

    if show_nice and nice_df is not None and not hist_df.empty:
        min_date = hist_df["quarter"].min()
        max_date = hist_df["quarter"].max() if fore_df_display.empty else fore_df_display["quarter"].max()
        if "Health spend" in selected_metric or "PESA" in selected_metric or "per capita" in selected_metric:
            latest_nice = nice_df[nice_df["category"] == "standard"].sort_values("date").iloc[-1]
            fig.add_hrect(
                y0=latest_nice["lower_gbp_per_qaly"],
                y1=latest_nice["upper_gbp_per_qaly"],
                line_width=0,
                fillcolor="rgba(212, 39, 40, 0.1)",
                annotation_text="NICE threshold £25k–£35k/QALY",
                annotation_position="top left",
            )
        else:
            local_nice = nice_df[(nice_df["date"] >= min_date) & (nice_df["date"] <= max_date)]
            for _, row in local_nice.iterrows():
                fig.add_shape(
                    type="line", x0=row["date"], x1=row["date"], y0=0, y1=1,
                    yref="paper", line=dict(color="#d62728", width=1.5, dash="dot")
                )
                fig.add_annotation(
                    x=row["date"], y=1.02, yref="paper",
                    text=f"NICE: {row['category']}", showarrow=False,
                    font=dict(size=9, color="#d62728"), textangle=-45
                )

    yaxis_settings = {"title": "Metric Values", "fixedrange": False}
    if zoom_forecast and not fore_df_display.empty and not selected_metric.startswith("GP"):
        range_values = [fore_df_display["value"]]
        if show_ci and "ci_lower" in fore_df_display.columns and "ci_upper" in fore_df_display.columns:
            range_values.append(fore_df_display["ci_lower"].fillna(fore_df_display["value"]))
            range_values.append(fore_df_display["ci_upper"].fillna(fore_df_display["value"]))
        all_vals = pd.concat(range_values)
        y_min, y_max = all_vals.min(), all_vals.max()
        pad = (y_max - y_min) * 0.1 if y_max > y_min else (abs(y_max) * 0.1 or 1)
        yaxis_settings["range"] = [y_min - pad, y_max + pad]
        yaxis_settings["autorange"] = False
    else:
        yaxis_settings["autorange"] = True

    fig.update_layout(
        title=dict(text=f"System Tracking Matrix: {selected_metric}", font=dict(size=16)),
        xaxis_title="Timeline Horizon",
        yaxis=dict(**yaxis_settings),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=40, r=40),
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

    if note:
        st.caption(note)

    if "Health spend" in selected_metric and nice_df is not None:
        latest_nice = nice_df[nice_df["category"] == "standard"].sort_values("date").iloc[-1]
        lower_threshold = latest_nice["lower_gbp_per_qaly"]
        upper_threshold = latest_nice["upper_gbp_per_qaly"]
        if not fore_df.empty:
            spend_per_person = fore_df["value"].mean()
            qaly_lower = spend_per_person / upper_threshold
            qaly_upper = spend_per_person / lower_threshold
            st.info(
                f"**NICE cost‑effectiveness context**: The current standard threshold is "
                f"£{lower_threshold:,}–£{upper_threshold:,} per QALY. "
                f"At an average forecast spend of £{spend_per_person:,.0f} per person, "
                f"this implies **{qaly_lower:.0f}–{qaly_upper:.0f} QALYs per person per year** "
                f"if the entire budget were allocated to cost‑effective interventions."
            )

    st.markdown("---")
    st.subheader("Summary Statistics")

    is_pct = is_percentage_metric(selected_metric) if not is_derived else False

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not hist_df.empty:
            range_years = (hist_df["quarter"].max() - hist_df["quarter"].min()).days / 365.25
            st.metric(label="Historical Range", value=f"{range_years:.1f} Years")
        else:
            st.metric(label="Historical Range", value="N/A")

    with col2:
        if not fore_df_display.empty and not selected_metric.startswith("GP"):
            st.metric(label="Forecast Horizon", value=f"{len(fore_df_display)} Quarters")
        else:
            st.metric(label="Forecast Horizon", value="0")

    with col3:
        if not fore_df_display.empty and not hist_df.empty and not selected_metric.startswith("GP"):
            last_hist = hist_df["value"].iloc[-1]
            last_fore = fore_df_display["value"].iloc[-1]
            if is_pct:
                change = last_fore - last_hist
                st.metric(label="Total Forecast Change", value=f"{change:+.1f} pp")
            else:
                growth = ((last_fore - last_hist) / abs(last_hist)) * 100 if last_hist != 0 else 0
                st.metric(label="Total Forecast Change", value=f"{growth:+.1f}%")
        else:
            st.metric(label="Total Forecast Change", value="N/A")

    with col4:
        if not fore_df_display.empty and show_ci and "ci_lower" in fore_df_display.columns and "ci_upper" in fore_df_display.columns:
            ci_width = calculate_ci_width(fore_df_display)
            if ci_width is not None and np.isfinite(ci_width) and ci_width < 500:
                st.metric(label="Avg CI Width", value=f"±{ci_width:.1f}%")
            else:
                st.metric(label="Avg CI Width", value="Wide (see note)")
        else:
            st.metric(label="Avg CI Width", value="N/A")

    if not fore_df_display.empty and show_ci and "ci_lower" in fore_df_display.columns:
        last_row = fore_df_display.iloc[-1]
        if last_row["value"] > 0:
            lower_dev = ((last_row["ci_lower"] - last_row["value"]) / last_row["value"]) * 100
            upper_dev = ((last_row["ci_upper"] - last_row["value"]) / last_row["value"]) * 100
            col1, col2, _ = st.columns([2, 2, 4])
            with col1:
                st.metric(label="Lower bound (last horizon)", value=f"{lower_dev:+.1f}%")
            with col2:
                st.metric(label="Upper bound (last horizon)", value=f"{upper_dev:+.1f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Historical Observations", value=f"{len(hist_df)}" if not hist_df.empty else "0")
    with col2:
        st.metric(label="Forecast Points", value=f"{len(fore_df_display)}" if not fore_df_display.empty else "0")
    with col3:
        if not fore_df_display.empty and not selected_metric.startswith("GP"):
            quality, direction, variation, change = calculate_forecast_quality(fore_df_display, hist_df, selected_metric)
            st.metric(label="Forecast Direction", value=direction)
        else:
            st.metric(label="Forecast Direction", value="N/A")
    with col4:
        if not fore_df_display.empty and not selected_metric.startswith("GP"):
            quality, direction, variation, change = calculate_forecast_quality(fore_df_display, hist_df, selected_metric)
            st.metric(label="Forecast Quality", value=quality)
        else:
            st.metric(label="Forecast Quality", value="N/A")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Historical Data**")
        if not hist_df.empty:
            hist_min = hist_df["value"].min()
            hist_max = hist_df["value"].max()
            hist_mean = hist_df["value"].mean()
            if is_pct:
                st.write(f"- Start: {hist_df['quarter'].min().strftime('%Y-%m-%d')}")
                st.write(f"- End: {hist_df['quarter'].max().strftime('%Y-%m-%d')}")
                st.write(f"- Min: {hist_min:.1%}")
                st.write(f"- Max: {hist_max:.1%}")
                st.write(f"- Mean: {hist_mean:.1%}")
            else:
                st.write(f"- Start: {hist_df['quarter'].min().strftime('%Y-%m-%d')}")
                st.write(f"- End: {hist_df['quarter'].max().strftime('%Y-%m-%d')}")
                st.write(f"- Min: {hist_min:,.0f}")
                st.write(f"- Max: {hist_max:,.0f}")
                st.write(f"- Mean: {hist_mean:,.0f}")
        else:
            st.write("No historical data available")

    with col2:
        st.markdown("**Forecast Data**")
        if not fore_df_display.empty and not selected_metric.startswith("GP"):
            fore_min = fore_df_display["value"].min()
            fore_max = fore_df_display["value"].max()
            fore_mean = fore_df_display["value"].mean()
            st.write(f"- Start: {fore_df_display['quarter'].min().strftime('%Y-%m-%d')}")
            st.write(f"- End: {fore_df_display['quarter'].max().strftime('%Y-%m-%d')}")
            if is_pct:
                st.write(f"- Min: {fore_min:.1%}")
                st.write(f"- Max: {fore_max:.1%}")
                st.write(f"- Mean: {fore_mean:.1%}")
            else:
                st.write(f"- Min: {fore_min:,.0f}")
                st.write(f"- Max: {fore_max:,.0f}")
                st.write(f"- Mean: {fore_mean:,.0f}")
            if not hist_df.empty:
                last_hist = hist_df["value"].iloc[-1]
                last_fore = fore_df_display["value"].iloc[-1]
                if is_pct:
                    change = last_fore - last_hist
                    st.write(f"- Change: {change:+.1f} pp")
                else:
                    growth = ((last_fore - last_hist) / abs(last_hist)) * 100 if last_hist != 0 else 0
                    st.write(f"- Change: {growth:+.1f}%")
            quality, direction, variation, change = calculate_forecast_quality(fore_df_display, hist_df, selected_metric)
            if variation is not None:
                st.write(f"- Variation: {variation:.1f}%")
        else:
            st.write("No forecast available")

    with col3:
        st.markdown("**Confidence Intervals**")
        if not fore_df_display.empty and show_ci and "ci_lower" in fore_df_display.columns and "ci_upper" in fore_df_display.columns:
            ci_lower = fore_df_display["ci_lower"].min()
            ci_upper = fore_df_display["ci_upper"].max()
            if is_pct:
                st.write(f"- CI lower: {ci_lower:.1%}")
                st.write(f"- CI upper: {ci_upper:.1%}")
            else:
                st.write(f"- CI lower: {ci_lower:,.0f}")
                st.write(f"- CI upper: {ci_upper:,.0f}")
            ci_width = calculate_ci_width(fore_df_display)
            if ci_width is not None and np.isfinite(ci_width):
                st.write(f"- Avg width: ±{ci_width:.1f}%")
            if not fore_df_display.empty:
                narrowest = ((fore_df_display["ci_upper"] - fore_df_display["ci_lower"]) / fore_df_display["value"]).min() * 100
                widest = ((fore_df_display["ci_upper"] - fore_df_display["ci_lower"]) / fore_df_display["value"]).max() * 100
                if np.isfinite(narrowest) and narrowest < 500:
                    st.write(f"- Narrowest: ±{narrowest:.1f}%")
                else:
                    st.write("- Narrowest: N/A (wide intervals)")
                if np.isfinite(widest) and widest < 500:
                    st.write(f"- Widest: ±{widest:.1f}%")
                else:
                    st.write("- Widest: N/A (wide intervals)")
        else:
            st.write("No confidence intervals available")

    display_calibration_disclaimer(selected_metric)

# ------------------------------------------------------------
# System Strain Overview Page
# ------------------------------------------------------------
def strain_overview_page():
    st.title("System Strain Overview")
    st.markdown("---")
    try:
        table_df, footnotes = compute_strain_table()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    cols_width = {
        "Metric": "15%",
        "Current trend (last 4 qtrs)": "12%",
        "Forecast direction (full horizon)": "15%",
        "Confidence rating": "12%",
        "Interpretation": "46%"
    }

    html_table = '<table style="width:100%; table-layout:fixed; border-collapse:collapse; font-size:0.9em;">'
    html_table += "<thead><tr>"
    for col in table_df.columns:
        html_table += f'<th style="width:{cols_width[col]}; text-align:left; padding:8px; border-bottom:2px solid #ddd;">{col}</th>'
    html_table += "</tr></thead><tbody>"

    for _, row in table_df.iterrows():
        html_table += "<tr>"
        for col in table_df.columns:
            cell_text = html.escape(str(row[col])).replace("\n", "<br>")
            html_table += (
                f'<td style="width:{cols_width[col]}; text-align:left; padding:8px; border-bottom:1px solid #ddd; '
                f'word-wrap:break-word; white-space:normal;">{cell_text}</td>'
            )
        html_table += "</tr>"
    html_table += "</tbody></table>"

    st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Footnotes")
    md_parts = []
    i = 0
    while i < len(footnotes):
        note = footnotes[i]
        if note == "Method notes:":
            md_parts.append("- **Method notes:**")
            i += 1
            while i < len(footnotes) and footnotes[i].startswith("  - "):
                sub_text = footnotes[i][4:].strip()
                md_parts.append(f"    - {sub_text}")
                i += 1
        else:
            md_parts.append(f"- {note}")
            i += 1
    footnotes_md = "\n".join(md_parts)
    st.markdown(footnotes_md)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Individual Series Analysis", "System Strain Overview"])
    if page == "Individual Series Analysis":
        individual_series_page()
    else:
        strain_overview_page()

if __name__ == "__main__":
    main()
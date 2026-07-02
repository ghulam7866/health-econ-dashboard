"""
app.py
------
Frontend Streamlit interactive dashboard application.

This module provides the user interface for the Health Economics Forecasting Dashboard.
It displays time series data with forecasts, confidence intervals, and NICE policy annotations.

Usage:
    streamlit run app.py

Input:
    data/processed/dashboard_forecasts.csv
    data/processed/nice_clean.csv

Output:
    Interactive Streamlit dashboard in the browser

Last updated: 2026-07-02
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import traceback
import numpy as np

ROOT_DIR = Path(__file__).parent.resolve()
FORECAST_PATH = ROOT_DIR / "data" / "processed" / "dashboard_forecasts.csv"
NICE_PATH = ROOT_DIR / "data" / "processed" / "nice_clean.csv"

st.set_page_config(
    page_title="Health Econometric Forecasting Dashboard",
    layout="wide"
)


def load_data():
    """
    Load forecast data and NICE reference table.

    Returns
    -------
    tuple
        (forecast_df, nice_df) where nice_df may be None if file not found

    Raises
    ------
    FileNotFoundError
        If the forecast file is not found
    """
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(f"Forecast file not found at: {FORECAST_PATH}")

    df = pd.read_csv(FORECAST_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    nice_df = None
    if NICE_PATH.exists():
        nice_df = pd.read_csv(NICE_PATH)
        nice_df["date"] = pd.to_datetime(nice_df["date"])

    return df, nice_df


def calculate_ci_width(fore_df):
    """
    Calculate the average confidence interval width as a percentage.

    Parameters
    ----------
    fore_df : pd.DataFrame
        Forecast DataFrame with ci_lower and ci_upper columns

    Returns
    -------
    float or None
        Average CI width as percentage, or None if not calculable
    """
    if fore_df.empty or "ci_lower" not in fore_df.columns or "ci_upper" not in fore_df.columns:
        return None

    ci_width = ((fore_df["ci_upper"] - fore_df["ci_lower"]) / fore_df["value"]).mean() * 100
    return ci_width


def calculate_forecast_quality(fore_df, metric_name=""):
    """
    Assess forecast quality based on variation and direction.

    Parameters
    ----------
    fore_df : pd.DataFrame
        Forecast DataFrame with value column
    metric_name : str
        Name of the metric for context

    Returns
    -------
    tuple
        (quality_rating, direction, variation_score)
    """
    if fore_df.empty or len(fore_df) < 3:
        return "Insufficient data", "Unknown", None

    values = fore_df["value"].values
    diff = np.diff(values)

    if len(diff) > 1:
        # Calculate coefficient of variation of changes
        mean_abs_change = np.mean(np.abs(diff))
        mean_value = np.mean(values)
        
        # Relative variation (as percentage of mean value)
        if mean_value > 0:
            rel_variation = (mean_abs_change / mean_value) * 100
        else:
            rel_variation = 0
        
        # Check if this is a volatile metric (breach flow, etc.)
        volatile_indicators = ["breach", "12-hour", "A&E 12-hour"]
        is_volatile = any(indicator in metric_name.lower() for indicator in volatile_indicators)
        
        # Build quality message with variation percentage
        if rel_variation < 0.5:
            quality = f"Very stable ({rel_variation:.1f}%)"
        elif rel_variation < 2.0:
            quality = f"Stable ({rel_variation:.1f}%)"
        elif rel_variation < 5.0:
            quality = f"Moderate variation ({rel_variation:.1f}%)"
        elif rel_variation < 12.0:
            quality = f"Typical health data variation ({rel_variation:.1f}%)"
        else:
            # High variation - could be due to volatility or over-fitting
            if is_volatile:
                quality = f"High variation: expected for volatile series ({rel_variation:.1f}%)"
            else:
                quality = f"High variation: possibly due to over-fitting or series volatility ({rel_variation:.1f}%)"
        
        variation = rel_variation
    else:
        quality = "Insufficient data for variation assessment"
        variation = None

    # Direction with tolerance for flat forecasts
    if len(values) > 1:
        first_val = values[0]
        last_val = values[-1]
        pct_change = ((last_val - first_val) / abs(first_val)) * 100 if first_val != 0 else 0
        if pct_change > 3.0:
            direction = "Increasing"
        elif pct_change < -3.0:
            direction = "Decreasing"
        else:
            direction = "Stable"
    else:
        direction = "Unknown"

    return quality, direction, variation


def main():
    """Main entry point for the Streamlit dashboard."""
    st.title("Health Systems Econometric Forecasting Dashboard")
    st.markdown("---")

    try:
        df, nice_df = load_data()

        if df is None:
            st.error(f"Missing master forecast data asset at: {FORECAST_PATH}.")
            return

        if df.empty:
            st.error("The forecast file is empty. Please run pipeline again.")
            return

        st.sidebar.header("Series Selection")

        if "metric" not in df.columns:
            st.error(f"Critical error: 'metric' column missing. Found columns instead: {list(df.columns)}")
            return

        unique_metrics = sorted(df["metric"].dropna().unique())
        selected_metric = st.sidebar.selectbox("Choose a system indicator to analyze:", unique_metrics)

        show_ci = st.sidebar.checkbox("Display 95% Forecast Confidence Intervals", value=True)
        show_nice = st.sidebar.checkbox("Overlay NICE QALY Policy Threshold Shifts", value=True)
        zoom_forecast = st.sidebar.checkbox("Zoom Y-Axis to Forecast Range", value=False)

        # Display warning for GP series with limited data
        if selected_metric.startswith("GP"):
            st.warning(
                "Data Limitation: GP appointments data is only available from October 2023. "
                "With only 11 quarterly observations, reliable forecasting is not possible. "
                "Only historical data is shown."
            )

        metric_df = df[df["metric"] == selected_metric].sort_values("quarter")

        hist_df = metric_df[metric_df["type"] == "history"]
        fore_df = metric_df[metric_df["type"] == "forecast"]

        fig = go.Figure()

        # Add historical data
        if not hist_df.empty:
            fig.add_trace(go.Scatter(
                x=hist_df["quarter"], y=hist_df["value"],
                mode="lines+markers", name="Observed History",
                line=dict(color="#1f77b4", width=2.5)
            ))

        # Add forecast data if available
        if not fore_df.empty and not selected_metric.startswith("GP"):
            if not hist_df.empty:
                last_hist = hist_df.iloc[-1:]
                plot_fore = pd.concat([last_hist, fore_df], ignore_index=True)
            else:
                plot_fore = fore_df

            fig.add_trace(go.Scatter(
                x=plot_fore["quarter"], y=plot_fore["value"],
                mode="lines+markers", name="Forecast Horizon",
                line=dict(color="#ff7f0e", width=2.5, dash="dash")
            ))

            # Add confidence intervals
            if show_ci and "ci_lower" in fore_df.columns and "ci_upper" in fore_df.columns:
                if not hist_df.empty:
                    plot_fore_ci = pd.concat([last_hist, fore_df], ignore_index=True)
                    plot_fore_ci["ci_lower"] = plot_fore_ci["ci_lower"].fillna(plot_fore_ci["value"])
                    plot_fore_ci["ci_upper"] = plot_fore_ci["ci_upper"].fillna(plot_fore_ci["value"])
                else:
                    plot_fore_ci = fore_df

                fig.add_trace(go.Scatter(
                    x=pd.concat([plot_fore_ci["quarter"], plot_fore_ci["quarter"].iloc[::-1]], ignore_index=True),
                    y=pd.concat([plot_fore_ci["ci_upper"], plot_fore_ci["ci_lower"].iloc[::-1]], ignore_index=True),
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.15)',
                    line=dict(color='rgba(255,127,14,0)'),
                    hoverinfo="skip",
                    name="95% Confidence Interval"
                ))
        elif selected_metric.startswith("GP") and not hist_df.empty:
            # Add annotation for GP series with no forecast
            last_date = hist_df["quarter"].iloc[-1]
            last_value = hist_df["value"].iloc[-1]
            fig.add_annotation(
                x=last_date,
                y=last_value * 0.9,
                text="Insufficient data for forecasting",
                showarrow=True,
                arrowhead=1,
                ax=50,
                ay=-30,
                font=dict(size=12, color="orange"),
                arrowcolor="orange"
            )

        # Add NICE policy annotations
        if show_nice and nice_df is not None and not metric_df.empty:
            min_date, max_date = metric_df["quarter"].min(), metric_df["quarter"].max()
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

        # Configure y-axis
        yaxis_settings = {"title": "Metric Values", "fixedrange": False}

        if zoom_forecast and not fore_df.empty and not selected_metric.startswith("GP"):
            range_values = [fore_df["value"]]
            if show_ci and "ci_lower" in fore_df.columns and "ci_upper" in fore_df.columns:
                range_values.append(fore_df["ci_lower"].fillna(fore_df["value"]))
                range_values.append(fore_df["ci_upper"].fillna(fore_df["value"]))
            all_vals = pd.concat(range_values)
            y_min, y_max = all_vals.min(), all_vals.max()
            pad = (y_max - y_min) * 0.1 if y_max > y_min else (abs(y_max) * 0.1 or 1)
            yaxis_settings["range"] = [y_min - pad, y_max + pad]
            yaxis_settings["autorange"] = False
        else:
            yaxis_settings["autorange"] = True

        # Update chart layout
        fig.update_layout(
            title=dict(text=f"System Tracking Matrix: {selected_metric}", font=dict(size=16)),
            xaxis_title="Timeline Horizon",
            yaxis=dict(
                title="Metric Values",
                autorange=True,
                fixedrange=False
            ),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=80, b=40, l=40, r=40),
            height=550
        )

        st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # SUMMARY STATISTICS SECTION
        # ============================================================

        st.markdown("---")
        st.subheader("Summary Statistics")

        # Row 1: Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if not hist_df.empty:
                range_years = (hist_df["quarter"].max() - hist_df["quarter"].min()).days / 365.25
                st.metric(label="Historical Range", value=f"{range_years:.1f} Years")
            else:
                st.metric(label="Historical Range", value="N/A")

        with col2:
            if not fore_df.empty and not selected_metric.startswith("GP"):
                st.metric(label="Forecast Horizon", value=f"{len(fore_df)} Quarters")
            else:
                st.metric(label="Forecast Horizon", value="0")

        with col3:
            if not fore_df.empty and not hist_df.empty and not selected_metric.startswith("GP"):
                growth = ((fore_df["value"].iloc[-1] - hist_df["value"].iloc[-1]) / hist_df["value"].iloc[-1]) * 100
                st.metric(label="Total Forecast Change", value=f"{growth:+.1f}%")
            else:
                st.metric(label="Total Forecast Change", value="N/A")

        with col4:
            if not fore_df.empty and show_ci:
                ci_width = calculate_ci_width(fore_df)
                if ci_width is not None:
                    st.metric(label="Avg CI Width", value=f"±{ci_width:.1f}%")
                else:
                    st.metric(label="Avg CI Width", value="N/A")
            else:
                st.metric(label="Avg CI Width", value="N/A")

        # Row 2: Additional details
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if not hist_df.empty:
                st.metric(label="Historical Observations", value=f"{len(hist_df)}")
            else:
                st.metric(label="Historical Observations", value="0")

        with col2:
            if not fore_df.empty and not selected_metric.startswith("GP"):
                st.metric(label="Forecast Points", value=f"{len(fore_df)}")
            else:
                st.metric(label="Forecast Points", value="0")

        with col3:
            if not fore_df.empty and not selected_metric.startswith("GP"):
                quality, direction, variation = calculate_forecast_quality(fore_df, selected_metric)
                st.metric(label="Forecast Direction", value=direction)
            else:
                st.metric(label="Forecast Direction", value="N/A")

        with col4:
            if not fore_df.empty and not selected_metric.startswith("GP"):
                quality, direction, variation = calculate_forecast_quality(fore_df, selected_metric)
                st.metric(label="Forecast Quality", value=quality)
            else:
                st.metric(label="Forecast Quality", value="N/A")

        # Row 3: Data completeness and details
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Historical Data**")
            if not hist_df.empty:
                st.write(f"- Start: {hist_df['quarter'].min().strftime('%Y-%m-%d')}")
                st.write(f"- End: {hist_df['quarter'].max().strftime('%Y-%m-%d')}")
                st.write(f"- Min: {hist_df['value'].min():,.0f}")
                st.write(f"- Max: {hist_df['value'].max():,.0f}")
                st.write(f"- Mean: {hist_df['value'].mean():,.0f}")
            else:
                st.write("No historical data available")

        with col2:
            st.markdown("**Forecast Data**")
            if not fore_df.empty and not selected_metric.startswith("GP"):
                st.write(f"- Start: {fore_df['quarter'].min().strftime('%Y-%m-%d')}")
                st.write(f"- End: {fore_df['quarter'].max().strftime('%Y-%m-%d')}")
                st.write(f"- Final value: {fore_df['value'].iloc[-1]:,.0f}")
                if not hist_df.empty:
                    st.write(f"- Change: {((fore_df['value'].iloc[-1] - hist_df['value'].iloc[-1]) / hist_df['value'].iloc[-1]) * 100:+.1f}%")
                # Forecast quality
                if len(fore_df) > 2:
                    quality, direction, variation = calculate_forecast_quality(fore_df, selected_metric)
                    st.write(f"- Variation: {quality}")
            else:
                st.write("No forecast available")

        with col3:
            st.markdown("**Confidence Intervals**")
            if not fore_df.empty and show_ci and "ci_lower" in fore_df.columns and "ci_upper" in fore_df.columns:
                ci_lower = fore_df["ci_lower"].min()
                ci_upper = fore_df["ci_upper"].max()
                st.write(f"- CI lower: {ci_lower:,.0f}")
                st.write(f"- CI upper: {ci_upper:,.0f}")
                ci_width = calculate_ci_width(fore_df)
                if ci_width is not None:
                    st.write(f"- Avg width: ±{ci_width:.1f}%")
                if not fore_df.empty:
                    narrowest = ((fore_df["ci_upper"] - fore_df["ci_lower"]) / fore_df["value"]).min() * 100
                    widest = ((fore_df["ci_upper"] - fore_df["ci_lower"]) / fore_df["value"]).max() * 100
                    st.write(f"- Narrowest: ±{narrowest:.1f}%")
                    st.write(f"- Widest: ±{widest:.1f}%")
            else:
                st.write("No confidence intervals available")

    except Exception as e:
        st.error("An execution error occurred within the dashboard rendering script:")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
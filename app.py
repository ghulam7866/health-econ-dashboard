"""
app.py
------
Frontend Streamlit interactive dashboard application.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import traceback

ROOT_DIR = Path(__file__).parent.resolve()
FORECAST_PATH = ROOT_DIR / "data" / "processed" / "dashboard_forecasts.csv"
NICE_PATH = ROOT_DIR / "data" / "processed" / "nice_clean.csv"

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
        
    return df, nice_df

def main():
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
        
        # ============================================================
        # ADD GP DATA LIMITATION MESSAGE HERE
        # ============================================================
        if selected_metric.startswith("GP"):
            st.warning(
                "**DATA LIMITATION**: GP appointments data is only available from October 2023. "
                "With only 11 quarterly observations, reliable forecasting is not possible. "
                "Only historical data is shown. "
            )
        
        metric_df = df[df["metric"] == selected_metric].sort_values("quarter")
        
        hist_df = metric_df[metric_df["type"] == "history"]
        fore_df = metric_df[metric_df["type"] == "forecast"]
        
        # Create the figure ONCE
        fig = go.Figure()
        
        # Always show historical data
        if not hist_df.empty:
            fig.add_trace(go.Scatter(
                x=hist_df["quarter"], y=hist_df["value"],
                mode="lines+markers", name="Observed History",
                line=dict(color="#1f77b4", width=2.5)
            ))
        
        # Only show forecast if there is forecast data AND it's not a GP series
        # For GP series, we skip forecasts entirely
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
            # For GP series with no forecast, add an annotation
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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Historical Base Entries", value=f"{len(hist_df)} Quarters")
        with col2:
            # Show 0 for GP series forecasts
            if selected_metric.startswith("GP"):
                st.metric(label="Projected Forecast Steps", value="0 (Data Limited)")
            else:
                st.metric(label="Projected Forecast Steps", value=f"{len(fore_df)} Quarters")
        with col3:
            if not fore_df.empty and not hist_df.empty and not selected_metric.startswith("GP"):
                growth = ((fore_df["value"].iloc[-1] - hist_df["value"].iloc[-1]) / hist_df["value"].iloc[-1]) * 100
                st.metric(label="Net Estimated Trend Drift", value=f"{growth:+.1f}%")
            else:
                st.metric(label="Net Estimated Trend Drift", value="N/A (Data Limited)")

    except Exception as e:
        st.error("An execution error occurred within the dashboard rendering script:")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
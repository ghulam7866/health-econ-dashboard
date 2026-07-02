"""
GP & A&E Classical Model Comparison (Stable)
============================================

Purpose
-------
Compare classical forecasting models for the short quarterly GP and
A&E series to determine which provides the best out-of-sample fit.

Models compared
---------------
- Random Walk
- Random Walk + Drift
- Simple Exponential Smoothing (SES)
- Holt Linear Trend
- ETS
- ARMA
- ARIMA
- SARIMA

Outputs
-------
- RMSE, MAE, MAPE, Bias
- model_comparison_results.csv
- best_models.csv
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    ExponentialSmoothing
)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =====================================================
# Project configuration
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR /
    "data" /
    "processed" /
    "combined_quarterly.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR /
    "reports" /
    "gp_ae_model_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# Import metric names
# =====================================================

sys.path.insert(
    0,
    str(PROJECT_DIR)
)

import exog_config as ec

METRIC_NAMES = ec.METRIC_NAMES


TARGET_SERIES = [
    "A&E attendances (flow)",
    "GP total appointments (flow)",
    "GP face-to-face appointments (flow)",
    "GP telephone appointments (flow)"
]

SEASONAL_PERIOD = 4
MIN_TRAIN_SIZE = 6
FORECAST_HORIZON = 1

# Maximum number of observations for SARIMA (to prevent explosion)
MAX_SARIMA_OBS = 30


def load_series(metric_key):
    """
    Load a single quarterly series from the
    combined quarterly dataset.
    """

    df = pd.read_csv(INPUT_FILE)

    df["quarter"] = pd.to_datetime(df["quarter"])

    metric_name = METRIC_NAMES[metric_key]

    subset = (
        df
        [df["metric"] == metric_name]
        .dropna(subset=["value"])
        .sort_values("quarter")
    )

    if subset.empty:
        raise ValueError(
            f"No data found for '{metric_key}' "
            f"({metric_name})"
        )

    return subset.reset_index(drop=True)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    safe_actual = np.where(
        actual == 0,
        np.nan,
        actual
    )

    mape = np.nanmean(
        np.abs(
            (actual - predicted)
            / safe_actual
        )
    ) * 100

    bias = np.mean(
        predicted - actual
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Bias": bias
    }


MODELS = [
    "Random Walk",
    "Random Walk + Drift",
    "SES",
    "Holt",
    "ETS",
    "ARMA",
    "ARIMA",
    "SARIMA"
]


# =====================================================
# Model Selection
# =====================================================

ARMA_ORDERS = [
    (1,0,1),
    (2,0,1),
    (1,0,2)
]

ARIMA_ORDERS = [
    (0,1,1),
    (1,1,0),
    (1,1,1),
    (2,1,1),
    (2,1,2)
]

SARIMA_ORDERS = [
    (
        (1,1,1),
        (1,0,1,SEASONAL_PERIOD)
    ),
    (
        (0,1,1),
        (1,0,1,SEASONAL_PERIOD)
    ),
    (
        (1,1,0),
        (1,0,1,SEASONAL_PERIOD)
    )
]


def best_arma(train):
    best_fit = None
    best_aic = np.inf

    for order in ARMA_ORDERS:
        try:
            fit = ARIMA(
                train,
                order=order
            ).fit(method_kwargs={'maxiter': 200})
            if fit.aic < best_aic:
                best_fit = fit
                best_aic = fit.aic
        except Exception:
            continue

    return best_fit


def best_arima(train):
    best_fit = None
    best_aic = np.inf

    for order in ARIMA_ORDERS:
        try:
            fit = ARIMA(
                train,
                order=order
            ).fit(method_kwargs={'maxiter': 200})
            if fit.aic < best_aic:
                best_fit = fit
                best_aic = fit.aic
        except Exception:
            continue

    return best_fit


def best_sarima(train):
    """Fit SARIMA with safeguards for stability"""
    # Skip SARIMA for very short series or too many observations
    n = len(train)
    if n < 2 * SEASONAL_PERIOD or n > MAX_SARIMA_OBS:
        return None
    
    best_fit = None
    best_aic = np.inf
    
    # Only try a subset of orders for stability
    safe_orders = [
        ((0,1,1), (1,0,1,SEASONAL_PERIOD)),
        ((1,1,0), (1,0,1,SEASONAL_PERIOD)),
    ]
    
    for order, seasonal in safe_orders:
        try:
            fit = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
                simple_differencing=False  # Better for short series
            ).fit(
                disp=False,
                method='bfgs',  # More stable optimizer
                maxiter=100,
                full_output=False
            )
            
            # Check for numerical stability
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                # Validate forecast to catch explosive models
                try:
                    test_forecast = fit.forecast(steps=1).iloc[0]
                    if np.isfinite(test_forecast) and abs(test_forecast) < 1e10:
                        best_fit = fit
                        best_aic = fit.aic
                except:
                    continue
                    
        except Exception:
            continue

    return best_fit


def rolling_backtest(
    series,
    model_name,
    horizon=1,
    min_train=6
):
    predictions = []
    actuals = []

    n = len(series)

    for end in range(
        min_train,
        n - horizon + 1
    ):
        train = series.iloc[:end]
        actual = series.iloc[
            end + horizon - 1
        ]

        try:
            # ---------------------------------
            if model_name == "Random Walk":
                forecast = train.iloc[-1]

            # ---------------------------------
            elif model_name == "Random Walk + Drift":
                drift = (
                    train.iloc[-1]
                    - train.iloc[0]
                ) / (len(train)-1) if len(train) > 1 else 0
                forecast = (
                    train.iloc[-1]
                    + drift * horizon
                )

            # ---------------------------------
            elif model_name == "SES":
                fit = SimpleExpSmoothing(
                    train
                ).fit()
                forecast = fit.forecast(
                    horizon
                ).iloc[-1]

            # ---------------------------------
            elif model_name == "Holt":
                # Use simpler trend for short series
                if len(train) < 10:
                    fit = ExponentialSmoothing(
                        train,
                        trend=None,  # No trend for very short series
                        seasonal=None
                    ).fit()
                else:
                    fit = ExponentialSmoothing(
                        train,
                        trend="add",
                        seasonal=None
                    ).fit()
                forecast = fit.forecast(
                    horizon
                ).iloc[-1]

            # ---------------------------------
            elif model_name == "ETS":
                # ETS stability guard for short series
                if len(train) < 12:
                    # fallback: SES (no trend, no seasonality)
                    fit = ExponentialSmoothing(
                        train,
                        trend=None,
                        seasonal=None
                    ).fit()
                else:
                    # seasonal ETS only when enough data exists
                    fit = ExponentialSmoothing(
                        train,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=SEASONAL_PERIOD
                    ).fit()
                forecast = fit.forecast(horizon).iloc[-1]

            # ---------------------------------
            elif model_name == "ARMA":
                fit = best_arma(train)
                if fit is None:
                    # Fallback to simple model if ARMA fails
                    forecast = train.iloc[-1]
                else:
                    forecast = fit.forecast(
                        horizon
                    ).iloc[-1]

            # ---------------------------------
            elif model_name == "ARIMA":
                fit = best_arima(train)
                if fit is None:
                    # Fallback to simple model if ARIMA fails
                    forecast = train.iloc[-1]
                else:
                    forecast = fit.forecast(
                        horizon
                    ).iloc[-1]

            # ---------------------------------
            elif model_name == "SARIMA":
                fit = best_sarima(train)
                if fit is None:
                    # Skip SARIMA if it fails or would be unstable
                    continue
                else:
                    forecast = fit.forecast(
                        horizon
                    ).iloc[-1]
                    # Validate forecast
                    if not np.isfinite(forecast) or abs(forecast) > 1e10:
                        continue

            else:
                continue

            # Validate forecast
            if np.isnan(forecast) or np.isinf(forecast):
                continue

            predictions.append(
                forecast
            )
            actuals.append(
                actual
            )

        except Exception:
            continue

    # Check if we have enough predictions
    if len(predictions) == 0:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "Bias": np.nan,
            "Model": model_name,
            "Num_Predictions": 0
        }

    metrics = calculate_metrics(
        actuals,
        predictions
    )
    metrics["Model"] = model_name
    metrics["Num_Predictions"] = len(predictions)

    return metrics


# =====================================================
# Main execution
# =====================================================

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("GP & A&E Classical Model Comparison")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("Running forecast models...")
    print("-" * 40)
    
    results = []
    
    for metric_key in TARGET_SERIES:
        print(f"\nProcessing: {metric_key}")
        
        # Load the series
        series_data = load_series(metric_key)
        series = series_data["value"]
        
        print(f"  Series length: {len(series)}")
        
        # Run each model
        for model_name in MODELS:
            print(f"  Running {model_name}...")
            
            metrics = rolling_backtest(
                series=series,
                model_name=model_name,
                horizon=FORECAST_HORIZON,
                min_train=MIN_TRAIN_SIZE
            )
            
            # Add metadata
            metrics["Metric_Key"] = metric_key
            metrics["Series_Length"] = len(series)
            
            results.append(metrics)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out models with zero predictions or unreasonable values
    results_df = results_df[results_df["Num_Predictions"] > 0]
    results_df = results_df[results_df["RMSE"] < 1e12]  # Remove explosive models
    
    # Save full comparison results
    print("\n" + "-" * 40)
    print("Saving results...")
    
    results_df.to_csv(
        OUTPUT_DIR / "model_comparison_results.csv",
        index=False
    )
    print(f"  Saved: model_comparison_results.csv")
    
    # Find best models for each metric and series
    best_models = []
    
    for metric_key in TARGET_SERIES:
        series_results = results_df[results_df["Metric_Key"] == metric_key]
        
        if series_results.empty:
            print(f"  Warning: No valid results for {metric_key}")
            continue
        
        # For each metric, find the best model
        for metric in ['RMSE', 'MAE', 'MAPE', 'Bias']:
            if metric == 'Bias':
                # For Bias, we want the one closest to zero
                best_idx = series_results[metric].abs().idxmin()
            else:
                best_idx = series_results[metric].idxmin()
            
            best_model = series_results.loc[best_idx].to_dict()
            best_model["Metric"] = metric
            best_models.append(best_model)
    
    best_models_df = pd.DataFrame(best_models)
    best_models_df.to_csv(
        OUTPUT_DIR / "best_models.csv",
        index=False
    )
    print(f"  Saved: best_models.csv")
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS BY SERIES")
    print("=" * 60)
    
    for metric_key in TARGET_SERIES:
        print(f"\n{metric_key}:")
        print("-" * 40)
        
        series_results = results_df[results_df["Metric_Key"] == metric_key]
        
        if series_results.empty:
            print("  No valid results")
            continue
        
        # Sort by RMSE (best first)
        series_results_sorted = series_results.sort_values("RMSE")
        
        print(f"  {'Model':<20} {'RMSE':>12} {'MAE':>12} {'MAPE':>10} {'Bias':>12} {'N':>5}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*5}")
        
        for _, row in series_results_sorted.iterrows():
            print(f"  {row['Model']:<20} "
                  f"{row['RMSE']:>12.0f} "
                  f"{row['MAE']:>12.0f} "
                  f"{row['MAPE']:>10.1f} "
                  f"{row['Bias']:>12.0f} "
                  f"{row['Num_Predictions']:>5.0f}")
    
    # Print best models
    print("\n" + "=" * 60)
    print("BEST MODELS BY SERIES AND METRIC")
    print("=" * 60)
    
    if not best_models_df.empty:
        for metric_key in TARGET_SERIES:
            print(f"\n{metric_key}:")
            print("-" * 40)
            
            series_best = best_models_df[best_models_df["Metric_Key"] == metric_key]
            
            if series_best.empty:
                print("  No best models found")
                continue
                
            for _, row in series_best.iterrows():
                metric_value = row[row['Metric']]
                if not np.isnan(metric_value) and metric_value < 1e12:
                    print(f"  Best for {row['Metric']}: {row['Model']} "
                          f"({row['Metric']}: {metric_value:,.0f})")
                else:
                    print(f"  Best for {row['Metric']}: {row['Model']} "
                          f"(value: {metric_value:.2e})")
    else:
        print("\nNo best models found.")
    
    # Summary statistics (excluding SARIMA due to instability)
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (excluding unstable SARIMA)")
    print("=" * 60)
    
    # Count how many times each model is best
    if not best_models_df.empty:
        model_counts = best_models_df["Model"].value_counts()
        print("\nModel selection frequency (across all metrics and series):")
        for model, count in model_counts.items():
            print(f"  {model}: {count} times")
        
        # Average metrics by model (exclude SARIMA and unstable values)
        stable_models = results_df[
            results_df["Model"] != "SARIMA"
        ]
        avg_metrics = stable_models.groupby("Model")[['RMSE', 'MAE', 'MAPE', 'Bias']].mean()
        print("\nAverage metrics by model (across all series):")
        for idx, row in avg_metrics.iterrows():
            print(f"  {idx}: RMSE={row['RMSE']:,.0f}, MAE={row['MAE']:,.0f}, "
                  f"MAPE={row['MAPE']:.1f}%, Bias={row['Bias']:,.0f}")
    else:
        print("\nNo models to summarize.")
    
    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    # Most frequent best model
    if not best_models_df.empty:
        best_model_counts = best_models_df["Model"].value_counts()
        top_model = best_model_counts.index[0]
        top_count = best_model_counts.iloc[0]
        print(f"\n• Most frequently selected model: {top_model} "
              f"(best in {top_count} out of {len(best_models_df)} metrics)")
        
        # Best model for each series (based on RMSE)
        print("\n• Recommended models by series (based on RMSE):")
        for metric_key in TARGET_SERIES:
            series_results = results_df[results_df["Metric_Key"] == metric_key]
            if not series_results.empty:
                best_rmse = series_results.loc[series_results["RMSE"].idxmin()]
                print(f"  - {metric_key}: {best_rmse['Model']} "
                      f"(RMSE: {best_rmse['RMSE']:,.0f})")
    
    print("\n" + "=" * 60)
    print(f"Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return results_df, best_models_df


if __name__ == "__main__":
    results_df, best_models_df = main()
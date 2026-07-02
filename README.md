# Health Economics Forecasting Dashboard

A comprehensive forecasting pipeline for UK health system metrics including RTT waiting lists, A&E attendances, workforce FTE, bed occupancy, and GP appointments.

## Overview

This dashboard provides quarterly forecasts for key NHS metrics using SARIMAX models with exogenous variables. The pipeline automatically scrapes data from NHS, ONS, and HMT sources, cleans and aligns it, and generates forecasts with confidence intervals.

## Features

- **Automated Data Pipeline**: Scrapes, cleans, and aligns data from multiple sources
- **SARIMAX Forecasting**: Advanced time series models with exogenous variables
- **Interactive Dashboard**: Streamlit-based visualization with NICE policy annotations
- **Model Diagnostics**: Comprehensive residual testing and forecast validation
- **Flexible Configuration**: Easy to add new metrics or modify existing models

## Metrics

| Metric | Model | Key Feature |
|--------|-------|-------------|
| RTT waiting list | MA(1) + quadratic_trend | Log transformation |
| A&E attendances | AR(1) with trend | Moving average smoothing |
| Workforce FTE | AR(2) + quadratic_trend | All exog significant |
| Bed occupancy | MA(1) + quadratic_trend | All exog significant |
| RTT % within 18 weeks | Random walk with drift | Smoothed transition |
| A&E 12-hour breach | Random walk + quadratic_trend | Stable forecast |

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/health-econ-dashboard.git
cd health-econ-dashboard

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Full Pipeline

```bash
python run_pipeline.py
```

### Launch the Dashboard

```bash
streamlit run app.py
```

### Run Individual Components

```bash
# Run diagnostics for a specific metric
python stress_test.py "RTT waiting list (level)"

# Validate forecasts
python validate_forecasts.py

# Run residual diagnostics
python test_residuals.py
```

## Project Structure

```
health-econ-dashboard/
├── app.py                 # Streamlit dashboard
├── run_pipeline.py        # Master pipeline runner
├── src/
│   ├── exog_config.py     # Model configuration
│   ├── master_forecast_engine.py  # Main forecasting engine
│   ├── align_merge.py     # Data alignment
│   ├── cleaner.py         # Data cleaning
│   ├── add_intervention_dummies.py  # COVID dummies
│   └── pesa_annual_pipeline.py  # PESA annual pipeline
├── tests/
│   ├── stress_test.py     # Individual metric diagnostics
│   ├── test_residuals.py  # Residual diagnostics
│   └── validate_forecasts.py  # Forecast validation
├── scripts/
│   ├── test_scraper.py    # Data scraper
│   ├── nice_reference.py  # NICE reference table
│   └── spot_check.py      # Data validation
├── data/
│   ├── raw/               # Raw data (not tracked)
│   └── processed/         # Processed data (not tracked)
└── reports/               # Generated reports (not tracked)
```

## Requirements

- Python 3.11+
- pandas >= 2.0.0
- numpy >= 1.24.0
- statsmodels >= 0.14.0
- streamlit >= 1.28.0
- plotly >= 5.17.0
- requests >= 2.31.0
- openpyxl >= 3.1.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0

See `requirements.txt` for full list.

## Data Sources

- **NHS England**: RTT waiting times, A&E attendances, Workforce FTE, Bed occupancy
- **NHS Digital**: GP appointments
- **ONS**: UK population estimates
- **HM Treasury**: PESA Health expenditure
- **NICE**: QALY cost-effectiveness thresholds

## Model Methodology

All models were selected through comprehensive diagnostic testing including:

- **Stationarity tests**: ADF, KPSS, ADF-GLS (ERS)
- **Seasonal unit root tests**: HEGY
- **AICc grid search**: For optimal ARIMA orders
- **AR/MA root stability**: To ensure model invertibility
- **Residual diagnostics**: Ljung-Box, Jarque-Bera, ARCH tests
- **Backtesting**: Rolling-origin expanding window validation

## Model Documentation

| Metric | Order | Seasonal Order | Trend | Transformation | Exogenous Variables |
|--------|-------|----------------|-------|----------------|---------------------|
| RTT waiting list | (0,1,1) | (1,0,1,4) | c | Log | quadratic_trend |
| A&E attendances | (1,0,0) | (0,0,0,4) | c | None | None |
| Workforce FTE | (2,1,0) | (1,0,1,4) | c | None | covid_pulse, post_covid_trend_break, quadratic_trend |
| Bed occupancy | (0,1,1) | (1,0,1,4) | c | None | covid_pulse, post_covid_regime, quadratic_trend |
| RTT % within 18 weeks | (0,1,0) | (0,0,0,4) | c | None | None |
| A&E 12-hour breach | (0,1,0) | (1,0,0,4) | None | None | quadratic_trend |

## Pipeline Outputs

| Output | Description |
|--------|-------------|
| `dashboard_forecasts.csv` | Combined historical and forecast data for the dashboard |
| `combined_quarterly.csv` | Quarterly aligned data with exogenous variables |
| `rtt_clean.csv` | Cleaned RTT waiting times data |
| `ae_clean.csv` | Cleaned A&E attendances data |
| `gp_appointments_clean.csv` | Cleaned GP appointments data |
| `workforce_clean.csv` | Cleaned workforce FTE data |
| `beds_clean.csv` | Cleaned bed occupancy data |
| `pesa_clean.csv` | Cleaned PESA health expenditure data |

## License

MIT

## Author

ghulam7866

## Acknowledgments

- NHS England for open data access
- NICE for QALY threshold guidance
- HM Treasury for PESA data
- ONS for population estimates
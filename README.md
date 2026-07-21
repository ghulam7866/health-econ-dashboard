# Health Economics Forecasting Dashboard

A rigorous, audit‑ready forecasting pipeline for UK health system metrics,
developed as part of a personal project portfolio. The pipeline produces
quarterly forecasts for nine NHS performance indicators using SARIMAX models
with an Interrupted Time Series (ITS) design, validated through rolling‑origin
backtesting.

## Overview

The project automates the full workflow:

1. **Data ingestion** – scrapes official statistics from NHS England, NHS Digital, ONS, and HM Treasury.
2. **Cleaning & alignment** – standardises disparate sources onto a common quarterly calendar.
3. **Model selection & fitting** – fits SARIMAX (or logit‑ARMA / random‑walk) models with COVID‑era structural breaks, using a "flag‑don't‑override" governance rule.
4. **Forecast & counterfactual generation** – produces 6‑year forecasts and ITS counterfactuals (what‑if without the break).
5. **Interactive dashboard** – built with Streamlit and Plotly, displaying history, forecasts, confidence intervals, NICE QALY thresholds, and counterfactual lines.

## Features

- **Interrupted Time Series (ITS)** – formal segmented regression for post‑COVID trend breaks
- **Heteroskedasticity‑robust standard errors** – via statsmodels' `cov_type='robust'`
- **Rolling‑origin backtesting** – 46–69 folds per series, with Kupiec (1995) coverage tests
- **Counterfactual analysis** – "no‑break" projections for capacity‑planning insights
- **System Strain Overview** – single‑page summary of trends, directions, and confidence ratings
- **Modular pipeline** – each stage can be run independently or via `run_pipeline.py`

## Metrics & Model Specifications

| Metric | Specification | Transform | Sigma Scale | Exogenous Variables |
|--------|--------------|-----------|-------------|---------------------|
| RTT waiting list (level) | ARIMA(0,1,1)×(1,0,1,4), trend='n' | log | 1.25 | `post_covid_trend_break` |
| A&E attendances (flow) | ARIMA(1,1,0)×(0,0,0,4), trend='c' | – | 1.5 | – |
| Workforce FTE (level) | ARIMA(2,1,0)×(1,0,1,4), trend='n' | – | 1.45 | `post_covid_trend_break`, `covid_pulse` |
| Nurse FTE (level) | ARIMA(2,1,0)×(1,0,1,4), trend='n' | log | 1.4 | `post_covid_trend_break` |
| Doctor FTE (level) | ARIMA(2,1,0)×(2,0,1,4), trend='c' | log | 1.8 | `post_covid_trend_break` |
| Bed occupancy (level) | ARIMA(1,1,1)×(1,0,1,4), trend='n' | – | 1.8 | `covid_pulse`, `post_covid_regime` |
| RTT % within 18 weeks | Logit ARMA(1,1), trend='n' | logit | 1.25 | `post_covid_trend_break` |
| A&E 12‑hour breach (flow) | ARIMA(2,1,0)×(0,0,0,4), trend=None, 2021+ window | – | 1.0 | – |
| PESA Health spend (level) | Random walk with drift | – | 1.0 | – |

*GP appointment series are excluded from forecasting due to insufficient history (11 quarterly observations).*

## Key Model Diagnostics

- **Stationarity tests:** ADF, KPSS, ADF‑GLS (DF‑GLS)
- **Residual diagnostics:** Ljung‑Box (autocorrelation), Jarque‑Bera (normality), ARCH‑LM (heteroskedasticity)
- **Structural break:** Coefficients reported with heteroskedasticity‑robust p‑values
- **Interval calibration:** Empirical sigma‑scale multipliers (1.0–1.8×) and t‑distribution‑based prediction intervals

## Known Limitations (Documented)

- **GP appointments** – excluded from production forecasting (insufficient data).
- **A&E 12‑hour breach** – small sample (22 obs post‑2021); forecast horizon capped at 8 quarters.
- **HAC‑robust standard errors** – not currently available for ITS coefficients (custom estimator requires `t` as an exogenous regressor, which no specification includes). Inference relies on heteroskedasticity‑robust (HC) SEs.
- **Counterfactual extrapolation** – ramp‑variable counterfactuals are bounded at 4 quarters to avoid explosive behaviour; percentage gaps for ramp‑affected series are not quoted as single headline figures (see technical write‑up).

## Installation

```bash
git clone https://github.com/ghulam7866/health-econ-dashboard.git
cd health-econ-dashboard
pip install -r requirements.txt
```

## Usage

### Run the full pipeline

```bash
python run_pipeline.py
```

### Launch the interactive dashboard

```bash
streamlit run app.py
```

### Run backtest for a single metric

```bash
python stress_test.py "Bed occupancy (level)"
```

### Standalone analyses

```bash
python capacity_gap.py            # actual vs counterfactual gaps
python bed_occ_kupiec.py          # Kupiec test for bed occupancy
python check_residuals.py         # residual kurtosis & Ljung‑Box
```

## Data Sources

- **NHS England** – RTT waiting times, A&E attendances, workforce statistics, bed occupancy
- **NHS Digital** – GP appointments
- **ONS** – UK population estimates
- **HM Treasury** – PESA Chapter 4 (health expenditure)
- **NICE** – QALY cost‑effectiveness thresholds

## Project Structure

```text
health-econ-dashboard/
├── app.py                          # Streamlit dashboard
├── run_pipeline.py                 # End‑to‑end pipeline runner
├── test_scraper.py                 # Raw data scraper
├── append_ae_june.py               # One‑off A&E data merge (June 2026)
├── stress_test.py                  # Rolling‑origin backtest engine
├── requirements.txt
├── README.md
├── src/
│   ├── exog_config.py              # Single source of truth for model specs
│   ├── master_forecast_engine.py   # Production forecasting engine
│   ├── align_merge.py              # Quarterly resampling & alignment
│   ├── cleaner.py                  # Data cleaning & extraction
│   ├── add_intervention_dummies.py # COVID intervention dummies
│   └── pesa_annual_pipeline.py     # PESA annual forecast pipeline
├── scripts/
│   ├── analysis/                   # Standalone analysis scripts
│   └── diagnostics/                # Diagnostic scripts
├── data/
│   ├── raw/                        # Raw data (not tracked)
│   └── processed/                  # Processed data (not tracked)
└── docs/                           # Executive summary & technical write‑up
```

## License

MIT

## Author

ghulam7866
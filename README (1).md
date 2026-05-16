# India Electricity Demand Forecast

A time series forecasting web application that predicts India's daily electricity demand using 10 years of real POSOCO grid data (2013-2023).

## Live Demo
[Deployed on Hugging Face Spaces](#) ← update this link after deployment

## Project Overview
India is the third largest electricity consumer in the world. Grid operators must forecast demand accurately every day — under-forecasting causes blackouts, over-forecasting wastes capacity.

This app uses **SARIMA(1,1,1)(1,1,1,12)** — the best performing model from a comparison of 5 models including ARIMA, Holt-Winters, Prophet, and LSTM.

## Features
- Forecast electricity demand up to 24 months ahead
- Select from 6 regions: All India, North, West, South, East, North-East
- 95% confidence interval on all forecasts
- Full historical series visualization (2013-2023)
- COVID-19 anomaly handled in preprocessing

## Model Performance (Test Set: 2021-2023)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| ARIMA(1,1,1) | 360.95 | 438.28 | 8.43% |
| Holt-Winters | 274.62 | 344.45 | 7.07% |
| Prophet | 378.80 | 451.17 | 8.77% |
| LSTM | 429.68 | 509.85 | 9.96% |
| **SARIMA(1,1,1)(1,1,1,12)** | **247.61** | **292.61** | **5.90%** |

## Dataset
- **Source:** POSOCO via Kaggle — Daily Power Generation in India
- **Period:** 31 March 2013 to 31 March 2023
- **Records:** 3,429 daily observations
- **Regions:** All India, NR, WR, SR, ER, NER
- **Unit:** Million Units (MU) per day

## How to Run Locally

```bash
git clone https://github.com/yourusername/electricity-demand-forecast
cd electricity-demand-forecast
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
electricity-demand-forecast/
├── app.py
├── requirements.txt
├── README.md
└── Daily_Power_Gen_Source_march_23.csv
```

## Tech Stack
Python · Streamlit · Statsmodels · Pandas · Matplotlib

## Author
B.Tech 4th Year | Time Series Analysis Project

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from forecast import forecast_arima, forecast_prophet, forecast_rf

# Load data
df = pd.read_csv("./data/daily_revenue_processed.csv", parse_dates=["date"])
df.set_index("date", inplace=True)
ts = df['daily_revenue'].asfreq('D').fillna(0)

# Sidebar controls
st.sidebar.title("Forecast Settings")
model = st.sidebar.selectbox("Select Model", ["ARIMA", "Prophet", "Random Forest"])
period = st.sidebar.slider("Days to Forecast", 7, 90, 30)

st.title("Time Series Forecasting Dashboard")
st.subheader("Historical Revenue")
st.line_chart(ts)

# Forecast
st.subheader(f"Forecast using {model}")
if model == "ARIMA":
    forecast = forecast_arima(ts, period)
elif model == "Prophet":
    forecast = forecast_prophet(ts, period)["yhat"]
else:
    forecast = forecast_rf(ts, period)

# Combine actual + forecast
combined = pd.concat([ts, forecast])
combined.name = "Revenue Forecast"
st.line_chart(combined)

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor

def forecast_arima(series, steps):
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def forecast_prophet(series, steps):
    df = series.reset_index()
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].set_index('ds')[-steps:]

def forecast_rf(series, steps):
    df = pd.DataFrame(series)
    for i in range(1, 8):
        df[f'lag_{i}'] = df['daily_revenue'].shift(i)
    df.dropna(inplace=True)

    X = df.drop('daily_revenue', axis=1)
    y = df['daily_revenue']

    rf = RandomForestRegressor()
    rf.fit(X, y)

    preds = []
    last_input = X.iloc[-1].values
    for _ in range(steps):
        X_pred = pd.DataFrame([last_input], columns=X.columns)
        pred = rf.predict(X_pred)[0]
        preds.append(pred)
        last_input = [pred] + list(last_input[:-1])

    future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.Series(preds, index=future_dates)

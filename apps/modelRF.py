import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st
import talib


def app():
    st.title('Model - Random Forest')

    start = '2004-08-18'
    end = '2022-01-27'

    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil', 'GOOG')

    stock_data = data.DataReader(user_input, 'yahoo', start, end)

    # Describiendo los datos

    st.subheader('Datos del 2004 al 2022')
    st.write(stock_data.describe())

    # Visualizaciones
    st.subheader('Precio de cierre ajustado')
    stock_data['Adj Close'].plot()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Adj Close'])
    plt.ylabel("Adjusted Close Prices")
    st.pyplot(fig)

    st.subheader('Cambio porcentual de cierre ajustado de 1 día')
    fig = plt.figure(figsize=(12, 6))
    plt.hist(stock_data['Adj Close'].pct_change(), bins=50)
    plt.ylabel("Frecuencia")
    plt.xlabel("Cambio porcentual de cierre ajustado de 1 día")
    st.pyplot(fig)

    feature_names = []
    for n in [14, 30, 50, 200]:
        stock_data['ma' +
                   str(n)] = talib.SMA(stock_data['Adj Close'].values, timeperiod=n)
        stock_data['rsi' +
                   str(n)] = talib.RSI(stock_data['Adj Close'].values, timeperiod=n)

        feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]

    stock_data['Volume_1d_change'] = stock_data['Volume'].pct_change()

    volume_features = ['Volume_1d_change']
    feature_names.extend(volume_features)

    stock_data['5d_future_close'] = stock_data['Adj Close'].shift(-5)

    stock_data['5d_close_future_pct'] = stock_data['5d_future_close'].pct_change(
        5)
    stock_data.dropna(inplace=True)

    X = stock_data[feature_names]
    y = stock_data['5d_close_future_pct']

    train_size = int(0.85 * y.shape[0])
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    grid = {'n_estimators': [200], 'max_depth': [3],
            'max_features': [4, 8], 'random_state': [42]}

    test_scores = []

    rf_model = RandomForestRegressor()

    for g in ParameterGrid(grid):
        rf_model.set_params(**g)
        rf_model.fit(X_train, y_train)
        test_scores.append(rf_model.score(X_test, y_test))

    best_index = np.argmax(test_scores)

    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=3, max_features=4, random_state=42)

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    st.subheader('Porcentaje de cambio de precio de cierre previsto de 5 días')
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_pred_series, 'r',
             label='Porcentaje de cambio de precio de cierre previsto de 5 días')

    plt.ylabel("Porcentaje de cambio de precio de cierre previsto de 5 días")
    plt.xlabel('Date')
    plt.legend()
    st.pyplot(fig2)

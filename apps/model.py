import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st


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

   # fig = plt.figure(figsize=(12, 6))
   # plt.plot(stock_data.Close)
   # st.pyplot(fig)

    st.subheader('Cambio porcentual de cierre ajustado de 1 día')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Adj Close'])
    plt.xlabel("Cambio porcentual de cierre ajustado de 1 día")
    st.pyplot(fig)

    #ma100 = stock_data.Close.rolling(100).mean()

    # plt.plot(ma100)
    # plt.plot(stock_data.Close)
    # st.pyplot(fig)

    st.subheader('Closing Price vs Time chart con 100MA & 200MA')
    ma100 = stock_data.Close.rolling(100).mean()
    ma200 = stock_data.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(stock_data.Close, 'b')
    st.pyplot(fig)

    # Splitting data into training and testing

    data_training = pd.DataFrame(
        stock_data['Close'][0:int(len(stock_data)*0.70)])
    data_testing = pd.DataFrame(stock_data['Close'][int(
        len(stock_data)*0.70): int(len(stock_data))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array = scaler.fit_transform(data_training)

    # Cargar mi modelo

    model = load_model('keras_model.h5')

    # Parte de prueba

    past_100_days = data_training.tail(100)
    final_stock_data = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_stock_data)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Grafico Final
    st.subheader('Precio predecido vs Precio Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Precio Original')
    plt.plot(y_predicted, 'r', label='Precio Predecido')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio')
    plt.legend()
    st.pyplot(fig2)

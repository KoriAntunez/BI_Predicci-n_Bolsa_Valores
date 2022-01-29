import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st

def app():
    st.title('Model - SVR')

    dataset = pd.read_csv('GOOG.csv')

    # Describiendo los datos
    #start = st.date_input('Start' , value=pd.to_datetime('2004-08-18'))
    #end = st.date_input('End' , value=pd.to_datetime('today'))

    #st.title('Predicción de tendencia de acciones')

    #user_input = st.text_input('Introducir cotización bursátil' , 'GOOG')

    #dataset = data.DataReader(user_input, 'yahoo', start, end)

    #st.subheader('Datos a travez del tiempo') 
    #st.write(dataset.describe())

     #Visualizaciones 

    X = dataset.iloc[:,4:5].values
    y = dataset.iloc[:,6:7].values
     # Ajustes de escalas
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)

    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X, y)

    # prediccion de un nuevo valor
    x_trans = sc_X.transform([[48.57373]])
    y_pred = regressor.predict(x_trans)
    y_pred = sc_y.inverse_transform(y_pred)

    #Graficando los valores reales
    x_real = sc_X.inverse_transform(X)
    y_real = sc_y.inverse_transform(y)

    X_grid = np.arange(min(x_real), max(x_real), 0.01) 
    X_grid = X_grid.reshape((len(X_grid), 1))

    x_grid_transform = sc_X.transform(X_grid)

    y_grid = regressor.predict(x_grid_transform)
    y_grid_real = sc_y.inverse_transform(y_grid)

    #GRAFICAA

    st.subheader('GRAFICA')
    fig2 = plt.figure(figsize=(12,6))
    plt.scatter(x_real, y_real, color = 'red')
    plt.plot(X_grid, y_grid_real, color = 'blue')
    plt.title('Modelo SVR')
    plt.xlabel('Precio')
    plt.ylabel('Volumen')
    st.pyplot(fig2)

    
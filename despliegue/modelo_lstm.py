import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Función para descargar datos y preparar el DataFrame
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Price_Change'] = data['Adj Close'].diff()
    data['Target'] = np.where(data['Price_Change'] > 0, 1, 0)
    return data

# Función para crear dataset con ventanas deslizantes
def create_dataset(dataset, target, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(target[i + time_step])
    return np.array(dataX), np.array(dataY)

# Función principal de la aplicación
def app():
    st.title("Predicción de la tendencia del precio de acciones con LSTM")

    ticker = st.text_input("Ingrese el símbolo de la acción (por ejemplo, 'BHP')", "BHP")
    start_date = st.date_input("Fecha de inicio", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Fecha de fin", pd.to_datetime("2023-12-31"))

    if st.button("Predecir tendencia"):
        data = load_data(ticker, start_date, end_date)

        st.write("### Descripción de los datos")
        st.write(data.describe())

        # Gráfico del precio (Closing Price vs Time)
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Adj Close'], label='Precio de Cierre Ajustado')
        plt.title('Precio de Cierre Ajustado vs Tiempo')
        plt.xlabel('Fecha')
        plt.ylabel('Precio de Cierre Ajustado')
        plt.legend()
        st.pyplot(plt)

        # Gráfico de Closing Price vs Time con 100MA & 200MA
        data['100MA'] = data['Adj Close'].rolling(window=100).mean()
        data['200MA'] = data['Adj Close'].rolling(window=200).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Adj Close'], label='Precio de Cierre Ajustado')
        plt.plot(data.index, data['100MA'], label='Media Móvil 100 días')
        plt.plot(data.index, data['200MA'], label='Media Móvil 200 días', color='red')
        plt.title('Precio de Cierre Ajustado con Medias Móviles de 100 y 200 días')
        plt.xlabel('Fecha')
        plt.ylabel('Precio de Cierre Ajustado')
        plt.legend()
        st.pyplot(plt)

        # Preparar los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

        time_step = 60
        X, y = create_dataset(scaled_data, data['Target'].values, time_step)

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Remodelar los datos para LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Crear el modelo LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1, activation='sigmoid'))

        # Compilar el modelo
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Entrenar el modelo
        model.fit(X_train, y_train, batch_size=1, epochs=20)

        # Hacer predicciones
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Evaluar el modelo
        st.write("### Matriz de Confusión")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("### Reporte de Clasificación")
        st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

        # Visualización de los resultados
        #results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()}, index=data.index[-len(y_test):])
        #plt.figure(figsize=(10, 6))
        #results['Actual'].plot(label='Actual')
        #results['Predicted'].plot(label='Predicted')
        #plt.title('LSTM Predictions vs Actual Trends')
        #plt.xlabel('Date')
        #plt.ylabel('Trend')
        #plt.legend()
        #st.pyplot(plt)

        # Precio Predicho vs Precio Original (Tendencias)
        plt.figure(figsize=(12, 6))
        plt.plot(results.index, results['Actual'], label='Tendencia Real', marker='o')
        plt.plot(results.index, results['Predicted'], label='Tendencia Predicha', alpha=0.7, linestyle='--', marker='x')
        plt.title('LSTM Predictions vs Actual Trends')
        plt.xlabel('Fecha')
        plt.ylabel('Trend (1: Subida, 0: Bajada)')
        plt.legend()
        st.pyplot(plt)

        # Predicción para el siguiente día
        last_60_days = scaled_data[-time_step:]
        X_next_day = last_60_days.reshape(1, time_step, 1)
        next_day_prediction_prob = model.predict(X_next_day)
        next_day_prediction = (next_day_prediction_prob > 0.5).astype(int)
        trend = "subida" if next_day_prediction[0][0] == 1 else "bajada"
        st.write(f"El precio de {ticker} tiene una tendencia de **{trend}** para el siguiente día.")

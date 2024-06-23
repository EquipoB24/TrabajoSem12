import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Función para descargar datos y preparar el DataFrame
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

# Función para crear secuencias para LSTM
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

# Función principal de la aplicación
def app():
    st.title("Predicción del precio de acciones con LSTM")

    ticker = st.text_input("Ingrese el símbolo de la acción (por ejemplo, 'BHP')", "BHP")
    start_date = st.date_input("Fecha de inicio", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Fecha de fin", pd.to_datetime("2023-12-31"))

    if st.button("Predecir precio"):
        df = load_data(ticker, start_date, end_date)

        st.write("### Descripción de los datos")
        st.write(df.describe())

        # Matriz de correlación
        correlation_matrix = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Matriz de Correlación de Todas las Variables')
        st.pyplot(plt)

        # Gráfico del precio ajustado de cierre
        df['Adj Close'].plot(title='Precio ajustado de cierre de BHP')
        plt.xlabel('Fecha')
        plt.ylabel('Precio ajustado de cierre')
        plt.legend()
        st.pyplot(plt)

        # Visualización de la distribución del precio
        sns.histplot(df['Adj Close'], kde=True)
        plt.title('Distribución del Precio Ajustado de Cierre')
        plt.xlabel('Precio Ajustado de Cierre')
        plt.legend()
        st.pyplot(plt)

        # Gráfico de precio con medias móviles de 30 y 60 días
        df['30MA'] = df['Adj Close'].rolling(window=30).mean()
        df['60MA'] = df['Adj Close'].rolling(window=60).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(df['Adj Close'], label='Precio de Cierre Ajustado')
        plt.plot(df['30MA'], label='Media Móvil 30 días')
        plt.plot(df['60MA'], label='Media Móvil 60 días', color='red')
        plt.title('Precio de Cierre Ajustado con Medias Móviles de 30 y 60 días')
        plt.xlabel('Fecha')
        plt.ylabel('Precio de Cierre Ajustado')
        plt.legend()
        st.pyplot(plt)

        # Preparar datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))

        # Crear secuencias para LSTM
        seq_length = 60  # Usaremos 60 días anteriores para predecir el siguiente día
        X, y = create_sequences(scaled_data, seq_length)

        # Dividir los datos en entrenamiento y prueba
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Construir el modelo LSTM
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        # Compilar el modelo
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Entrenar el modelo
        model.fit(X_train, y_train, batch_size=1, epochs=20)

        # Hacer predicciones
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)  # Desescalar los datos

        # Evaluar el modelo
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        st.write(f'Mean Squared Error: {mse:.2f}')
        st.write(f'Root Mean Squared Error: {rmse:.2f}')
        st.write(f'Mean Absolute Error: {mae:.2f}')

        # Crear un DataFrame con los resultados
        results = pd.DataFrame({'Actual': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), 'Predicted': predictions.flatten()})

        # Comparación del Precio Actual vs Precio Predicho
        plt.figure(figsize=(10, 6))
        plt.plot(results['Actual'], label='Actual')
        plt.plot(results['Predicted'], label='Predicted', linestyle='--')
        plt.legend()
        plt.title('LSTM Predictions vs Actual Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        st.pyplot(plt)

        # Mostrar los primeros resultados
        st.write(results.head())

        # Mostrar predicción para el siguiente día
        last_sequence = scaled_data[-seq_length:]
        last_sequence = last_sequence.reshape((1, seq_length, 1))
        next_day_prediction = model.predict(last_sequence)
        next_day_prediction = scaler.inverse_transform(next_day_prediction)
        st.write(f"El precio de BHP se pronóstica según el modelo LSTM para el siguiente día como: ${next_day_prediction[0][0]:.2f} por acción.")

        # Gráficos de barras para las métricas
        metrics = pd.DataFrame({
            'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error'],
            'Value': [mse, rmse, mae]
        })

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Value', data=metrics)
        plt.title('Métricas de Evaluación del Modelo LSTM')
        st.pyplot(plt)

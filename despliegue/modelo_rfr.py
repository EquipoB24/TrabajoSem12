import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Función para descargar datos y preparar el DataFrame
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

# Función principal de la aplicación
def app():
    st.title("Predicción del precio de acciones con Random Forest")

    ticker = st.text_input("Ingrese el símbolo de la acción (por ejemplo, 'BHP')", "BHP")
    start_date = st.date_input("Fecha de inicio", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Fecha de fin", pd.to_datetime("2023-12-31"))

    if st.button("Predecir precio"):
        data = load_data(ticker, start_date, end_date)

        st.write("### Descripción de los datos")
        st.write(data.describe())

        # Matriz de correlación
        correlation_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Matriz de Correlación de Todas las Variables')
        st.pyplot(plt)

        # Gráfico del precio ajustado de cierre
        plt.figure(figsize=(12, 6))
        data['Adj Close'].plot(title='Precio ajustado de cierre de BHP')
        plt.xlabel('Fecha')
        plt.ylabel('Precio ajustado de cierre')
        st.pyplot(plt)

        # Visualización de la distribución del precio
        plt.figure(figsize=(12, 6))
        sns.histplot(data['Adj Close'], kde=True)
        plt.title('Distribución del Precio Ajustado de Cierre')
        plt.xlabel('Precio Ajustado de Cierre')
        st.pyplot(plt)

        # Gráfico de precio con medias móviles de 30 y 60 días
        data['30MA'] = data['Adj Close'].rolling(window=30).mean()
        data['60MA'] = data['Adj Close'].rolling(window=60).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(data['Adj Close'], label='Precio de Cierre Ajustado')
        plt.plot(data['30MA'], label='Media Móvil 30 días')
        plt.plot(data['60MA'], label='Media Móvil 60 días', color='red')
        plt.title('Precio de Cierre Ajustado con Medias Móviles de 30 y 60 días')
        plt.xlabel('Fecha')
        plt.ylabel('Precio de Cierre Ajustado')
        plt.legend()
        st.pyplot(plt)

        # Preparar los datos
        data['Lag_1'] = data['Adj Close'].shift(1)
        data.dropna(inplace=True)

        X = data[['Lag_1']]
        y = data['Adj Close']

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Escalar los datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Crear y entrenar el modelo Random Forest Regression
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)

        # Hacer predicciones
        y_pred = rf_model.predict(X_test_scaled)

        # Crear un DataFrame con los resultados
        results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred}, index=y_test.index)

        # Comparación del Precio Actual vs Precio Predicho
        plt.figure(figsize=(10, 6))
        plt.plot(results['Actual'], label='Actual')
        plt.plot(results['Predicted'], label='Predicted', linestyle='--')
        plt.legend()
        plt.title('Random Forest Predictions vs Actual Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        st.pyplot(plt)

        # Evaluar el modelo
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f'Mean Squared Error: {mse:.2f}')
        st.write(f'Root Mean Squared Error: {rmse:.2f}')
        st.write(f'Mean Absolute Error: {mae:.2f}')

        # Predicción para el siguiente día
        last_price = scaler.transform([[y.iloc[-1]]])
        next_day_prediction = rf_model.predict(last_price)
        next_day_prediction = scaler.inverse_transform(next_day_prediction.reshape(-1, 1))
        st.write(f"El precio de BHP se pronóstica según el modelo Random Forest para el siguiente día como: ${next_day_prediction[0][0]:.2f} por acción.")

        # Gráficos de barras para las métricas
        metrics = pd.DataFrame({
            'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error'],
            'Value': [mse, rmse, mae]
        })

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Value', data=metrics)
        plt.title('Métricas de Evaluación del Modelo Random Forest')
        st.pyplot(plt)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Función para descargar datos y preparar el DataFrame
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Price_Change'] = data['Adj Close'].diff()
    data['Target'] = np.where(data['Price_Change'] > 0, 1, 0)
    data['Lag_1'] = data['Adj Close'].shift(1)
    data.dropna(inplace=True)
    return data

# Función principal de la aplicación
def app():
    st.title("Predicción de la tendencia del precio de acciones")

    ticker = st.text_input("Ingrese el símbolo de la acción (por ejemplo, 'BHP')", "BHP")
    start_date = st.date_input("Fecha de inicio", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Fecha de fin", pd.to_datetime("2023-12-31"))

    if st.button("Predecir tendencia"):
        data = load_data(ticker, start_date, end_date)

        st.write("### Descripción de los datos")
        st.write(data.describe())

        st.write("### Matriz de Correlación")
        correlation_matrix = data.corr()
        st.write(correlation_matrix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Matriz de Correlación de Todas las Variables')
        st.pyplot(plt)

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

        X = data[['Lag_1']]
        y = data['Target']

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Escalar los datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Crear y entrenar el modelo Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)

        # Hacer predicciones
        y_pred = rf_model.predict(X_test_scaled)

        # Evaluar el modelo
        st.write("### Matriz de Confusión")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("### Reporte de Clasificación")
        st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

        # Visualización de los resultados
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
        plt.figure(figsize=(10, 6))
        results['Actual'].plot(label='Actual')
        results['Predicted'].plot(label='Predicted')
        plt.title('Random Forest Predictions vs Actual Trends')
        plt.xlabel('Date')
        plt.ylabel('Trend')
        plt.legend()
        st.pyplot(plt)

        # Precio Predicho vs Precio Original (Tendencias)
        plt.figure(figsize=(12, 6))
        plt.plot(results.index, results['Actual'], label='Tendencia Real', marker='o')
        plt.plot(results.index, results['Predicted'], label='Tendencia Predicha', alpha=0.7, linestyle='--', marker='x')
        plt.title('Random Forest Predictions vs Actual Trends')
        plt.xlabel('Fecha')
        plt.ylabel('Tendencia (1: Subida, 0: Bajada)')
        plt.legend()
        st.pyplot(plt)

        # Predicción para el siguiente día
        last_price = scaler.transform([[X.iloc[-1]['Lag_1']]])
        next_day_prediction = rf_model.predict(last_price)
        trend = "subida" if next_day_prediction[0] == 1 else "bajada"
        st.write(f"El precio de {ticker} tiene una tendencia de **{trend}** para el siguiente día.")

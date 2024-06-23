import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Función para descargar datos y preparar el DataFrame
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['Price_Change'] = df['Adj Close'].diff()
    df['Target'] = (df['Price_Change'] > 0).astype(int)
    df['Lag_1'] = df['Adj Close'].shift(1)
    df.dropna(inplace=True)
    return df

# Función principal de la aplicación
def app():
    st.title("Predicción del precio de acciones con SVR")

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
        X = df[['Lag_1']]
        y = df['Adj Close']

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Escalar los datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Crear y entrenar el modelo SVR
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        model.fit(X_train_scaled, y_train)

        # Predecir
        y_pred = model.predict(X_test_scaled)

        

        # Crear un DataFrame con los resultados
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

        # Comparación del Precio Actual vs Precio Predicho
        plt.figure(figsize=(10, 6))
        plt.plot(results['Actual'].values, label='Actual')
        plt.plot(results['Predicted'].values, label='Predicted', linestyle='--')
        plt.legend()
        plt.title('SVR Predictions vs Actual Prices')
        plt.xlabel('Index')
        plt.ylabel('Price')
        st.pyplot(plt)

        # Visualizar resultados
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.index, y_test, label='Real')
        plt.plot(y_test.index, y_pred, label='Predicción', linestyle='--')
        plt.title('Predicción del precio de BHP usando SVR')
        plt.xlabel('Fecha')
        plt.ylabel('Precio USD')
        plt.legend()
        st.pyplot(plt)

        # Mostrar los primeros resultados
        st.write(results.head())

        # Gráficos de barras para las métricas
        metrics = pd.DataFrame({
            'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error'],
            'Value': [mse, rmse, mae]
        })

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Value', data=metrics)
        plt.title('Métricas de Evaluación del Modelo SVR')
        st.pyplot(plt)

        # Evaluar el modelo
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'Root Mean Squared Error: {rmse}')
        st.write(f'Mean Absolute Error: {mae}')

        # Mostrar predicción para el siguiente día
        next_day_prediction = model.predict(scaler.transform([[y.iloc[-1]]]))
        st.write(f"El precio de BHP se pronóstica según el modelo SVR para el siguiente día como: ${next_day_prediction[0]:.2f} por acción.")

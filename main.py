import streamlit as st

st.set_page_config(
    page_title="Tarea Semana 12",
    page_icon="👋",
)

st.write("# Despliegue web de modelos del Grupo G 🤖")

st.sidebar.success("Seleccione un modelo del menú")

st.markdown(
    """
    # Grupo B - Integrantes:
    | Nombre | Participación|
    |--|--|
    ### Especificaciones:
    **Donde muestra las predicciones/los resultados:**
    - Gráficamente. 
    - Númericamente los valores de las predicciones (print de dataframe con la predicción o clasificación).
    
    **Donde se muestra el EDA:**
    - Ploteo de los precios reales.
    (Ploteo de media móvil los precios reales.)

    **Donde el usuario pueda indicar:**
    - El modelo ejecutar.
    - La acción o instrumento financiero que quiera analizar.
"""
)

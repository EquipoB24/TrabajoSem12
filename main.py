import streamlit as st
from multiapp import MultiApp
#from despliegue import home, modelo_random_forest_regression, modelo_svc, modelo_lstm, modelo_kmeans, modelo_svr, modelo_clustering_jerarquico_lq
from despliegue import home, modelo_arbol_decision, modelo_random_forest_classifier, modelo_lstm2, modelo_svc
from despliegue import modelo_svr, modelo_lstmr, modelo_rfr, modelo_knnr
#from despliegue import scrapping_twitter
# from despliegue import modelo_lstm, modelo_arima, modelo_decision_tree, modelo_prophet,  modelo_svr


app = MultiApp()
st.markdown("# Inteligencia de Negocios - Equipo B - Semana 13 ")


# Add all your application here
app.add_app("Home", home.app)
# app.add_app("Modelo Arima", modelo_arima.app)
app.add_app("Modelo Árbol de decisión", modelo_arbol_decision.app)
app.add_app("Modelo de Random Forest Classifier", modelo_random_forest_classifier.app)
app.add_app("Modelo LSTM", modelo_lstm2.app)
app.add_app("Modelo SVC", modelo_svc.app)
app.add_app("Modelo SVR", modelo_svr.app)
app.add_app("Modelo LSTM R", modelo_lstmr.app)
app.add_app("Modelo RFR", modelo_rfr.app)
app.add_app("Modelo KNN R", modelo_knnr.app)

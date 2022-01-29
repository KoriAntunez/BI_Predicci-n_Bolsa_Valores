import streamlit as st
from multiapp import MultiApp
from apps import home, model # import your app modules here

app = MultiApp()

st.markdown("""
#  Inteligencia de Negocios - Grupo 6

""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Modelo", model.app)
# The main app
app.run()




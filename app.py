import streamlit as st

st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.image(
    "assets/logo.png",
    width=150
)
# Load custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Customer Churn Intelligence Platform")

st.markdown("""
Welcome to the **Customer Churn Intelligence Platform**.

This application provides:

- 📈 Executive Dashboard
- 📊 Data Explorer
- 🤖 ML Model Performance
- 🎯 Single Customer Prediction
- 📂 Batch Prediction
- 🔍 Explainable AI (SHAP)
- 👥 Customer Segmentation
- 📡 Model Monitoring
""")

st.info("Use the left sidebar to navigate through the application.")

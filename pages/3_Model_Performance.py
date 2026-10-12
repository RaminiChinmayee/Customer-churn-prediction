import streamlit as st
import json
from PIL import Image

st.title("🤖 Model Performance")

with open("models/model_metrics.json") as f:
    metrics = json.load(f)

st.subheader("Model Comparison")

st.dataframe(metrics)

st.subheader("Confusion Matrix")

st.image("reports/confusion_matrix.png")

st.subheader("ROC Curve")

st.image("reports/roc_curve.png")
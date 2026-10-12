import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.explain import explain

st.title("🔍 Explainable AI")

uploaded = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)

    values = explain(df)

    fig = plt.figure()

    shap.summary_plot(
        values,
        df,
        show=False
    )

    st.pyplot(fig)
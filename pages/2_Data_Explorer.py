import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📂 Data Explorer")

df = pd.read_csv("data/telecom.csv")

st.subheader("Dataset")

st.dataframe(df)

st.subheader("Statistics")

st.dataframe(df.describe())

st.subheader("Missing Values")

st.dataframe(df.isnull().sum())

column = st.selectbox(
    "Choose Feature",
    df.columns
)

fig = px.histogram(
    df,
    x=column,
    color="Churn"
)

st.plotly_chart(fig, use_container_width=True)
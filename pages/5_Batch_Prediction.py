import streamlit as st
import pandas as pd

from src.predict import predict_customer

st.title("📂 Batch Prediction")

uploaded = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

if uploaded:

    df = pd.read_csv(uploaded)

    predictions = []

    probs = []

    risks = []

    actions = []

    progress = st.progress(0)

    for i in range(len(df)):

        result = predict_customer(df.iloc[[i]])

        predictions.append(result["prediction"])

        probs.append(result["probability"])

        risks.append(result["risk"])

        actions.append(result["recommendation"])

        progress.progress((i + 1) / len(df))

    df["Prediction"] = predictions

    df["Probability"] = probs

    df["Risk"] = risks

    df["Recommendation"] = actions

    st.success("Batch Prediction Finished")

    st.dataframe(df)

    csv = df.to_csv(index=False)

    st.download_button(
        "Download Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )
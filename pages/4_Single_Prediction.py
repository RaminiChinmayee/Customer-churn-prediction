import streamlit as st
import pandas as pd
import time

from src.predict import predict_customer

st.set_page_config(layout="wide")

st.title("🎯 Customer Churn Prediction")

st.write("Enter customer information to predict the likelihood of churn.")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox("Gender", ["Female", "Male"])

    senior = st.selectbox("Senior Citizen", [0, 1])

    partner = st.selectbox("Partner", ["Yes", "No"])

    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    phone = st.selectbox("Phone Service", ["Yes", "No"])

    multiple = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:

    security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

monthly = st.number_input(
    "Monthly Charges",
    value=70.0
)

total = st.number_input(
    "Total Charges",
    value=900.0
)

if st.button("Predict Customer Churn"):

    customer = pd.DataFrame({

        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phone],
        "MultipleLines":[multiple],
        "InternetService":[internet],
        "OnlineSecurity":[security],
        "OnlineBackup":[backup],
        "DeviceProtection":[device],
        "TechSupport":[support],
        "StreamingTV":[tv],
        "StreamingMovies":[movies],
        "Contract":[contract],
        "PaperlessBilling":[billing],
        "PaymentMethod":[payment],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total]

    })

    start = time.time()

    result = predict_customer(customer)

    latency = time.time() - start

    st.success("Prediction Complete")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Prediction",
        "Churn" if result["prediction"] else "No Churn"
    )

    c2.metric(
        "Probability",
        f"{result['probability']:.2%}"
    )

    c3.metric(
        "Risk Level",
        result["risk"]
    )

    st.info(result["recommendation"])

    st.caption(f"Inference Time: {latency:.4f} seconds")
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import pandas as pd

from config import *

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

features = joblib.load(FEATURE_COLUMNS)

encoders = joblib.load(ENCODER_PATH)


def preprocess_input(df):

    for col, encoder in encoders.items():

        if col in df.columns:

            df[col] = encoder.transform(df[col])

    df = df[features]

    df = scaler.transform(df)

    return df


def predict_customer(df):

    x = preprocess_input(df)

    probability = model.predict_proba(x)[0][1]

    prediction = int(probability >= 0.5)

    if probability < 0.30:
        risk = "Low"

    elif probability < 0.60:
        risk = "Medium"

    elif probability < 0.80:
        risk = "High"

    else:
        risk = "Critical"

    recommendations = {
        "Low": "No action required.",
        "Medium": "Email retention campaign.",
        "High": "Customer success call.",
        "Critical": "Offer discount and loyalty benefits."
    }

    return {
        "prediction": prediction,
        "probability": probability,
        "risk": risk,
        "recommendation": recommendations[risk]
    }
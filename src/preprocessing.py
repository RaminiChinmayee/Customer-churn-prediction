import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

import joblib

from config import *

def preprocess():

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.fillna(df.median(numeric_only=True), inplace=True)

    df.drop("customerID", axis=1, inplace=True)

    encoders = {}

    for col in df.select_dtypes(include="object").columns:

        if col != "Churn":

            le = LabelEncoder()

            df[col] = le.fit_transform(df[col])

            encoders[col] = le

    target = LabelEncoder()

    df["Churn"] = target.fit_transform(df["Churn"])

    X = df.drop("Churn", axis=1)

    y = df["Churn"]

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)

    joblib.dump(encoders, ENCODER_PATH)

    joblib.dump(list(df.drop("Churn", axis=1).columns),
                FEATURE_COLUMNS)

    return X, y
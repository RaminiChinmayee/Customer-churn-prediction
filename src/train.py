import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import joblib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from preprocessing import preprocess
from config import *

# -----------------------------
# Load Data
# -----------------------------
X, y = preprocess()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# -----------------------------
# Models for Comparison
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ),
    "LightGBM": LGBMClassifier(
        random_state=42
    ),
    "CatBoost": CatBoostClassifier(
        verbose=False,
        random_state=42
    )
}

results = {}

print("=" * 60)
print("Training Models")
print("=" * 60)

# -----------------------------
# Compare All Models
# -----------------------------
for name, model in models.items():

    print(f"\nTraining {name}")

    start = time.time()

    model.fit(X_train, y_train)

    end = time.time()

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC_AUC": round(roc_auc_score(y_test, prob), 4),
        "Training Time (sec)": round(end - start, 2)
    }

# -----------------------------
# Save Comparison Metrics
# -----------------------------
with open("models/model_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

# -----------------------------
# Train Final Production Model
# -----------------------------
print("\n" + "=" * 60)
print("Training Final Production Model (XGBoost)")
print("=" * 60)

final_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

final_model.fit(X_train, y_train)

joblib.dump(final_model, MODEL_PATH)
joblib.dump(final_model, "models/xgb_model.pkl")

print("\nProduction Model Saved Successfully!")
print("Saved as:")
print("   models/best_model.pkl")
print("   models/xgb_model.pkl")
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import shap

from config import MODEL_PATH

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load(MODEL_PATH)

# -----------------------------
# Create SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)

# -----------------------------
# Return SHAP values
# -----------------------------
def explain(data):
    """
    Returns SHAP values for the given preprocessed data.
    """
    shap_values = explainer.shap_values(data)
    return shap_values


# -----------------------------
# Return explainer (optional)
# -----------------------------
def get_explainer():
    return explainer


# -----------------------------
# Return feature importance
# -----------------------------
def feature_importance():
    return model.feature_importances_
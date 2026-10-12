# Customer-churn-prediction
Overview

Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to leave a service. It compares multiple classification algorithms, evaluates their performance using standard metrics, and uses SHAP to help explain the predictions made by the trained model.

Features

Data preprocessing and preparation for machine learning

Training and comparison of six classification algorithms

Evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC

Comparison of model training time

Saving evaluation results in JSON format

Saving the trained XGBoost model using Joblib

Model interpretability using SHAP feature explanations

Machine Learning Models

The project compares the following algorithms:

Logistic Regression

Decision Tree

Random Forest

XGBoost

LightGBM

CatBoost

These models provide a comparison between a linear classifier, individual decision trees, ensemble methods, and gradient-boosting algorithms.

Evaluation Metrics

Metric

Purpose

Accuracy

Measures the proportion of correct predictions

Precision

Measures how many predicted churn cases are actual churn cases

Recall

Measures how many actual churn cases are correctly identified

F1-score

Balances precision and recall

ROC-AUC

Measures how well the model distinguishes between the two classes

Training Time

Measures the time taken to train each model

Model Explainability

The project uses SHAP (SHapley Additive exPlanations) to understand feature contributions to model predictions.

shap.TreeExplainer explains predictions from the trained tree-based model.

SHAP values help identify which features influence an individual prediction.

Model feature importance provides an overall view of influential features.

Technologies Used

Language: Python

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost, LightGBM, CatBoost

Model Explainability: SHAP

Model Persistence: Joblib

Results Storage: JSON

Version Control: Git and GitHub

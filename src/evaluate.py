
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from preprocessing import preprocess

from sklearn.model_selection import train_test_split

from config import *

X,y=preprocess()

X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

model=joblib.load(MODEL_PATH)

fig,ax=plt.subplots(figsize=(5,5))

ConfusionMatrixDisplay.from_estimator(
model,
X_test,
y_test,
ax=ax
)

plt.savefig("reports/confusion_matrix.png")

plt.close()

fig,ax=plt.subplots(figsize=(5,5))

RocCurveDisplay.from_estimator(
model,
X_test,
y_test,
ax=ax
)

plt.savefig("reports/roc_curve.png")

plt.close()

print("Evaluation Finished")
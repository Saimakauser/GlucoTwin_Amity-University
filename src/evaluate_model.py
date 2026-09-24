import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "training_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "glucotwin_xgboost.pkl"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD DATA + MODEL
# ============================================================

print("=" * 60)
print("GLUCOTWIN MODEL EVALUATION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df.shape}"
)

print("\nLoading trained model...")

model = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# PREPARE FEATURES
# ============================================================

TARGET = "glucose_spike_2h"

X = df.drop(
    columns=[
        "patient_id",
        "timestamp",
        "future_glucose_2h",
        TARGET
    ]
)

y = df[TARGET]

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.fillna(0)


# ============================================================
# SAME TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist(),
    "test_samples": int(len(y_test))
}

metrics_path = os.path.join(
    MODELS_DIR,
    "evaluation_metrics.json"
)

with open(
    metrics_path,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(6, 5)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Spike",
        "Spike"
    ]
)

disp.plot(
    ax=plt.gca()
)

plt.title(
    "GlucoTwin - Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    MODELS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "GlucoTwin - ROC Curve"
)

plt.legend()

plt.tight_layout()

roc_path = os.path.join(
    MODELS_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

top_features = feature_importance.head(
    15
).sort_values(
    "importance",
    ascending=True
)

plt.figure(
    figsize=(9, 6)
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "GlucoTwin - Top Feature Importance"
)

plt.tight_layout()

feature_path = os.path.join(
    MODELS_DIR,
    "feature_importance.png"
)

plt.savefig(
    feature_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)

print(
    "EVALUATION COMPLETE"
)

print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(
    cm
)

print(
    "\nArtifacts generated:"
)

print(
    f"✓ {metrics_path}"
)

print(
    f"✓ {cm_path}"
)

print(
    f"✓ {roc_path}"
)

print(
    f"✓ {feature_path}"
)

print("=" * 60)
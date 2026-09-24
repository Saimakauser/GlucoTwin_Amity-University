import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


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

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("GLUCOTWIN MODEL TRAINING")
print("=" * 60)

print("\nLoading processed dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# TARGET
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


# ============================================================
# ENSURE NUMERIC FEATURES
# ============================================================

print(
    "\nPreparing model features..."
)

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.fillna(0)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)

print(
    f"Number of features: {X.shape[1]}"
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nTraining XGBoost..."
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# EVALUATION
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
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODELS_DIR,
    "glucotwin_xgboost.pkl"
)

joblib.dump(
    model,
    model_path
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_list_path = os.path.join(
    MODELS_DIR,
    "model_features.json"
)

with open(
    feature_list_path,
    "w"
) as file:

    json.dump(
        list(X.columns),
        file,
        indent=4
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
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "feature_count": int(X.shape[1])
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
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance_path = os.path.join(
    MODELS_DIR,
    "feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 60)

print(
    "MODEL TRAINING COMPLETE"
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
    "\nTop 15 Features:"
)

print(
    importance.head(15).to_string(
        index=False
    )
)

print(
    "\nModel saved:"
)

print(
    model_path
)

print(
    "\nFeature list saved:"
)

print(
    feature_list_path
)

print(
    "\nMetrics saved:"
)

print(
    metrics_path
)

print(
    "\nFeature importance saved:"
)

print(
    importance_path
)

print("=" * 60)
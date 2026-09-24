import os
import json
import joblib
import pandas as pd
import shap

from digital_twin import load_data, get_digital_twin
from prediction_engine import create_feature_vector


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "glucotwin_xgboost.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_features.json"
)


def load_model_and_features():
    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    return model, features


def clean_feature_name(feature):
    """
    Convert technical feature names into
    dashboard-friendly labels.
    """

    replacements = {
        "past_diagnosis_": "Past diagnosis: ",
        "medication_": "Medication: ",
        "genetic_risk_": "Genetic risk: ",
        "sleep_stage_": "Sleep stage: ",
        "activity_": "Activity: ",
        "_": " "
    }

    name = feature

    for old, new in replacements.items():
        name = name.replace(old, new)

    return name.strip().title()


def explain_patient(patient_id=42):

    print("=" * 60)
    print("GlucoTwin SHAP Explainability")
    print("=" * 60)

    # Load model
    model, feature_names = load_model_and_features()

    # Load patient data
    ehr, training_data = load_data()

    twin = get_digital_twin(
        patient_id,
        ehr,
        training_data
    )

    # Create exact model input
    X = create_feature_vector(
        twin,
        feature_names
    )

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        values = shap_values[1][0]

    else:
        values = shap_values[0]

        # Some SHAP versions return shape (1, features, classes)
        if hasattr(values, "ndim") and values.ndim == 2:
            values = values[:, 1]

    # Create explanation table
    explanation = pd.DataFrame({
        "feature": feature_names,
        "value": X.iloc[0].values,
        "shap_value": values
    })

    # Absolute importance
    explanation["abs_shap"] = explanation["shap_value"].abs()

    explanation = explanation.sort_values(
        "abs_shap",
        ascending=False
    )

    # Save complete explanation
    output_path = os.path.join(
        BASE_DIR,
        "models",
        f"shap_patient_{patient_id}.csv"
    )

    explanation.to_csv(
        output_path,
        index=False
    )

    # Top 5 contributors
    top_features = explanation.head(5)

    print("\nPatient ID:", patient_id)

    print(
        f"Current Glucose: "
        f"{twin['current_glucose']:.1f} mg/dL"
    )

    print("\nTop prediction contributors:")

    for _, row in top_features.iterrows():

        direction = (
            "increases prediction"
            if row["shap_value"] > 0
            else "decreases prediction"
        )

        print(
            f"- {clean_feature_name(row['feature'])}: "
            f"{row['shap_value']:.4f} "
            f"({direction})"
        )

    print("\nFull SHAP explanation saved to:")

    print(output_path)

    print("\nSHAP explainability test completed.")


if __name__ == "__main__":

    explain_patient(42)
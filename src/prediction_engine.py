import os
import json
import joblib
import pandas as pd


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


def load_model():
    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    return model, features


def create_feature_vector(twin, feature_names):
    """
    Convert the Digital Twin state into the exact
    45-feature structure expected by the trained model.
    """

    # Start every expected feature at zero.
    row = {
        feature: 0
        for feature in feature_names
    }

    # -----------------------------
    # Numeric features
    # -----------------------------

    numeric_features = [
        "age",
        "bmi",
        "hba1c",
        "fasting_glucose",
        "systolic_bp",
        "diastolic_bp",
        "cholesterol",
        "diabetes_years",
        "glucose",
        "heart_rate",
        "hrv",
        "steps",
        "sleep_stage_level",
        "activity_level",
        "hour",
        "day_of_week",
        "glucose_change_15m",
        "glucose_change_30m",
        "glucose_change_60m",
        "glucose_mean_1h",
        "glucose_mean_2h",
        "glucose_std_1h",
        "steps_1h",
        "steps_2h",
        "heart_rate_mean_1h",
        "hrv_mean_1h"
    ]

    for feature in numeric_features:

        if feature in row and feature in twin:

            value = twin[feature]

            if pd.notna(value):
                row[feature] = float(value)

    # -----------------------------
    # Past diagnosis
    # -----------------------------

    diagnosis = str(
        twin["past_diagnosis"]
    )

    diagnosis_column = (
        f"past_diagnosis_{diagnosis}"
    )

    if diagnosis_column in row:
        row[diagnosis_column] = 1

    # -----------------------------
    # Medication
    # -----------------------------

    medication = str(
        twin["medication"]
    )

    medication_column = (
        f"medication_{medication}"
    )

    if medication_column in row:
        row[medication_column] = 1

    # -----------------------------
    # Genetic risk
    # -----------------------------

    genetic_risk = str(
        twin["genetic_risk"]
    )

    genetic_column = (
        f"genetic_risk_{genetic_risk}"
    )

    if genetic_column in row:
        row[genetic_column] = 1

    # -----------------------------
    # Sleep stage
    # -----------------------------

    sleep_stage = str(
        twin["sleep_stage"]
    )

    sleep_column = (
        f"sleep_stage_{sleep_stage}"
    )

    if sleep_column in row:
        row[sleep_column] = 1

    # -----------------------------
    # Activity
    # -----------------------------

    activity = str(
        twin["activity"]
    )

    activity_column = (
        f"activity_{activity}"
    )

    if activity_column in row:
        row[activity_column] = 1

    # Ensure exact feature ordering.
    X = pd.DataFrame(
        [[row[feature] for feature in feature_names]],
        columns=feature_names
    )

    return X


def predict_patient(twin):

    model, feature_names = load_model()

    X = create_feature_vector(
        twin,
        feature_names
    )

    probability = model.predict_proba(X)[0][1]

    prediction = int(
        probability >= 0.5
    )

    risk_percentage = probability * 100

    if risk_percentage >= 70:
        category = "HIGH"

    elif risk_percentage >= 40:
        category = "MODERATE"

    else:
        category = "LOW"

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_percentage": risk_percentage,
        "category": category,
        "digital_twin": twin
    }


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    from digital_twin import (
        load_data,
        get_digital_twin
    )

    print("=" * 60)
    print("GLUCOTWIN PREDICTION ENGINE")
    print("=" * 60)

    ehr, training_data = load_data()

    patient_id = 42

    twin = get_digital_twin(
        patient_id,
        ehr,
        training_data
    )

    result = predict_patient(twin)

    print(
        f"\nPatient ID: {patient_id}"
    )

    print(
        f"Current Glucose: "
        f"{twin['current_glucose']:.1f} mg/dL"
    )

    print(
        f"Predicted 2-Hour Spike Probability: "
        f"{result['risk_percentage']:.2f}%"
    )

    print(
        f"Prototype Risk Category: "
        f"{result['category']}"
    )

    print(
        f"Binary Prediction: "
        f"{result['prediction']}"
    )

    print(
        "\nFeature vector successfully generated."
    )

    print(
        "Prediction engine test completed."
    )
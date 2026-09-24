import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

np.random.seed(42)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

os.makedirs(
    RAW_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

NUM_PATIENTS = 100
DAYS = 7
FREQUENCY = "15min"


# ============================================================
# STATIC / HISTORICAL EHR DATA
# ============================================================

patients = []

medications = [
    "Metformin",
    "Metformin + Insulin",
    "Metformin + GLP-1",
    "Insulin",
    "Lifestyle Management"
]

genetic_risks = [
    "Low",
    "Moderate",
    "High"
]

diagnoses = [
    "Type 2 Diabetes",
    "Type 2 Diabetes + Hypertension",
    "Type 2 Diabetes + Dyslipidemia",
    "Type 2 Diabetes + Hypertension + Dyslipidemia"
]

for patient_id in range(
    1,
    NUM_PATIENTS + 1
):

    age = np.random.randint(
        35,
        76
    )

    sex = np.random.choice(
        ["Male", "Female"]
    )

    bmi = round(
        np.random.normal(
            27.5,
            3.8
        ),
        1
    )

    bmi = max(
        19,
        min(
            bmi,
            40
        )
    )

    hba1c = round(
        np.random.normal(
            7.8,
            1.1
        ),
        1
    )

    hba1c = max(
        6.0,
        min(
            hba1c,
            11.5
        )
    )

    fasting_glucose = round(
        np.random.normal(
            145,
            25
        ),
        1
    )

    fasting_glucose = max(
        90,
        min(
            fasting_glucose,
            230
        )
    )

    systolic_bp = int(
        np.random.normal(
            135,
            15
        )
    )

    diastolic_bp = int(
        np.random.normal(
            85,
            10
        )
    )

    cholesterol = int(
        np.random.normal(
            205,
            35
        )
    )

    cholesterol = max(
        120,
        min(
            cholesterol,
            320
        )
    )

    diabetes_years = np.random.randint(
        1,
        21
    )

    medication = np.random.choice(
        medications
    )

    genetic_risk = np.random.choice(
        genetic_risks,
        p=[
            0.25,
            0.50,
            0.25
        ]
    )

    diagnosis = np.random.choice(
        diagnoses,
        p=[
            0.45,
            0.25,
            0.20,
            0.10
        ]
    )

    patients.append(
        {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "hba1c": hba1c,
            "fasting_glucose": fasting_glucose,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "cholesterol": cholesterol,
            "diabetes_years": diabetes_years,
            "past_diagnosis": diagnosis,
            "medication": medication,
            "genetic_risk": genetic_risk
        }
    )


ehr_df = pd.DataFrame(
    patients
)


# ============================================================
# DYNAMIC / WEARABLE DATA
# ============================================================

wearable_records = []

start_date = pd.Timestamp(
    "2026-01-01 00:00:00"
)

timestamps = pd.date_range(
    start=start_date,
    periods=DAYS * 96,
    freq=FREQUENCY
)


for _, patient in ehr_df.iterrows():

    baseline_glucose = (
        patient["fasting_glucose"]
    )

    patient_hba1c = (
        patient["hba1c"]
    )

    for timestamp in timestamps:

        hour = timestamp.hour

        # ----------------------------------------------------
        # Circadian / meal-related glucose pattern
        # ----------------------------------------------------

        meal_effect = 0

        if 7 <= hour <= 9:

            meal_effect += np.random.normal(
                25,
                10
            )

        if 12 <= hour <= 14:

            meal_effect += np.random.normal(
                35,
                12
            )

        if 19 <= hour <= 21:

            meal_effect += np.random.normal(
                40,
                14
            )

        # ----------------------------------------------------
        # Activity
        # ----------------------------------------------------

        activity_probability = np.random.rand()

        if activity_probability < 0.45:

            activity = "Low"

            steps = np.random.randint(
                0,
                80
            )

        elif activity_probability < 0.82:

            activity = "Moderate"

            steps = np.random.randint(
                80,
                250
            )

        else:

            activity = "High"

            steps = np.random.randint(
                250,
                700
            )

        # ----------------------------------------------------
        # Sleep stage
        # ----------------------------------------------------

        if 23 <= hour or hour < 6:

            sleep_probability = np.random.rand()

            if sleep_probability < 0.15:

                sleep_stage = "Awake"

            elif sleep_probability < 0.55:

                sleep_stage = "Light"

            elif sleep_probability < 0.85:

                sleep_stage = "Deep"

            else:

                sleep_stage = "REM"

        else:

            sleep_stage = "Awake"

        # ----------------------------------------------------
        # Sleep effect
        # ----------------------------------------------------

        if sleep_stage == "Deep":

            sleep_effect = -8

        elif sleep_stage == "Light":

            sleep_effect = -4

        elif sleep_stage == "REM":

            sleep_effect = -2

        else:

            sleep_effect = 0

        # ----------------------------------------------------
        # Glucose
        # ----------------------------------------------------

        glucose_noise = np.random.normal(
            0,
            8
        )

        glucose = (
            baseline_glucose
            + meal_effect
            + sleep_effect
            + glucose_noise
            + ((patient_hba1c - 7.0) * 8)
            - (steps * 0.025)
        )

        glucose = max(
            65,
            min(
                glucose,
                350
            )
        )

        glucose = round(
            glucose,
            1
        )

        # ----------------------------------------------------
        # Heart rate
        # ----------------------------------------------------

        heart_rate = (
            70
            + np.random.normal(
                0,
                7
            )
        )

        if activity == "Moderate":

            heart_rate += 10

        elif activity == "High":

            heart_rate += 20

        if sleep_stage != "Awake":

            heart_rate -= 8

        heart_rate = max(
            45,
            min(
                heart_rate,
                150
            )
        )

        heart_rate = round(
            heart_rate,
            1
        )

        # ----------------------------------------------------
        # HRV
        # ----------------------------------------------------

        hrv = (
            45
            + np.random.normal(
                0,
                8
            )
        )

        if sleep_stage == "Deep":

            hrv += 10

        elif sleep_stage == "REM":

            hrv += 5

        if activity == "High":

            hrv -= 5

        hrv = max(
            15,
            min(
                hrv,
                100
            )
        )

        hrv = round(
            hrv,
            1
        )

        # ----------------------------------------------------
        # Save wearable record
        # ----------------------------------------------------

        wearable_records.append(
            {
                "patient_id": patient["patient_id"],
                "timestamp": timestamp,
                "glucose": glucose,
                "heart_rate": heart_rate,
                "hrv": hrv,
                "steps": steps,
                "sleep_stage": sleep_stage,
                "activity": activity
            }
        )


wearable_df = pd.DataFrame(
    wearable_records
)


# ============================================================
# SAVE DATA
# ============================================================

ehr_path = os.path.join(
    RAW_DIR,
    "synthetic_ehr.csv"
)

wearable_path = os.path.join(
    RAW_DIR,
    "synthetic_wearable.csv"
)

ehr_df.to_csv(
    ehr_path,
    index=False
)

wearable_df.to_csv(
    wearable_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)

print(
    "GLUCOTWIN SYNTHETIC DATA GENERATION"
)

print("=" * 60)

print(
    f"EHR patients: {len(ehr_df)}"
)

print(
    f"Wearable records: {len(wearable_df)}"
)

print(
    "\nEHR columns:"
)

print(
    list(ehr_df.columns)
)

print(
    "\nWearable columns:"
)

print(
    list(wearable_df.columns)
)

print(
    "\nPast diagnosis distribution:"
)

print(
    ehr_df["past_diagnosis"].value_counts()
)

print(
    "\nSleep stage distribution:"
)

print(
    wearable_df["sleep_stage"].value_counts()
)

print(
    "\nFiles saved:"
)

print(
    ehr_path
)

print(
    wearable_path
)

print("=" * 60)
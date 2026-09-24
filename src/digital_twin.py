import os
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

EHR_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "synthetic_ehr.csv"
)

TRAINING_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "training_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading EHR data...")

    ehr = pd.read_csv(
        EHR_PATH
    )

    print(
        f"EHR records loaded: {len(ehr)}"
    )

    print(
        "Loading processed wearable data..."
    )

    training_data = pd.read_csv(
        TRAINING_PATH
    )

    print(
        f"Wearable records loaded: "
        f"{len(training_data)}"
    )

    return ehr, training_data


# ============================================================
# GET DIGITAL TWIN STATE
# ============================================================

def get_digital_twin(
    patient_id,
    ehr,
    training_data
):

    # --------------------------------------------------------
    # Patient EHR
    # --------------------------------------------------------

    patient_ehr = ehr[
        ehr["patient_id"] == patient_id
    ]

    if patient_ehr.empty:

        raise ValueError(
            f"Patient {patient_id} not found in EHR data."
        )

    patient_ehr = patient_ehr.iloc[0]


    # --------------------------------------------------------
    # Patient time-series
    # --------------------------------------------------------

    patient_data = training_data[
        training_data["patient_id"] == patient_id
    ].copy()

    if patient_data.empty:

        raise ValueError(
            f"No wearable data found for patient {patient_id}."
        )

    patient_data = patient_data.sort_values(
        "timestamp"
    )


    # --------------------------------------------------------
    # Latest physiological state
    # --------------------------------------------------------

    latest = patient_data.iloc[-1]


    # --------------------------------------------------------
    # Sleep stage
    # --------------------------------------------------------

    sleep_stage = "Unknown"

    for stage in [
        "Awake",
        "Light",
        "Deep",
        "REM"
    ]:

        column = (
            f"sleep_stage_{stage}"
        )

        if column in latest.index:

            try:

                if float(
                    latest[column]
                ) == 1:

                    sleep_stage = stage
                    break

            except:

                pass


    # --------------------------------------------------------
    # Activity
    # --------------------------------------------------------

    activity = "Unknown"

    for level in [
        "Low",
        "Moderate",
        "High"
    ]:

        column = (
            f"activity_{level}"
        )

        if column in latest.index:

            try:

                if float(
                    latest[column]
                ) == 1:

                    activity = level
                    break

            except:

                pass


    # --------------------------------------------------------
    # Build Digital Twin
    # --------------------------------------------------------

    twin = {

        # --------------------------------------------
        # Patient identity
        # --------------------------------------------

        "patient_id":
            int(patient_id),

        # --------------------------------------------
        # Historical EHR
        # --------------------------------------------

        "age":
            float(patient_ehr["age"]),

        "sex":
            patient_ehr["sex"],

        "bmi":
            float(patient_ehr["bmi"]),

        "hba1c":
            float(patient_ehr["hba1c"]),

        "fasting_glucose":
            float(
                patient_ehr[
                    "fasting_glucose"
                ]
            ),

        "systolic_bp":
            float(
                patient_ehr[
                    "systolic_bp"
                ]
            ),

        "diastolic_bp":
            float(
                patient_ehr[
                    "diastolic_bp"
                ]
            ),

        "cholesterol":
            float(
                patient_ehr[
                    "cholesterol"
                ]
            ),

        "diabetes_years":
            float(
                patient_ehr[
                    "diabetes_years"
                ]
            ),

        "past_diagnosis":
            patient_ehr[
                "past_diagnosis"
            ],

        "medication":
            patient_ehr[
                "medication"
            ],

        "genetic_risk":
            patient_ehr[
                "genetic_risk"
            ],

        # --------------------------------------------
        # Dynamic wearable state
        # --------------------------------------------

        "current_glucose":
            float(
                latest["glucose"]
            ),

        "heart_rate":
            float(
                latest["heart_rate"]
            ),

        "hrv":
            float(
                latest["hrv"]
            ),

        "steps":
            float(
                latest["steps"]
            ),

        "sleep_stage":
            sleep_stage,

        "activity":
            activity,

        # --------------------------------------------
        # Dynamic trends
        # --------------------------------------------

        "glucose_change_15m":
            float(
                latest[
                    "glucose_change_15m"
                ]
            ),

        "glucose_change_30m":
            float(
                latest[
                    "glucose_change_30m"
                ]
            ),

        "glucose_change_60m":
            float(
                latest[
                    "glucose_change_60m"
                ]
            ),

        "glucose_mean_1h":
            float(
                latest[
                    "glucose_mean_1h"
                ]
            ),

        "glucose_mean_2h":
            float(
                latest[
                    "glucose_mean_2h"
                ]
            ),

        "glucose_std_1h":
            float(
                latest[
                    "glucose_std_1h"
                ]
            ),

        "steps_1h":
            float(
                latest[
                    "steps_1h"
                ]
            ),

        "steps_2h":
            float(
                latest[
                    "steps_2h"
                ]
            ),

        "heart_rate_mean_1h":
            float(
                latest[
                    "heart_rate_mean_1h"
                ]
            ),

        "hrv_mean_1h":
            float(
                latest[
                    "hrv_mean_1h"
                ]
            ),

        "hour":
            int(
                latest["hour"]
            ),

        "day_of_week":
            int(
                latest["day_of_week"]
            ),

        "last_updated":
            latest["timestamp"]
    }

    return twin


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    ehr, training_data = load_data()

    patient_id = 42

    twin = get_digital_twin(
        patient_id,
        ehr,
        training_data
    )

    print("\n" + "=" * 60)

    print(
        "GLUCOTWIN DIGITAL TWIN"
    )

    print("=" * 60)

    for key, value in twin.items():

        print(
            f"{key}: {value}"
        )

    print("=" * 60)
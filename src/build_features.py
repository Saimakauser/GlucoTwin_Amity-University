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

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

ehr_path = os.path.join(
    RAW_DIR,
    "synthetic_ehr.csv"
)

wearable_path = os.path.join(
    RAW_DIR,
    "synthetic_wearable.csv"
)

print("=" * 60)
print("GLUCOTWIN FEATURE ENGINEERING")
print("=" * 60)

print("\nLoading EHR data...")

ehr = pd.read_csv(
    ehr_path
)

print(
    f"EHR records: {len(ehr)}"
)

print("\nLoading wearable data...")

wearable = pd.read_csv(
    wearable_path
)

wearable["timestamp"] = pd.to_datetime(
    wearable["timestamp"]
)

print(
    f"Wearable records: {len(wearable)}"
)


# ============================================================
# MERGE EHR + WEARABLE
# ============================================================

print("\nMerging EHR + wearable data...")

df = wearable.merge(
    ehr,
    on="patient_id",
    how="left"
)

df = df.sort_values(
    [
        "patient_id",
        "timestamp"
    ]
).reset_index(
    drop=True
)


# ============================================================
# ACTIVITY ENCODING
# ============================================================

activity_mapping = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}

df["activity_level"] = (
    df["activity"]
    .map(activity_mapping)
)


# ============================================================
# SLEEP STAGE ENCODING
# ============================================================

sleep_mapping = {
    "Awake": 0,
    "Light": 1,
    "REM": 2,
    "Deep": 3
}

df["sleep_stage_level"] = (
    df["sleep_stage"]
    .map(sleep_mapping)
)


# ============================================================
# TIME FEATURES
# ============================================================

df["hour"] = (
    df["timestamp"]
    .dt.hour
)

df["day_of_week"] = (
    df["timestamp"]
    .dt.dayofweek
)


# ============================================================
# GLUCOSE CHANGE FEATURES
# ============================================================

df["glucose_change_15m"] = (
    df.groupby("patient_id")["glucose"]
    .diff(1)
)

df["glucose_change_30m"] = (
    df.groupby("patient_id")["glucose"]
    .diff(2)
)

df["glucose_change_60m"] = (
    df.groupby("patient_id")["glucose"]
    .diff(4)
)


# ============================================================
# ROLLING GLUCOSE FEATURES
# ============================================================

df["glucose_mean_1h"] = (
    df.groupby("patient_id")["glucose"]
    .transform(
        lambda x:
        x.rolling(
            window=4,
            min_periods=4
        ).mean()
    )
)

df["glucose_mean_2h"] = (
    df.groupby("patient_id")["glucose"]
    .transform(
        lambda x:
        x.rolling(
            window=8,
            min_periods=8
        ).mean()
    )
)

df["glucose_std_1h"] = (
    df.groupby("patient_id")["glucose"]
    .transform(
        lambda x:
        x.rolling(
            window=4,
            min_periods=4
        ).std()
    )
)


# ============================================================
# ACTIVITY / STEP FEATURES
# ============================================================

df["steps_1h"] = (
    df.groupby("patient_id")["steps"]
    .transform(
        lambda x:
        x.rolling(
            window=4,
            min_periods=4
        ).sum()
    )
)

df["steps_2h"] = (
    df.groupby("patient_id")["steps"]
    .transform(
        lambda x:
        x.rolling(
            window=8,
            min_periods=8
        ).sum()
    )
)


# ============================================================
# HEART RATE FEATURES
# ============================================================

df["heart_rate_mean_1h"] = (
    df.groupby("patient_id")["heart_rate"]
    .transform(
        lambda x:
        x.rolling(
            window=4,
            min_periods=4
        ).mean()
    )
)


# ============================================================
# HRV FEATURES
# ============================================================

df["hrv_mean_1h"] = (
    df.groupby("patient_id")["hrv"]
    .transform(
        lambda x:
        x.rolling(
            window=4,
            min_periods=4
        ).mean()
    )
)


# ============================================================
# 2-HOUR FUTURE TARGET
# ============================================================

print(
    "\nCreating 2-hour prediction target..."
)

df["future_glucose_2h"] = (
    df.groupby("patient_id")["glucose"]
    .shift(-8)
)


# ============================================================
# ADVERSE EVENT DEFINITION
# ============================================================

# Target:
# Will glucose be >= 180 mg/dL exactly 2 hours later?

df["glucose_spike_2h"] = (
    df["future_glucose_2h"] >= 180
).astype(int)


# ============================================================
# MODEL FEATURES
# ============================================================

feature_columns = [

    # Historical / EHR
    "age",
    "bmi",
    "hba1c",
    "fasting_glucose",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "diabetes_years",
    "past_diagnosis",
    "medication",
    "genetic_risk",

    # Dynamic wearable
    "glucose",
    "heart_rate",
    "hrv",
    "steps",
    "sleep_stage",
    "activity",

    # Encoded dynamic features
    "sleep_stage_level",
    "activity_level",

    # Time
    "hour",
    "day_of_week",

    # Glucose dynamics
    "glucose_change_15m",
    "glucose_change_30m",
    "glucose_change_60m",
    "glucose_mean_1h",
    "glucose_mean_2h",
    "glucose_std_1h",

    # Activity dynamics
    "steps_1h",
    "steps_2h",

    # Physiological trends
    "heart_rate_mean_1h",
    "hrv_mean_1h"
]


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "sex",
    "past_diagnosis",
    "medication",
    "genetic_risk",
    "sleep_stage",
    "activity"
]

print(
    "\nEncoding categorical variables..."
)

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=False
)


# ============================================================
# FIND FINAL FEATURE COLUMNS
# ============================================================

final_feature_columns = []

for column in feature_columns:

    if column in df.columns:

        final_feature_columns.append(
            column
        )

    else:

        matching_columns = [
            col
            for col in df.columns
            if col.startswith(
                column + "_"
            )
        ]

        final_feature_columns.extend(
            matching_columns
        )


# Remove duplicate columns

final_feature_columns = list(
    dict.fromkeys(
        final_feature_columns
    )
)


# ============================================================
# KEEP REQUIRED COLUMNS
# ============================================================

required_columns = [
    "patient_id",
    "timestamp",
    "future_glucose_2h",
    "glucose_spike_2h"
]

output_columns = (
    required_columns
    + final_feature_columns
)

output_columns = list(
    dict.fromkeys(
        output_columns
    )
)

df = df[
    output_columns
]


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

before_rows = len(df)

df = df.dropna().reset_index(
    drop=True
)

after_rows = len(df)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

output_path = os.path.join(
    PROCESSED_DIR,
    "training_dataset.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)

print(
    "FEATURE ENGINEERING COMPLETE"
)

print("=" * 60)

print(
    f"Rows before cleaning: {before_rows}"
)

print(
    f"Rows after cleaning: {after_rows}"
)

print(
    f"Features available: {len(final_feature_columns)}"
)

print(
    "\nTarget distribution:"
)

print(
    df["glucose_spike_2h"]
    .value_counts()
)

print(
    "\nFinal feature columns:"
)

for column in final_feature_columns:

    print(
        f" - {column}"
    )

print(
    "\nSaved:"
)

print(
    output_path
)

print("=" * 60)
import os
import sys
import json
import joblib
import shap

import pandas as pd
import streamlit as st


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# =========================================================
# PROJECT IMPORTS
# =========================================================

try:

    from digital_twin import (
        load_data,
        get_digital_twin
    )

    from prediction_engine import (
        predict_patient,
        create_feature_vector
    )

except Exception as import_error:

    st.error(
        "GlucoTwin modules could not be loaded."
    )

    st.code(
        str(import_error)
    )

    st.stop()


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="GlucoTwin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #4b5563;
        background-color: #1f2937;
        color: #ffffff;
        margin-bottom: 20px;
        font-size: 16px;
        line-height: 1.6;
    }

    .info-box b {
        color: #ffffff;
    }

    .small-note {
        color: #9ca3af;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧬 GlucoTwin</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Personalized Digital Twin for 2-Hour Glucose Spike Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
        <b>Research Prototype</b><br>
        GlucoTwin demonstrates the fusion of synthetic historical
        EHR information with dynamic physiological data to estimate
        the probability of an upcoming glucose excursion at a
        2-hour prediction horizon.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_project_data():

    return load_data()


try:

    ehr, training_data = load_project_data()

except Exception as data_error:

    st.error(
        "Unable to load GlucoTwin data."
    )

    st.exception(data_error)

    st.stop()


# =========================================================
# BASIC DATA VALIDATION
# =========================================================

required_ehr_columns = [
    "patient_id",
    "age",
    "sex",
    "bmi",
    "hba1c",
    "fasting_glucose",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "diabetes_years",
    "past_diagnosis",
    "medication",
    "genetic_risk"
]

required_training_columns = [
    "patient_id",
    "timestamp",
    "glucose",
    "heart_rate",
    "hrv",
    "steps"
]


missing_ehr = [
    col
    for col in required_ehr_columns
    if col not in ehr.columns
]


missing_training = [
    col
    for col in required_training_columns
    if col not in training_data.columns
]


if missing_ehr:

    st.error(
        "Required EHR columns are missing:"
    )

    st.write(
        missing_ehr
    )

    st.stop()


if missing_training:

    st.error(
        "Required training-data columns are missing:"
    )

    st.write(
        missing_training
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🧬 GlucoTwin"
)

st.sidebar.header(
    "Patient Selection"
)


patient_ids = sorted(
    ehr["patient_id"]
    .dropna()
    .unique()
    .tolist()
)


if len(patient_ids) == 0:

    st.error(
        "No patients found in EHR data."
    )

    st.stop()


default_index = 0

if 42 in patient_ids:

    default_index = patient_ids.index(42)


patient_id = st.sidebar.selectbox(
    "Select Patient",
    patient_ids,
    index=default_index
)


st.sidebar.markdown("---")

st.sidebar.caption(
    "Synthetic / anonymized research data"
)

st.sidebar.caption(
    "Digital Twin Challenge 2026"
)


# =========================================================
# DIGITAL TWIN
# =========================================================

try:

    twin = get_digital_twin(
        patient_id,
        ehr,
        training_data
    )

except Exception as twin_error:

    st.error(
        "Unable to create the Digital Twin."
    )

    st.exception(twin_error)

    st.stop()


# =========================================================
# MODEL PREDICTION
# =========================================================

try:

    result = predict_patient(
        twin
    )

except Exception as prediction_error:

    st.error(
        "Unable to generate the prediction."
    )

    st.exception(prediction_error)

    st.stop()


risk_percentage = float(
    result.get(
        "risk_percentage",
        0.0
    )
)

probability = float(
    result.get(
        "probability",
        0.0
    )
)

prediction = int(
    result.get(
        "prediction",
        0
    )
)

category = str(
    result.get(
        "category",
        "LOW"
    )
)


# =========================================================
# PATIENT DIGITAL TWIN
# =========================================================

st.markdown(
    '<div class="section-title">'
    '👤 Patient Digital Twin'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Age",
        f"{float(twin['age']):.0f}"
    )

    st.metric(
        "BMI",
        f"{float(twin['bmi']):.1f}"
    )


with col2:

    st.metric(
        "HbA1c",
        f"{float(twin['hba1c']):.1f}%"
    )

    st.metric(
        "Fasting Glucose",
        f"{float(twin['fasting_glucose']):.1f} mg/dL"
    )


with col3:

    st.metric(
        "Systolic BP",
        f"{float(twin['systolic_bp']):.0f}"
    )

    st.metric(
        "Diastolic BP",
        f"{float(twin['diastolic_bp']):.0f}"
    )


with col4:

    st.metric(
        "Diabetes Years",
        f"{float(twin['diabetes_years']):.0f}"
    )

    st.metric(
        "Genetic Risk",
        str(twin["genetic_risk"])
    )


st.write(
    f"**Past diagnosis:** {twin['past_diagnosis']}"
)

st.write(
    f"**Medication:** {twin['medication']}"
)


# =========================================================
# CURRENT PHYSIOLOGICAL STATE
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📡 Current Physiological State'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Glucose",
        f"{float(twin['current_glucose']):.1f} mg/dL"
    )


with col2:

    st.metric(
        "Heart Rate",
        f"{float(twin['heart_rate']):.1f} bpm"
    )


with col3:

    st.metric(
        "HRV",
        f"{float(twin['hrv']):.1f} ms"
    )


with col4:

    st.metric(
        "Steps",
        f"{float(twin['steps']):.0f}"
    )


with col5:

    st.metric(
        "Sleep Stage",
        str(twin["sleep_stage"])
    )


st.write(
    f"**Activity:** {twin['activity']}"
)

st.write(
    f"**Last updated:** {twin['last_updated']}"
)


# =========================================================
# 2-HOUR PREDICTION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🔮 2-Hour Glucose Prediction'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Spike Probability",
        f"{risk_percentage:.2f}%"
    )


with col2:

    st.metric(
        "Prototype Category",
        category
    )


with col3:

    if prediction == 1:

        output_text = "Spike predicted"

    else:

        output_text = "No spike predicted"

    st.metric(
        "Model Output",
        output_text
    )


# ---------------------------------------------------------
# Progress bar
# ---------------------------------------------------------

progress_value = float(
    max(
        0.0,
        min(
            risk_percentage / 100.0,
            1.0
        )
    )
)

st.progress(
    progress_value
)


st.caption(
    "Target: probability that glucose reaches "
    "≥180 mg/dL at the exact 2-hour prediction horizon."
)


# =========================================================
# DYNAMIC SHAP EXPLAINABILITY
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🧠 Why Did the Model Predict This?'
    '</div>',
    unsafe_allow_html=True
)


try:

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "glucotwin_xgboost.pkl"
    )

    features_path = os.path.join(
        BASE_DIR,
        "models",
        "model_features.json"
    )


    if not os.path.exists(model_path):

        raise FileNotFoundError(
            "Model file not found: "
            + model_path
        )


    if not os.path.exists(features_path):

        raise FileNotFoundError(
            "Feature file not found: "
            + features_path
        )


    model = joblib.load(
        model_path
    )


    with open(
        features_path,
        "r"
    ) as f:

        feature_names = json.load(f)


    # -----------------------------------------------------
    # Validate feature list
    # -----------------------------------------------------

    if not isinstance(
        feature_names,
        list
    ):

        raise ValueError(
            "model_features.json must contain a list."
        )


    # -----------------------------------------------------
    # Create model input
    # -----------------------------------------------------

    X_patient = create_feature_vector(
        twin,
        feature_names
    )


    # -----------------------------------------------------
    # SHAP explainer
    # -----------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )


    shap_values = explainer.shap_values(
        X_patient
    )


    # -----------------------------------------------------
    # Convert SHAP output to 1D array
    # -----------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) > 1:

            shap_array = shap_values[1][0]

        else:

            shap_array = shap_values[0]


    else:

        shap_array = shap_values


        if hasattr(
            shap_array,
            "ndim"
        ):

            if shap_array.ndim == 3:

                shap_array = (
                    shap_array[0, :, 1]
                )

            elif shap_array.ndim == 2:

                shap_array = (
                    shap_array[0]
                )


    shap_array = list(
        shap_array
    )


    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if len(shap_array) != len(
        feature_names
    ):

        raise ValueError(
            "SHAP feature count does not match "
            "model feature count."
        )


    # -----------------------------------------------------
    # Explanation dataframe
    # -----------------------------------------------------

    shap_df = pd.DataFrame(
        {
            "feature": feature_names,
            "value": X_patient.iloc[0].values,
            "shap_value": shap_array
        }
    )


    shap_df["absolute_shap"] = (
        shap_df["shap_value"]
        .abs()
    )


    shap_df = shap_df.sort_values(
        "absolute_shap",
        ascending=False
    )


    # -----------------------------------------------------
    # Format feature names
    # -----------------------------------------------------

    def format_feature_name(
        feature
    ):

        name = str(feature)

        replacements = [
            (
                "past_diagnosis_",
                "Past diagnosis: "
            ),
            (
                "medication_",
                "Medication: "
            ),
            (
                "genetic_risk_",
                "Genetic risk: "
            ),
            (
                "sleep_stage_",
                "Sleep stage: "
            ),
            (
                "activity_",
                "Activity: "
            )
        ]

        for old, new in replacements:

            name = name.replace(
                old,
                new
            )

        name = name.replace(
            "_",
            " "
        )

        return name.title()


    # -----------------------------------------------------
    # Top 5
    # -----------------------------------------------------

    top_shap = shap_df.head(
        5
    ).copy()


    top_shap["Feature"] = (
        top_shap["feature"]
        .apply(
            format_feature_name
        )
    )


    top_shap["Direction"] = (
        top_shap["shap_value"]
        .apply(
            lambda value:
            "Increases prediction"
            if float(value) > 0
            else "Decreases prediction"
        )
    )


    top_shap["SHAP Contribution"] = (
        top_shap["shap_value"]
        .apply(
            lambda value:
            f"{float(value):.4f}"
        )
    )


    top_shap["Feature Value"] = (
        top_shap["value"]
    )


    display_shap = top_shap[
        [
            "Feature",
            "Feature Value",
            "SHAP Contribution",
            "Direction"
        ]
    ]


    st.caption(
        f"Personalized explanation generated "
        f"for Patient {patient_id}"
    )


    st.dataframe(
        display_shap,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # SHAP chart
    # -----------------------------------------------------

    chart_df = top_shap[
        [
            "Feature",
            "shap_value"
        ]
    ].copy()


    chart_df = chart_df.sort_values(
        "shap_value"
    )


    chart_df = chart_df.set_index(
        "Feature"
    )


    st.bar_chart(
        chart_df[
            "shap_value"
        ]
    )


    st.caption(
        "Positive SHAP values push the model toward a "
        "higher predicted probability. Negative values "
        "push the model toward a lower predicted probability."
    )


except Exception as shap_error:

    st.warning(
        "SHAP explanation could not be generated."
    )

    st.caption(
        f"Technical detail: {shap_error}"
    )


# =========================================================
# GLUCOSE TIMELINE
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📈 Glucose Timeline'
    '</div>',
    unsafe_allow_html=True
)


try:

    patient_history = training_data[
        training_data["patient_id"] == patient_id
    ].copy()


    if len(patient_history) > 0:

        patient_history["timestamp"] = (
            pd.to_datetime(
                patient_history["timestamp"]
            )
        )


        patient_history = (
            patient_history
            .sort_values(
                "timestamp"
            )
        )


        glucose_chart = (
            patient_history[
                [
                    "timestamp",
                    "glucose"
                ]
            ]
            .set_index(
                "timestamp"
            )
        )


        st.line_chart(
            glucose_chart
        )


    else:

        st.info(
            "No glucose timeline available."
        )


except Exception as chart_error:

    st.warning(
        "Glucose timeline could not be displayed."
    )

    st.caption(
        str(chart_error)
    )


# =========================================================
# DIGITAL TWIN DYNAMICS
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📊 Digital Twin Dynamics'
    '</div>',
    unsafe_allow_html=True
)


try:

    dynamic_columns = [
        "timestamp",
        "glucose",
        "heart_rate",
        "hrv",
        "steps",
        "sleep_stage_level",
        "activity_level",
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


    available_columns = [
        column
        for column in dynamic_columns
        if column in patient_history.columns
    ]


    if available_columns:

        display_history = (
            patient_history[
                available_columns
            ]
            .tail(20)
            .copy()
        )


        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Dynamic feature data unavailable."
        )


except Exception as dynamic_error:

    st.warning(
        "Digital Twin dynamics could not be displayed."
    )

    st.caption(
        str(dynamic_error)
    )


# =========================================================
# MODEL VALIDATION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📊 Model Validation'
    '</div>',
    unsafe_allow_html=True
)


metrics_path = os.path.join(
    BASE_DIR,
    "models",
    "evaluation_metrics.json"
)


if os.path.exists(
    metrics_path
):

    try:

        with open(
            metrics_path,
            "r"
        ) as f:

            metrics = json.load(f)


        accuracy = float(
            metrics.get(
                "accuracy",
                0.0
            )
        )


        precision = float(
            metrics.get(
                "precision",
                0.0
            )
        )


        recall = float(
            metrics.get(
                "recall",
                0.0
            )
        )


        f1_value = float(
            metrics.get(
                "f1",
                metrics.get(
                    "f1_score",
                    0.0
                )
            )
        )


        roc_auc = float(
            metrics.get(
                "roc_auc",
                0.0
            )
        )


        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )


        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy:.3f}"
            )


        with col2:

            st.metric(
                "Precision",
                f"{precision:.3f}"
            )


        with col3:

            st.metric(
                "Recall",
                f"{recall:.3f}"
            )


        with col4:

            st.metric(
                "F1 Score",
                f"{f1_value:.3f}"
            )


        with col5:

            st.metric(
                "ROC-AUC",
                f"{roc_auc:.3f}"
            )


    except Exception as metrics_error:

        st.warning(
            "Model validation metrics could not be displayed."
        )

        st.caption(
            str(metrics_error)
        )

else:

    st.info(
        "Model validation file not found."
    )


# =========================================================
# MODEL VISUALIZATIONS
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📉 Model Evaluation Visualizations'
    '</div>',
    unsafe_allow_html=True
)


confusion_matrix_path = os.path.join(
    BASE_DIR,
    "models",
    "confusion_matrix.png"
)

roc_curve_path = os.path.join(
    BASE_DIR,
    "models",
    "roc_curve.png"
)

feature_importance_path = os.path.join(
    BASE_DIR,
    "models",
    "feature_importance.png"
)


col1, col2 = st.columns(2)


with col1:

    if os.path.exists(
        confusion_matrix_path
    ):

        st.image(
            confusion_matrix_path,
            caption="Confusion Matrix",
            use_container_width=True
        )

    else:

        st.info(
            "Confusion matrix image not found."
        )


    if os.path.exists(
        feature_importance_path
    ):

        st.image(
            feature_importance_path,
            caption="Feature Importance",
            use_container_width=True
        )

    else:

        st.info(
            "Feature importance image not found."
        )


with col2:

    if os.path.exists(
        roc_curve_path
    ):

        st.image(
            roc_curve_path,
            caption="ROC Curve",
            use_container_width=True
        )

    else:

        st.info(
            "ROC curve image not found."
        )


# =========================================================
# DATA FUSION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🔗 Digital Twin Data Fusion'
    '</div>',
    unsafe_allow_html=True
)


fusion_col1, fusion_col2, fusion_col3 = (
    st.columns(3)
)


with fusion_col1:

    st.info(
        """
        **STATIC / HISTORICAL EHR**

        • Demographics

        • Past diagnoses

        • Laboratory results

        • Medication

        • Genetic risk
        """
    )


with fusion_col2:

    st.info(
        """
        **DYNAMIC WEARABLE / IoT**

        • CGM glucose

        • Heart rate

        • HRV

        • Step count

        • Sleep stages

        • Activity
        """
    )


with fusion_col3:

    st.info(
        """
        **PERSONALIZED DIGITAL TWIN**

        • Current physiology

        • Historical trends

        • Rolling features

        • Personalized model input

        • 2-hour prediction
        """
    )


# =========================================================
# COMPLETE DIGITAL TWIN
# =========================================================

with st.expander(
    "🔍 View Complete Digital Twin State"
):

    twin_table = pd.DataFrame(
        list(twin.items()),
        columns=[
            "Variable",
            "Value"
        ]
    )


    st.dataframe(
        twin_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PROJECT PIPELINE
# =========================================================

st.markdown(
    '<div class="section-title">'
    '⚙️ GlucoTwin Pipeline'
    '</div>',
    unsafe_allow_html=True
)


pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4 = (
    st.columns(4)
)


with pipeline_col1:

    st.success(
        "1️⃣ EHR\n\n"
        "Historical patient profile"
    )


with pipeline_col2:

    st.success(
        "2️⃣ Wearables\n\n"
        "Dynamic physiological signals"
    )


with pipeline_col3:

    st.success(
        "3️⃣ Digital Twin\n\n"
        "Personalized patient state"
    )


with pipeline_col4:

    st.success(
        "4️⃣ AI Prediction\n\n"
        "2-hour glucose excursion"
    )


# =========================================================
# TECHNICAL SUMMARY
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🛠️ Technical Summary'
    '</div>',
    unsafe_allow_html=True
)


tech_col1, tech_col2, tech_col3 = st.columns(3)


with tech_col1:

    st.info(
        """
        **Data**

        • Synthetic EHR

        • Synthetic wearable streams

        • 100 patients

        • 7-day physiological timelines
        """
    )


with tech_col2:

    st.info(
        """
        **Machine Learning**

        • XGBoost

        • Binary classification

        • 45 engineered features

        • 2-hour prediction horizon
        """
    )


with tech_col3:

    st.info(
        """
        **Explainability**

        • SHAP TreeExplainer

        • Patient-specific explanation

        • Top contributing factors

        • Doctor-facing visualization
        """
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("---")

st.warning(
    """
    **Research Prototype Disclaimer**

    GlucoTwin is a proof-of-concept Digital Twin developed using
    synthetic/anonymized data. Model outputs are intended only
    for research, demonstration and educational purposes.

    This prototype is not a medical device and should not be used
    for clinical diagnosis, treatment or medical decision-making.
    """
)


st.caption(
    "GlucoTwin | Digital Twin Challenge 2026"
)
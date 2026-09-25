# GlucoTwin

## A Personalized Digital Twin for 2-Hour Glucose Spike Prediction

**Happiest Health Digital Twin Challenge 2026 — Phase 1 Prototype**

---

## Participant

- **Name:** Saima Kauser
- **University:** Amity University
- **Participation:** Solo / Individual
- **Incubator:** N/A

---

# 1. Project Overview

GlucoTwin is a research-oriented Digital Twin prototype designed to demonstrate personalized prediction of an upcoming glucose excursion using a combination of historical Electronic Health Record (EHR) information and dynamic physiological data.

The system creates a patient-specific digital representation by combining:

- Demographic information
- Historical diagnoses
- Laboratory measurements
- Medication information
- Genetic-risk indicators
- Continuous glucose measurements
- Heart rate
- Heart-rate variability (HRV)
- Step count
- Sleep stage
- Activity level

The prototype predicts the probability that a patient's glucose will reach or exceed **180 mg/dL at the 2-hour prediction horizon**.

The goal is to demonstrate how continuously updated physiological information can be combined with historical patient information to create a personalized computational representation of a patient.

> **Important:** GlucoTwin is a research prototype using synthetic data. It is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.

---

# 2. Problem Statement

Glucose levels can change significantly depending on a combination of historical health characteristics and current physiological conditions.

Traditional patient information is often stored as relatively static clinical records, while wearable devices continuously generate dynamic physiological measurements.

A key challenge is:

> **How can historical patient information and continuously changing physiological signals be combined to create a personalized model capable of predicting an upcoming glucose excursion?**

GlucoTwin addresses this problem by creating a computational Digital Twin that combines static patient context with dynamic physiological state.

---

# 3. Proposed Solution

GlucoTwin follows a multi-stage pipeline:

1. Generate anonymized synthetic EHR data.
2. Generate simulated dynamic wearable/IoT time-series data.
3. Fuse EHR and wearable data.
4. Engineer temporal and physiological features.
5. Train a machine-learning prediction model.
6. Generate a patient-specific Digital Twin state.
7. Predict the probability of a glucose spike at the 2-hour horizon.
8. Explain the prediction using SHAP.
9. Present the Digital Twin state and prediction through a Streamlit dashboard.

---

# 4. Digital Twin Concept

The GlucoTwin Digital Twin combines two types of information.

## Static / Historical Patient Context

The EHR component contains:

- Age
- Sex
- BMI
- HbA1c
- Fasting glucose
- Blood pressure
- Cholesterol
- Diabetes duration
- Previous diagnoses
- Medication
- Genetic-risk indicator

## Dynamic Physiological State

The simulated wearable/IoT stream contains:

- Glucose
- Heart rate
- HRV
- Step count
- Sleep stage
- Activity level
- Timestamp

The Digital Twin combines these two layers to represent the patient's current computational state.

---

# 5. Prediction Target

The model predicts:

> **Probability that glucose reaches ≥180 mg/dL at the 2-hour prediction horizon.**

The wearable data is sampled at 15-minute intervals.

Therefore:

**2 hours / 15 minutes = 8 time steps**

The prediction target is created using the future glucose value at this horizon.

---

# 6. Dataset

The prototype uses fully synthetic data.

No real patient data is used.

## Synthetic EHR Dataset

The project generates:

- 100 synthetic patients
- Demographic attributes
- Clinical laboratory measurements
- Diagnosis history
- Medication information
- Genetic-risk indicators

## Synthetic Wearable Dataset

The prototype generates:

- 7 days of simulated observations per patient
- 15-minute sampling interval
- 67,200 wearable observations before preprocessing

Dynamic signals include:

- Glucose
- Heart Rate
- HRV
- Steps
- Sleep Stage
- Activity

---

# 7. Data Fusion

The EHR and wearable streams are joined using:

`patient_id`

The resulting dataset contains both long-term patient context and short-term physiological state.

This allows the machine-learning model to learn relationships between historical patient characteristics and current physiological conditions.

---

# 8. Feature Engineering

The feature engineering pipeline creates temporal and physiological features including:

## Current State

- Current glucose
- Heart rate
- HRV
- Steps
- Sleep stage
- Activity

## Glucose Dynamics

- 15-minute glucose change
- 30-minute glucose change
- 60-minute glucose change
- 1-hour rolling glucose mean
- 2-hour rolling glucose mean
- 1-hour glucose standard deviation

## Activity Dynamics

- 1-hour step count
- 2-hour step count

## Physiological Dynamics

- 1-hour mean heart rate
- 1-hour mean HRV

## Temporal Features

- Hour of day
- Day of week

## Historical Clinical Features

- Age
- BMI
- HbA1c
- Fasting glucose
- Blood pressure
- Cholesterol
- Diabetes duration
- Diagnosis
- Medication
- Genetic-risk indicator

The final training dataset contains **45 model features**.

---

# 9. Machine Learning Model

The prototype uses an:

## XGBoost Classifier

The model is trained as a binary classification problem.

### Output

- `0` → Glucose does not reach the defined threshold at the 2-hour horizon
- `1` → Glucose reaches ≥180 mg/dL at the 2-hour horizon

The dashboard additionally displays the model probability as a percentage.

---

# 10. Model Evaluation

The prototype was evaluated using a stratified train/test split.

## Results

| Metric | Result |
|---|---:|
| Accuracy | 90.59% |
| Precision | 84.51% |
| Recall | 80.66% |
| F1 Score | 82.54% |
| ROC-AUC | 96.69% |

These results demonstrate the predictive capability of the prototype on the generated synthetic dataset.

## Important Evaluation Limitation

The current prototype uses a random row-level train/test split.

Because multiple observations can originate from the same synthetic patient and time series, this evaluation setup can introduce patient-level or temporal information leakage.

Therefore, these metrics should **not** be interpreted as clinical performance.

A future version should use patient-level and/or time-based validation to provide a more rigorous evaluation.

---

# 11. Explainable AI

GlucoTwin uses **SHAP (SHapley Additive exPlanations)** to provide feature-level explanations for individual predictions.

The dashboard displays the features that contributed most strongly to the current prediction.

Example contributors can include:

- Fasting glucose
- Current glucose
- HbA1c
- Time of day
- Recent glucose averages
- Sleep-related features

This provides transparency into the model's prediction rather than presenting only a probability.

---

# 12. Digital Twin State

For a selected patient, GlucoTwin constructs a current Digital Twin state containing:

- Patient profile
- Historical clinical context
- Current physiological state
- Recent temporal dynamics
- Prediction
- Explainability information

The Digital Twin state can be updated using the latest available physiological observation.

---

# 13. Doctor-Facing Conceptual Dashboard

The project includes a Streamlit dashboard designed as a conceptual research interface.

## Patient Profile

The dashboard displays:

- Age
- BMI
- HbA1c
- Fasting glucose
- Blood pressure
- Diabetes duration
- Diagnosis
- Medication
- Genetic-risk indicator

## Current Physiological State

The dashboard displays:

- Current glucose
- Heart rate
- HRV
- Steps
- Sleep stage
- Activity
- Last update timestamp

## 2-Hour Prediction

The dashboard displays:

- Prediction probability
- Prototype risk category
- Binary model output

## Explainability

The dashboard provides:

- SHAP feature contributions

## Visualization

The dashboard includes:

- Glucose timeline
- Confusion matrix
- ROC curve
- Feature importance

## Digital Twin

The dashboard provides:

- Current computational patient state
- Historical context
- Dynamic physiological context

---

# 14. System Architecture


                  ┌─────────────────────────┐
                  │   Synthetic EHR Data    │
                  │                         │
                  │ Demographics            │
                  │ Diagnoses               │
                  │ Lab Results             │
                  │ Medication              │
                  │ Genetic Risk            │
                  └────────────┬────────────┘
                               │
                               │
                               ▼
                  ┌─────────────────────────┐
                  │   Synthetic Wearable    │
                  │       Data Stream       │
                  │                         │
                  │ Glucose                 │
                  │ Heart Rate              │
                  │ HRV                     │
                  │ Steps                   │
                  │ Sleep                   │
                  │ Activity                │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │       Data Fusion       │
                  │                         │
                  │   EHR + Wearable Data   │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │   Feature Engineering   │
                  │                         │
                  │ Temporal Features       │
                  │ Rolling Statistics      │
                  │ Physiological Dynamics  │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │      XGBoost Model      │
                  │                         │
                  │ 2-Hour Glucose Spike    │
                  │ Prediction              │
                  └────────────┬────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
      ┌────────────────────┐      ┌────────────────────┐
      │ SHAP Explainability│      │   Digital Twin     │
      │                    │      │                    │
      │ Feature            │      │ Patient State      │
      │ Contributions      │      │                    │
      └──────────┬─────────┘      └──────────┬─────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │   Streamlit Dashboard   │
                  │                         │
                  │ Patient View            │
                  │ Prediction              │
                  │ Explanation             │
                  │ Trends                  │
                  │ Model Validation        │
                  └─────────────────────────┘

                  
                  
# 15. Technology Stack

Programming Language
Python
Machine Learning
XGBoost
Scikit-learn
Explainable AI
SHAP
Data Processing
Pandas
NumPy
Visualization
Matplotlib
Plotly
Dashboard
Streamlit
Model Persistence
Joblib
Development Environment
Visual Studio Code
Git
GitHub
16. Project Structure
GlucoTwin/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── generate_data.py
│   ├── build_features.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── digital_twin.py
│   ├── prediction_engine.py
│   └── explain_prediction.py
│
├── models/
│
├── dashboard/
│   └── app.py
│
├── docs/
│
├── presentation/
│
├── notebooks/
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore

17. How to Run the Project

Step 1 — Clone the Repository
git clone <PUBLIC_GITHUB_REPOSITORY_URL>
cd GlucoTwin

Step 2 — Create a Virtual Environment

For Windows:

python -m venv .venv

Activate it:

.venv\Scripts\activate
Step 3 — Install Dependencies
pip install -r requirements.txt
Step 4 — Generate Synthetic Data
python src/generate_data.py
Step 5 — Build Features
python src/build_features.py
Step 6 — Train the Model
python src/train_model.py
Step 7 — Evaluate the Model
python src/evaluate_model.py
Step 8 — Run the Dashboard
streamlit run dashboard/app.py

The Streamlit application will open in the browser.

 ## 19. Project Demo Video

A 2–5 minute demonstration of the GlucoTwin Digital Twin prototype:

**YouTube Demo:**  
https://youtu.be/YN3yz5F8esc

The video demonstrates the Digital Twin state, EHR and wearable data fusion, 2-hour glucose prediction, SHAP explainability, model evaluation, and the conceptual doctor-facing dashboard.

18. Reproducibility

The project uses deterministic random seeds where applicable to make the synthetic data generation and model training reproducible.

All data used by the prototype is synthetic and can be regenerated using the provided scripts.


19. Privacy and Data Safety

GlucoTwin does not use real patient information.

The prototype is based entirely on:

Synthetic EHR data
Simulated physiological time-series data

No personally identifiable health information is included in the repository.


20. Limitations

This prototype has several limitations:

The data is synthetic rather than real-world clinical data.
Wearable signals are simulated rather than collected from physical devices.
The model has not undergone clinical validation.
The current evaluation uses a random row-level split and may contain patient/time leakage.
The prediction threshold is a prototype research definition and does not represent an individualized clinical decision threshold.
The Digital Twin represents selected physiological and clinical variables rather than a complete whole-body digital twin.
The dashboard is a conceptual research interface and is not intended for clinical deployment.


21. Future Work

Future development could include:

Validation using appropriate open clinical datasets
Patient-level and time-based validation
Real wearable/IoT integration
Continuous model updating
Personalized baseline modeling
More advanced temporal models
Federated learning for privacy-preserving training
Calibration of predicted probabilities
External validation across different patient populations
Integration with clinical workflow systems
Prospective evaluation in controlled research settings

22. Research Prototype Disclaimer

GlucoTwin is an academic/research prototype developed using synthetic data for demonstration purposes.
It is not a medical device, diagnostic system, treatment recommendation system, or substitute for professional medical judgment.
The predictions shown by this prototype should not be used to make real-world medical decisions.

23. License

This project is released under the MIT License.
See the LICENSE file for details.



24. Author

Saima Kauser
Amity University
Solo Participant


25. Project Objective

The objective of GlucoTwin is to demonstrate how Digital Twin technology can combine historical patient context with continuously changing physiological signals to support personalized predictive healthcare research.

The prototype focuses on a specific, measurable outcome:

Predicting whether glucose reaches ≥180 mg/dL at the 2-hour prediction horizon.

This focused use case demonstrates the complete Digital Twin pipeline from synthetic data generation and multimodal data fusion to machine-learning prediction, explainability, and an interactive visualization interface.
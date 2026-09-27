# GlucoTwin

## A Personalized Digital Twin for 2-Hour Glucose Spike Prediction

**Happiest Health Digital Twin Challenge 2026 — Phase 1 Prototype**

### Participant

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

A key challenge is therefore:

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

Therefore, the 2-hour future glucose value corresponds to an 8-step temporal shift:

```text
2 hours / 15 minutes = 8 time steps

6. Synthetic Dataset

The prototype uses fully synthetic data.

Synthetic EHR
100 simulated patients
Demographics
Medical history
Laboratory values
Medication information
Genetic-risk indicators
Synthetic Wearable / IoT Data
7 days of simulated physiological data
15-minute sampling interval
67,200 initial wearable records
Glucose
Heart rate
HRV
Steps
Sleep stage
Activity level

After temporal feature construction and removal of rows without a valid 2-hour future target, the training dataset contains approximately 65,700 usable records.

No real patient data or personally identifiable health information is used.

7. Data Fusion and Feature Engineering

The system joins the static EHR information with dynamic wearable measurements using the patient identifier.

Temporal and physiological features include:

Glucose change over 15 minutes
Glucose change over 30 minutes
Glucose change over 60 minutes
1-hour rolling glucose mean
2-hour rolling glucose mean
1-hour glucose variability
1-hour step count
2-hour step count
1-hour mean heart rate
1-hour mean HRV
Hour of day
Day of week
Sleep-stage encoding
Activity-level encoding

This creates a fused patient state that combines historical context with recent physiological dynamics.

8. Machine Learning Model

The prototype uses XGBoost for binary classification.

Model Objective

The model estimates the probability that the patient's glucose will reach or exceed 180 mg/dL at the 2-hour prediction horizon.

Model Configuration
Algorithm: XGBoost Classifier
Estimators: 300
Maximum depth: 6
Learning rate: 0.05
Subsample: 0.8
Column sampling: 0.8
Objective: Binary Logistic Classification
Evaluation metric: Log Loss
Random state: 42

The model uses the fused EHR and wearable features to generate a personalized prediction.

9. Model Evaluation

The prototype was evaluated using a stratified 80/20 train-test split.

Results
Metric	Result
Accuracy	90.59%
Precision	84.51%
Recall	80.66%
F1 Score	82.54%
ROC-AUC	96.69%

The repository also contains:

Confusion matrix
ROC curve
Feature importance visualization
Evaluation metrics JSON
Evaluation Limitation

The current prototype uses a random row-level train-test split.

Because multiple time-series observations from the same simulated patient can occur in both training and testing data, this evaluation may contain patient/time leakage.

Therefore, these results should not be interpreted as clinical validation or real-world performance.

A stronger future evaluation would use:

Patient-level holdout
Time-based validation
External validation
Real-world prospective evaluation
10. Explainability with SHAP

GlucoTwin uses SHAP (SHapley Additive exPlanations) to explain individual predictions.

SHAP identifies which features are contributing toward increasing or decreasing the model's predicted probability.

Examples of influential features include:

Fasting glucose
HbA1c
Current glucose
Glucose rolling averages
Hour of day
HRV
Sleep-related features

This provides a more interpretable view of the prediction rather than presenting only a probability.

11. Digital Twin Dashboard

The project includes a conceptual doctor-facing Streamlit dashboard.

The dashboard provides:

Patient Digital Twin
Age
BMI
HbA1c
Fasting glucose
Blood pressure
Diabetes duration
Diagnosis
Medication
Genetic-risk indicator
Current Physiological State
Current glucose
Heart rate
HRV
Steps
Sleep stage
Activity
Last updated timestamp
Prediction
2-hour glucose spike probability
Prototype risk category
Binary prediction
Explainability
SHAP feature contributions
Visualization
Glucose timeline
Digital Twin dynamics
Model evaluation
Confusion matrix
ROC curve
Feature importance
12. Technical Architecture
             SYNTHETIC EHR DATA
                     |
                     |
                     v
        +-------------------------+
        | Historical Patient Data |
        | Demographics            |
        | Diagnoses               |
        | Labs                    |
        | Medication              |
        | Genetic Risk            |
        +------------+------------+
                     |
                     |
                     | Patient ID
                     |
                     v
        +-------------------------+
        | Dynamic Wearable / IoT  |
        | CGM / Glucose           |
        | Heart Rate              |
        | HRV                     |
        | Steps                   |
        | Sleep                   |
        | Activity                |
        +------------+------------+
                     |
                     v
             DATA FUSION
                     |
                     v
          FEATURE ENGINEERING
                     |
                     v
             XGBOOST MODEL
                     |
             +-------+-------+
             |               |
             v               v
       2-HOUR GLUCOSE     SHAP
          PREDICTION      EXPLANATION
             |               |
             +-------+-------+
                     |
                     v
          DIGITAL TWIN STATE
                     |
                     v
          STREAMLIT DASHBOARD
13. Technology Stack
Programming
Python
Data Processing
Pandas
NumPy
Machine Learning
XGBoost
Scikit-learn
Explainability
SHAP
Visualization
Matplotlib
Plotly
Dashboard
Streamlit
Model Persistence
Joblib
Development
VS Code
Git
GitHub
14. Project Structure
GlucoTwin/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
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
├── dashboard/
│   └── app.py
│
├── docs/
│   ├── architecture.pdf
│   └── architecture.pptx
│
├── presentation/
│   └── GlucoTwin_Pitch_Deck.pptx
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
15. How to Run
Step 1 — Clone the repository
git clone https://github.com/Saimakauser/GlucoTwin_Amity-University.git
cd GlucoTwin_Amity-University
Step 2 — Create a virtual environment
python -m venv .venv
Step 3 — Activate the environment
Windows PowerShell
.\.venv\Scripts\Activate.ps1
Step 4 — Install dependencies
pip install -r requirements.txt
Step 5 — Run the dashboard
streamlit run dashboard/app.py

The Streamlit dashboard will open in the browser.

16. Reproducibility

The repository contains the synthetic dataset, trained model, feature definitions, evaluation outputs, architecture documents, and presentation required to reproduce and inspect the prototype.

The clean-clone test was also performed by cloning the public GitHub repository into a separate directory, creating a fresh virtual environment, installing the requirements, and launching the Streamlit dashboard successfully.

17. Project Demo Video

A 2–5 minute demonstration of the GlucoTwin Digital Twin prototype:

YouTube Demo:
https://youtu.be/YN3yz5F8esc

The video demonstrates:

Digital Twin state
EHR and wearable data fusion
2-hour glucose prediction
SHAP explainability
Model evaluation
Conceptual doctor-facing dashboard
18. Required Submission Documents
Architecture Diagram

The system architecture diagram is provided in both PDF and PowerPoint formats:

Architecture Diagram – PDF
Architecture Diagram – PowerPoint
Project Presentation

The complete project presentation is provided in PowerPoint format:

GlucoTwin Project Presentation – PPTX
19. Privacy and Data Safety

This project uses synthetic data only.

No real patient records, personally identifiable information, or confidential healthcare data are included in the repository.

The project is intended only as a research and educational prototype.

20. Limitations

The current prototype has several limitations:

The dataset is synthetic.
The wearable signals are simulated rather than collected from real devices.
The model has not undergone clinical validation.
The current evaluation uses a random row-level split and may contain patient/time leakage.
The glucose threshold is a prototype target and should not be interpreted as a clinical decision rule.
Real-world deployment would require extensive validation, privacy controls, monitoring, and clinical collaboration.
21. Future Work

Potential future improvements include:

Patient-level and time-based validation
External validation using appropriate open datasets
Integration with real CGM and wearable devices
More advanced temporal models
Personalized model calibration
Continuous model monitoring
Privacy-preserving learning
Clinical validation
Integration with healthcare workflows
22. Research Prototype Disclaimer

GlucoTwin is a research prototype created using synthetic data for the Happiest Health Digital Twin Challenge 2026. It is not a medical device, does not provide medical advice, and must not be used for diagnosis, treatment, or clinical decision-making.

23. License

This project is released under the MIT License.

See the LICENSE file for details.

24. Author

Saima Kauser
Amity University
Solo / Individual Participant
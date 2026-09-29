GlucoTwin

Personalized Digital Twin for 2-Hour Glucose Spike Prediction

Research Prototype

GlucoTwin is a research prototype demonstrating the fusion of synthetic historical EHR information with dynamic physiological data to estimate the probability of an upcoming glucose excursion at a 2-hour prediction horizon.

1. Participant Information

Participant: Saima Kauser
Institution: Amity University
Participation: Solo / Individual
Incubator: N/A

2. Project Overview

GlucoTwin is a personalized Digital Twin prototype designed to demonstrate how historical patient information and dynamic physiological signals can be combined to support predictive healthcare analytics.

The system creates a computational representation of a patient by combining:

Historical and static EHR information
Dynamic wearable / IoT physiological signals
Temporal and rolling features
Machine-learning based prediction
Patient-specific explainability

The prototype focuses on predicting whether glucose reaches or exceeds 180 mg/dL at the exact 2-hour prediction horizon.

All data used in this prototype is synthetic.

3. Problem Statement

Glucose levels can change significantly over time and are influenced by multiple patient-specific factors.

A single glucose measurement does not fully represent a patient's physiological state.

Traditional predictive approaches may consider isolated measurements or limited historical information. A Digital Twin approach can instead combine:

Patient demographics
Medical history
Laboratory values
Medication information
Genetic-risk indicators
Current physiological signals
Recent physiological trends

GlucoTwin demonstrates how these different data sources can be fused into a personalized computational patient state for predictive modeling.

4. Proposed Solution

GlucoTwin creates a personalized Digital Twin by combining two major layers of information.

Static / Historical EHR
Demographics
Past diagnoses
Laboratory results
Medication
Genetic risk
Dynamic Wearable / IoT Data
CGM glucose
Heart rate
HRV
Step count
Sleep stages
Activity level

These data sources are combined into a personalized Digital Twin state.

The resulting state is transformed into model-ready features and passed to an XGBoost classification model.

The model estimates the probability that glucose reaches ≥180 mg/dL at the exact 2-hour prediction horizon.

5. Digital Twin Concept

The GlucoTwin Digital Twin combines two layers.

Historical Patient Context
Age
BMI
HbA1c
Fasting glucose
Blood pressure
Diabetes duration
Past diagnosis
Medication
Genetic risk
Dynamic Physiological State
Current glucose
Heart rate
HRV
Steps
Sleep stage
Activity level
Timestamp

The Digital Twin combines these two layers to represent the patient's current computational state.

6. Prediction Target

The model predicts:

Probability that glucose reaches ≥180 mg/dL at the exact 2-hour prediction horizon.

The wearable data is sampled at 15-minute intervals.

Therefore, the 2-hour future glucose value corresponds to an 8-step temporal shift.

2 hours / 15 minutes = 8 time steps

The target is created by checking whether the glucose value at the exact 2-hour future point reaches or exceeds 180 mg/dL.

7. Synthetic Dataset

The prototype uses fully synthetic data.

Synthetic EHR
100 simulated patients
Demographics
Medical history
Laboratory values
Medication information
Genetic-risk indicators
Synthetic Wearable / IoT Data
7-day physiological timelines
15-minute sampling intervals
Glucose
Heart rate
HRV
Steps
Sleep stage
Activity

The synthetic dataset contains no real patient information.

8. Data Fusion

GlucoTwin combines historical EHR information with dynamic physiological data.

The data-fusion process consists of:

EHR → Wearable Data → Feature Engineering → Digital Twin State → AI Prediction

The EHR provides the patient's historical context, while wearable data represents the changing physiological state.

The combined representation is then used as input to the predictive model.

9. Feature Engineering

The prototype generates temporal and rolling features from the physiological data.

Features include:

Current glucose
Heart rate
HRV
Steps
Sleep stage
Activity level
Hour
Day of week
Glucose change over 15 minutes
Glucose change over 30 minutes
Glucose change over 60 minutes
1-hour glucose mean
2-hour glucose mean
1-hour glucose standard deviation
1-hour steps
2-hour steps
1-hour mean heart rate
1-hour mean HRV

The final model input contains 45 engineered features.

10. Machine Learning Model

GlucoTwin uses XGBoost for binary classification.

Model Configuration
Algorithm: XGBoost Classifier
Task: Binary classification
Number of engineered features: 45
Prediction horizon: 2 hours
Sampling interval: 15 minutes
Random state: 42

The model predicts whether the future glucose value at the defined 2-hour horizon reaches ≥180 mg/dL.

11. Model Evaluation

The current prototype evaluation produced the following results:

Metric	Result
Accuracy	90.59%
Precision	84.51%
Recall	80.66%
F1 Score	82.54%
ROC-AUC	96.69%

These metrics are based on the synthetic dataset and the current prototype evaluation methodology.

Evaluation Limitation

The current prototype uses a random row-level train-test split.

Because multiple physiological observations belong to the same patient and occur over time, this evaluation approach can allow related observations from the same patient to appear in both training and testing data.

Therefore, these results should be considered prototype-level evaluation results rather than clinical validation.

Future work should use patient-level and time-aware validation.

12. Explainability with SHAP

GlucoTwin uses SHAP TreeExplainer to provide patient-specific explanations.

SHAP helps show how individual features contributed to a specific prediction.

The dashboard displays:

Top contributing features
Positive SHAP contributions
Negative SHAP contributions
Patient-specific explanation

A positive SHAP value pushes the prediction toward a higher probability, while a negative SHAP value pushes the prediction toward a lower probability for the specific prediction being explained.

13. Digital Twin Dashboard

The Streamlit dashboard provides a conceptual healthcare-facing interface.

The dashboard includes:

Patient Digital Twin
Current Physiological State
2-Hour Glucose Prediction
SHAP Explainability
Glucose Timeline
Digital Twin Dynamics
Model Validation
Confusion Matrix
Feature Importance
ROC Curve
Data Fusion Overview
Complete Digital Twin State
Technical Summary

The dashboard is designed as a research and demonstration interface.

14. Architecture

The high-level architecture is:

Synthetic EHR Data

↓

Synthetic Wearable / IoT Data

↓

Data Fusion & Feature Engineering

↓

Personalized Digital Twin

↓

XGBoost Prediction Model

↓

2-Hour Glucose Prediction

↓

SHAP Explainability + Dashboard

The architecture document is available here:

Architecture PDF

Architecture PowerPoint

15. Technology Stack

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

16. Project Structure

GlucoTwin/

data/ — synthetic raw and processed datasets
models/ — trained model and evaluation artifacts
src/ — data generation, feature engineering, training, prediction and explainability scripts
dashboard/ — Streamlit dashboard
docs/ — architecture documentation
presentation/ — project presentation
notebooks/ — experimentation and analysis
README.md — project documentation
requirements.txt — Python dependencies
LICENSE — MIT License
17. How to Run

Clone the repository:

GitHub Repository

Install the required dependencies using:

pip install -r requirements.txt

Run the Streamlit dashboard using:

streamlit run dashboard/app.py

The dashboard will open locally in the browser.

18. Reproducibility

The repository contains the source code, synthetic datasets, trained model artifacts, dashboard, documentation and required dependencies.

The project can be reproduced using the provided requirements file and project structure.

The prototype does not require access to real patient data.

19. Privacy and Data Safety

GlucoTwin uses synthetic data for demonstration.

No real patient records are included in the repository.

The prototype is designed to demonstrate the technical concept without exposing personal healthcare information.

20. Limitations

The current prototype has several limitations:

The dataset is synthetic.
The physiological signals are simulated.
The current evaluation uses a random row-level split.
The model has not been clinically validated.
The prediction has not been externally validated.
The dashboard is a conceptual research interface.
The model should not be used for clinical decision-making.

Future development would require appropriately governed real-world research datasets, patient-level and time-aware validation, external validation, calibration, uncertainty estimation and clinical evaluation.

21. Future Work

Potential future improvements include:

Patient-level and time-aware validation
Larger longitudinal datasets
Real-world appropriately governed research data
Additional wearable signals
Continuous glucose monitoring integration
Improved uncertainty estimation
Model calibration
Longitudinal Digital Twin updates
External validation
More advanced personalized temporal models
Clinical research collaboration


22. Research Prototype Disclaimer

GlucoTwin is a proof-of-concept Digital Twin developed using synthetic/anonymized data.

Model outputs are intended only for research, demonstration and educational purposes.

This prototype is not a medical device and should not be used for clinical diagnosis, treatment or medical decision-making.

23. Project Demo Video

GlucoTwin — Personalized Digital Twin for 2-Hour Glucose Spike Prediction

Watch the GlucoTwin Prototype Demo :- https://youtu.be/pxY6XqOuaw0

The video demonstrates the working prototype, including the Digital Twin dashboard, patient state, dynamic physiological data, 2-hour prediction, SHAP explainability, glucose timeline, model evaluation and technical pipeline.

24. Required Submission Documents
Project Repository

Public GitHub Repository

Architecture Document

Architecture PDF

Architecture PowerPoint

Project Presentation

GlucoTwin Pitch Deck

Demo Video

GlucoTwin Prototype Demo

25. License

This project is released under the MIT License.

See the LICENSE file for details.

26. Author

Saima Kauser
Amity University

Project: GlucoTwin
Challenge: Digital Twin Challenge 2026
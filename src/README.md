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

GlucoTwin addresses this problem by creating a computational Digital Twin that continuously combines static patient context with dynamic physiological state.

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
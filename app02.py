#``python
import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import json

# =====================================================
# LOAD SCALER
# =====================================================

with open("model2/scaler2.pkl", "rb") as f:
    scaler = pickle.load(f)

# =====================================================
# LOAD FEATURES
# =====================================================

with open("model2/features2.json", "r") as f:
    feature_names = json.load(f)

# =====================================================
# LOAD MODEL
# =====================================================

model = xgb.Booster()
model.load_model("model2/sepsis_model2.json")

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("Sepsis Risk Prediction")

st.write(
    "Enter patient information to estimate the probability of sepsis."
)

HR = st.number_input(
    "Heart Rate (bpm)",
    30.0, 220.0, 90.0
)

O2Sat = st.number_input(
    "Oxygen Saturation (%)",
    60.0, 100.0, 98.0
)

Temp = st.number_input(
    "Temperature (°C)",
    34.0, 42.0, 37.0
)

MAP = st.number_input(
    "Mean Arterial Pressure",
    30.0, 150.0, 80.0
)

Resp = st.number_input(
    "Respiratory Rate",
    5.0, 50.0, 18.0
)

BUN = st.number_input(
    "BUN",
    0.0, 100.0, 15.0
)

Chloride = st.number_input(
    "Chloride",
    70.0, 130.0, 100.0
)

Creatinine = st.number_input(
    "Creatinine",
    0.1, 10.0, 1.0
)

Glucose = st.number_input(
    "Glucose",
    40.0, 400.0, 100.0
)

Hct = st.number_input(
    "Hematocrit",
    10.0, 60.0, 40.0
)

Hgb = st.number_input(
    "Hemoglobin",
    5.0, 25.0, 13.0
)

WBC = st.number_input(
    "WBC",
    0.0, 30.0, 8.0
)

Platelets = st.number_input(
    "Platelets",
    20.0, 600.0, 250.0
)

Age = st.number_input(
    "Age",
    18.0, 100.0, 60.0
)

HospAdmTime = st.number_input(
    "Hospital Admission Time",
    -100.0, 25.0, -1.0
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

gender_0 = 1 if gender == "Female" else 0
gender_1 = 1 if gender == "Male" else 0

# =====================================================
# CREATE PATIENT DATAFRAME
# =====================================================

patient = pd.DataFrame([{
    "HR": HR,
    "O2Sat": O2Sat,
    "Temp": Temp,
    "MAP": MAP,
    "Resp": Resp,
    "BUN": BUN,
    "Chloride": Chloride,
    "Creatinine": Creatinine,
    "Glucose": Glucose,
    "Hct": Hct,
    "Hgb": Hgb,
    "WBC": WBC,
    "Platelets": Platelets,
    "Age": Age,
    "HospAdmTime": HospAdmTime,
    "0": gender_0,
    "1": gender_1
}])

# =====================================================
# SCALE FEATURES
# =====================================================

scaled_columns = [
    "HR",
    "O2Sat",
    "Temp",
    "MAP",
    "Resp",
    "BUN",
    "Chloride",
    "Creatinine",
    "Glucose",
    "Hct",
    "Hgb",
    "WBC",
    "Platelets"
]

patient[scaled_columns] = scaler.transform(
    patient[scaled_columns]
)

# =====================================================
# MATCH TRAINING ORDER
# =====================================================

patient = patient[feature_names]

# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict Risk"):

    dpatient = xgb.DMatrix(patient)

    risk = model.predict(dpatient)[0]

    st.subheader(
        f"Sepsis Risk: {risk*100:.1f}%"
    )

    if risk < 0.20:
        st.success("Low Risk")

    elif risk < 0.50:
        st.warning("Moderate Risk")

    elif risk < 0.75:
        st.warning("High Risk")

    else:
        st.error("Very High Risk")

    st.write("Probability:", risk)

    st.write("Processed Input:")

    st.dataframe(patient)
#```

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load('xgboost_rpl_model.pkl')

# Title
st.title("Recurrent Pregnancy Loss (RPL) Risk Predictor")

# Input form
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 18, 45, 30)
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 26.0)
previous_losses = st.sidebar.slider("Previous Losses", 0, 5, 0)
tsh = st.sidebar.slider("TSH Level", 0.1, 10.0, 2.5)

aps = st.sidebar.selectbox("Antiphospholipid Syndrome (APS)", ["No", "Yes"])
uterine = st.sidebar.selectbox("Uterine Abnormality", ["No", "Yes"])
genetic = st.sidebar.selectbox("Genetic Translocation", ["No", "Yes"])
pcos = st.sidebar.selectbox("PCOS", ["No", "Yes"])
aab = st.sidebar.selectbox("Antiphospholipid Antibodies", ["Negative", "Positive"])
fam_history = st.sidebar.selectbox("Family History of RPL", ["No", "Yes"])
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])

# Convert input to numeric format for prediction
input_data = pd.DataFrame([{
    'Age': age,
    'BMI': bmi,
    'Previous_Losses': previous_losses,
    'TSH_Level': tsh,
    'APS': 1 if aps == "Yes" else 0,
    'Uterine_Abnormality': 1 if uterine == "Yes" else 0,
    'Genetic_Translocation': 1 if genetic == "Yes" else 0,
    'PCOS': 1 if pcos == "Yes" else 0,
    'Antiphospholipid_Antibodies': 1 if aab == "Positive" else 0,
    'Family_History_RPL': 1 if fam_history == "Yes" else 0,
    'Smoking': 1 if smoking == "Yes" else 0
}])

# Prediction
if st.button("Predict RPL Risk"):
    prediction = model.predict(input_data)[0]
    pred_label = "Likely RPL" if prediction == 1 else "Unlikely RPL"
    st.subheader(f"🩺 Prediction: {pred_label}")

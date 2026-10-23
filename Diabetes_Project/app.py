import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_regression_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

st.title("Diabetes Prediction App")
st.write("Enter the health measurements below to get a prediction from the trained Logistic Regression model.")
st.warning("This app is for educational purposes only and is not a medical diagnosis.")

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.error("Model files are missing. Add logistic_regression_model.pkl and scaler.pkl to this project folder.")
    st.stop()

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

model, scaler = load_artifacts()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
    with col2:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=FEATURES)
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_input)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])

    if prediction == 1:
        st.error("Prediction: Diabetes indicated by the model")
    else:
        st.success("Prediction: Diabetes not indicated by the model")

    if probability is not None:
        st.metric("Estimated model probability for class 1", f"{probability:.1%}")
    st.caption("Predictions depend on the training dataset and model. Consult a qualified healthcare professional for medical advice.")

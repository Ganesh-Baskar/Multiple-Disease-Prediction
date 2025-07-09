import streamlit as st
import numpy as np
import pickle

# Load models
def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)

parkinson_model = load_model("parkinson_model.pkl")
liver_model = load_model("liver_model.pkl")
kidney_model = load_model("kidney_model.pkl")

# Prediction function
def predict_disease(model, input_data):
    data = np.array(input_data).reshape(1, -1)
    return model.predict(data)[0]

# UI
st.set_page_config(page_title="🧠 Disease Predictor", layout="centered")
st.title("🧬 Multi-Disease Prediction App")
st.markdown("Predict **Parkinson's**, **Liver**, and **Kidney** conditions using ML models.")

st.sidebar.title("🔍 Select Disease to Predict")
disease = st.sidebar.selectbox("Choose a disease", ["Parkinson’s disease", "Liver disease", "Kidney disease"])

# 🔹 Parkinson’s UI
if disease == "Parkinson’s disease":
    st.subheader("🧠 Parkinson's Disease Parameters")
    with st.expander("Fill the values below 👇"):
        inputs = [
            st.number_input("Fo (Hz)"), st.number_input("Fhi (Hz)"), st.number_input("Flo (Hz)"),
            st.number_input("Jitter (%)"), st.number_input("Jitter (Abs)"), st.number_input("RAP"),
            st.number_input("PPQ"), st.number_input("Jitter DDP"), st.number_input("Shimmer"),
            st.number_input("Shimmer dB"), st.number_input("Shimmer APQ3"), st.number_input("Shimmer APQ5"),
            st.number_input("APQ"), st.number_input("Shimmer DDA"), st.number_input("NHR"),
            st.number_input("HNR"), st.number_input("RPDE"), st.number_input("DFA"),
            st.number_input("Spread1"), st.number_input("Spread2"), st.number_input("D2"), st.number_input("PPE")
        ]
    if st.button("🧪 Predict Parkinson's"):
        prediction = predict_disease(parkinson_model, inputs)
        st.success("✅ You are healthy!") if prediction == 0 else st.error("⚠️ Parkinson’s detected. Please consult a specialist.")

# 🔹 Liver UI
elif disease == "Liver disease":
    st.subheader("🧫 Liver Health Check")
    with st.expander("Enter details:"):
        inputs = [
            st.number_input("Age"), st.number_input("Total Bilirubin"),
            st.number_input("Direct Bilirubin"), st.number_input("Alkaline Phosphotase"),
            st.number_input("Alamine Aminotransferase"), st.number_input("Aspartate Aminotransferase"),
            st.number_input("Total Proteins"), st.number_input("Albumin"),
            st.number_input("Albumin and Globulin Ratio"), st.number_input("Gender (Male=1, Female=0)")
        ]
    if st.button("🧪 Predict Liver Condition"):
        prediction = predict_disease(liver_model, inputs)
        st.success("✅ Liver is healthy!") if prediction == 0 else st.error("⚠️ Liver condition detected. Medical attention advised.")

# 🔹 Kidney UI
elif disease == "Kidney disease":
    st.subheader("🩺 Kidney Function Evaluation")

    st.markdown("### Enter Numeric Values:")
    num_inputs = [st.number_input(label) for label in [
        "Age", "Blood Pressure (mmHg)", "Specific Gravity", "Albumin", "Sugar",
        "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium", "Potassium",
        "Hemoglobin", "Packed Cell Volume", "White Blood Cell Count", "Red Blood Cell Count"
    ]]

    st.markdown("### Choose Categorical Options:")
    options = {
        "Red Blood Cells": ["normal", "abnormal"],
        "Pus Cell": ["normal", "abnormal"],
        "Pus Cell Clumps": ["notpresent", "present"],
        "Bacteria": ["notpresent", "present"],
        "Hypertension": ["no", "yes"],
        "Diabetes Mellitus": ["no", "yes"],
        "Coronary Artery Disease": ["no", "yes"],
        "Appetite": ["poor", "good"],
        "Pedal Edema": ["no", "yes"],
        "Anemia": ["no", "yes"]
    }
    mapping = {"normal": 0, "abnormal": 1, "notpresent": 0, "present": 1, "no": 0, "yes": 1, "poor": 0, "good": 1}
    cat_inputs = [mapping[st.selectbox(label, choices)] for label, choices in options.items()]

    if st.button("🧪 Predict Kidney Health"):
        prediction = predict_disease(kidney_model, num_inputs + cat_inputs)
        st.success("✅ Kidney is functioning well!") if prediction == 0 else st.error("⚠️ Possible kidney disease. Please consult a doctor.")
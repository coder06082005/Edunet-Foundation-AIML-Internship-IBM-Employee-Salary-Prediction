# app.py

import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Page configuration
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

# Title Section
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: 700;
            color: #4A4A4A;
        }
        .sub-title {
            font-size: 18px;
            color: #6c757d;
        }
    </style>
    <div class="main-title">💼 Employee Salary Prediction</div>
    <div class="sub-title">Predict if a person earns more than 50K/year based on their details</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    st.header("📋 Personal Details")
    age = st.slider("Age", 18, 90, 30)
    gender = st.selectbox("Gender", label_encoders['gender'].classes_)
    race = st.selectbox("Race", label_encoders['race'].classes_)
    marital_status = st.selectbox("Marital Status", label_encoders['marital-status'].classes_)
    relationship = st.selectbox("Relationship", label_encoders['relationship'].classes_)

with col2:
    st.header("🏢 Employment Info")
    workclass = st.selectbox("Workclass", label_encoders['workclass'].classes_)
    education = st.selectbox("Education Level", label_encoders['education'].classes_)
    educational_num = st.slider("Education Number", 1, 16, 9)
    occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_)
    native_country = st.selectbox("Native Country", label_encoders['native-country'].classes_)

st.markdown("---")
st.header("💰 Financial Information")

# Financial section
col3, col4, col5 = st.columns(3)
with col3:
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 150000)

with col4:
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)

with col5:
    capital_loss = st.number_input("Capital Loss", 0, 5000, 0)

hours_per_week = st.slider("Hours Worked per Week", 1, 100, 40)

# Predict button
if st.button("🚀 Predict Salary Class"):
    input_data = [
        age,
        label_encoders['workclass'].transform([workclass])[0],
        fnlwgt,
        label_encoders['education'].transform([education])[0],
        educational_num,
        label_encoders['marital-status'].transform([marital_status])[0],
        label_encoders['occupation'].transform([occupation])[0],
        label_encoders['relationship'].transform([relationship])[0],
        label_encoders['race'].transform([race])[0],
        label_encoders['gender'].transform([gender])[0],
        capital_gain,
        capital_loss,
        hours_per_week,
        label_encoders['native-country'].transform([native_country])[0]
    ]

    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.success("✅ Predicted Income: >50K")
    else:
        st.warning("❌ Predicted Income: <=50K")

    if probability:
        st.info(f"🔎 Confidence: {probability * 100:.2f}%")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by [Badavath Tharun] | Powered by KNN Model", unsafe_allow_html=True)

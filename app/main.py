import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

# Load the saved model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("Model files not found. Please run 'src/model.py' first.")
    st.stop()

# App Title and Description
st.title("🏥 Health Insurance Premium Estimator")
st.write("Enter your details below to get an estimated annual insurance cost.")

# Input Form
with st.form("insurance_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", options=["male", "female"])
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
    
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Do you smoke?", options=["yes", "no"])
        region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])
    
    submit_button = st.form_submit_button("Calculate Quote")

if submit_button:
    # 1. Prepare data for prediction (Same steps as in data_prep.py)
    input_data = {
        'age': age,
        'sex': 1 if sex == "male" else 0,
        'bmi': bmi,
        'children': children,
        'smoker': 1 if smoker == "yes" else 0,
        'bmi_smoker': bmi * (1 if smoker == "yes" else 0),
        'region_northwest': 1 if region == "northwest" else 0,
        'region_southeast': 1 if region == "southeast" else 0,
        'region_southwest': 1 if region == "southwest" else 0
    }
    
    # Create DataFrame (Ensure the order of columns matches the training data)
    input_df = pd.DataFrame([input_data])
    
    # 2. Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # 3. Predict
    prediction = model.predict(input_scaled)[0]
    
    # 4. Display result
    st.success(f"### Estimated Annual Premium: ${prediction:,.2f}")
    
    # Add a disclaimer
    st.info("Note: This is a machine learning estimate based on historical data and not a final insurance quote.")

    # Business Insight
    if smoker == "yes":
        st.warning("💡 Tip: Quitting smoking could significantly reduce your annual premium based on our analysis.")
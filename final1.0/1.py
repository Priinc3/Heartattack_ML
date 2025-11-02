import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Resolve paths relative to this script so it works on Streamlit Cloud
BASE_DIR = Path(__file__).parent

# Load the trained model and preprocessor (from the same folder as this app)
try:
    model = joblib.load(BASE_DIR / "LRM.joblib")
except Exception as e:
    st.error(f"Failed to load model LRM.joblib: {e}")
    st.stop()

try:
    preprocessor = joblib.load(BASE_DIR / "preprocessor.joblib")
except FileNotFoundError:
    st.error("Preprocessor not found! Please save the preprocessor from your training notebook.")
    st.stop()

st.title("❤️ Heart Disease Prediction App")
st.write("Enter your details below:")

# Numerical features
bmi = st.number_input("BMI")
physical_health = st.number_input("Physical Health", 0, 30)
mental_health = st.number_input("Mental Health", 0, 30)
sleep_time = st.number_input("Sleep Time (hours)", 0, 24)

# Binary features
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol_drinking = st.selectbox("Alcohol Drinking", ["No", "Yes"])
stroke = st.selectbox("Had Stroke", ["No", "Yes"])
diff_walking = st.selectbox("Difficulty Walking", ["No", "Yes"])
sex = st.selectbox("Sex", ["Female", "Male"])
physical_activity = st.selectbox("Physical Activity", ["No", "Yes"])
asthma = st.selectbox("Asthma", ["No", "Yes"])
kidney_disease = st.selectbox("Kidney Disease", ["No", "Yes"])
skin_cancer = st.selectbox("Skin Cancer", ["No", "Yes"])

# Categorical features (ordinal and nominal)
age_category = st.selectbox("Age Category", 
    ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
     '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'])
race = st.selectbox("Race", 
    ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 
     'Other', 'Hispanic'])
diabetic = st.selectbox("Diabetic", ['No', 'Yes', 'Borderline'])
gen_health = st.selectbox("General Health", 
    ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])

if st.button("Predict"):
    # Create input dataframe with all features in the correct order
    input_dict = {
        'BMI': bmi,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'SleepTime': sleep_time,
        'Smoking': smoking,
        'AlcoholDrinking': alcohol_drinking,
        'Stroke': stroke,
        'DiffWalking': diff_walking,
        'Sex': sex,
        'AgeCategory': age_category,
        'Race': race,
        'Diabetic': diabetic,
        'GenHealth': gen_health,
        'PhysicalActivity': physical_activity,
        'Asthma': asthma,
        'KidneyDisease': kidney_disease,
        'SkinCancer': skin_cancer
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Apply preprocessing
    input_transformed = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_transformed)[0]
    prediction_proba = model.predict_proba(input_transformed)[0]
    
    if prediction == 1:
        st.error(f"💔 High risk of Heart Disease (Probability: {prediction_proba[1]:.2%})")
    else:
        st.success(f"❤️ Low risk of Heart Disease (Probability: {prediction_proba[0]:.2%})")
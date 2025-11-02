import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Resolve paths relative to this script so it works on Streamlit Cloud
BASE_DIR = Path(__file__).parent
EXPORTED_MODELS_DIR = BASE_DIR / "exported_models"

# Load all three models from exported_models directory (now in final1.0 folder)
models = {}
model_names = ["RandomForest", "GradientBoosting", "LogisticRegression"]

for model_name in model_names:
    model_path = EXPORTED_MODELS_DIR / model_name / "model.joblib"
    try:
        models[model_name] = joblib.load(model_path)
        st.sidebar.success(f"✅ Loaded {model_name}")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load {model_name}: {e}")

if not models:
    st.error("No models could be loaded! Please ensure exported_models directory exists with model files.")
    st.stop()

# Load preprocessor (using the one from RandomForest directory, they're all the same)
try:
    preprocessor = joblib.load(EXPORTED_MODELS_DIR / "RandomForest" / "preprocessor.joblib")
except FileNotFoundError:
    # Fallback to the local preprocessor if exported_models not available
    try:
        preprocessor = joblib.load(BASE_DIR / "preprocessor.joblib")
    except FileNotFoundError:
        st.error("Preprocessor not found! Please save the preprocessor from your training notebook.")
        st.stop()

st.title("❤️ Heart Disease Prediction App")
st.write("### Multi-Model Ensemble Prediction System")
st.write("Get predictions from three different machine learning models:")
st.info("🔹 **Random Forest** • 🔹 **Gradient Boosting** • 🔹 **Logistic Regression**")
st.write("---")
st.write("**Enter your health details below:**")

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

if st.button("🔮 Predict with All Models"):
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
    
    # Display results from all three models
    st.write("---")
    st.subheader("🤖 Predictions from All Models")
    
    predictions_summary = []
    
    for model_name, model in models.items():
        # Make prediction
        prediction = model.predict(input_transformed)[0]
        prediction_proba = model.predict_proba(input_transformed)[0]
        
        # Store for summary
        predictions_summary.append({
            'Model': model_name,
            'Prediction': 'High Risk ⚠️' if prediction == 1 else 'Low Risk ✅',
            'Risk Probability': f"{prediction_proba[1]:.2%}",
            'No Risk Probability': f"{prediction_proba[0]:.2%}"
        })
        
        # Display individual model prediction
        with st.container():
            st.write(f"### 🔹 {model_name}")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(f"**Prediction:** 💔 High Risk of Heart Disease")
                else:
                    st.success(f"**Prediction:** ❤️ Low Risk of Heart Disease")
            
            with col2:
                st.metric("Heart Disease Risk", f"{prediction_proba[1]:.2%}")
            
            st.progress(float(prediction_proba[1]))
            st.write("---")
    
    # Summary table
    st.subheader("📊 Predictions Summary")
    summary_df = pd.DataFrame(predictions_summary)
    st.dataframe(summary_df, use_container_width=True)
    
    # Consensus
    high_risk_count = sum([1 for p in predictions_summary if 'High Risk' in p['Prediction']])
    st.write("---")
    st.subheader("🎯 Consensus")
    if high_risk_count == 3:
        st.error("⚠️ **All three models predict HIGH RISK** - Please consult a healthcare professional!")
    elif high_risk_count == 2:
        st.warning("⚠️ **Two out of three models predict HIGH RISK** - Consider medical consultation.")
    elif high_risk_count == 1:
        st.info("ℹ️ **One model predicts HIGH RISK** - Results are mixed. Monitor your health.")
    else:
        st.success("✅ **All models predict LOW RISK** - Keep maintaining a healthy lifestyle!")
# Heart Disease Prediction - Multi-Model Ensemble App

## Overview
This Streamlit app uses **three machine learning models** to predict heart disease risk:
- 🌲 **Random Forest**
- 📈 **Gradient Boosting**
- 📊 **Logistic Regression**

Each model provides its own prediction, and the app shows a consensus view.

## Running the App

### Prerequisites
Make sure you have all the required models exported. Run the export cell in the notebook first:
```bash
cd draft
jupyter notebook heart_disease_Chatgpt_downsample.ipynb
# Run the export cell (last cell in the notebook)
```

### Launch the App
```bash
cd final1.0
streamlit run 1.py
```

## Features

### 1. **Individual Model Predictions**
Each model displays:
- Prediction (High Risk or Low Risk)
- Probability percentage
- Visual progress bar

### 2. **Summary Table**
Comparative view of all three model predictions with probabilities

### 3. **Consensus Decision**
- ✅ All models agree: Low Risk
- ℹ️ One model disagrees: Mixed results
- ⚠️ Two models agree: High Risk (warning)
- 🚨 All models agree: High Risk (urgent)

### 4. **Sidebar Status**
Shows which models loaded successfully

## Directory Structure
```
final1.0/
├── 1.py                    # Streamlit app
├── preprocessor.joblib     # Fallback preprocessor
└── LRM.joblib             # Legacy model (not used anymore)

draft/exported_models/
├── RandomForest/
│   ├── model.joblib
│   └── preprocessor.joblib
├── GradientBoosting/
│   ├── model.joblib
│   └── preprocessor.joblib
└── LogisticRegression/
    ├── model.joblib
    └── preprocessor.joblib
```

## How It Works

1. **Model Loading**: All three models are loaded from `../draft/exported_models/`
2. **Input Processing**: User inputs are preprocessed using the same pipeline from training
3. **Prediction**: Each model makes independent predictions
4. **Ensemble View**: Results are aggregated and presented with consensus

## Deployment Notes

For Streamlit Cloud deployment, ensure:
- All model files are committed to the repository
- The `exported_models` directory is in the correct relative path
- Dependencies are listed in `requirements.txt`

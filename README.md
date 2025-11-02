# Heart Disease Prediction — Streamlit App

This repository contains notebooks for data prep/modeling and a Streamlit app in `final1.0/` for inference.

## Structure
- `draft/` — notebooks used to prepare data and train models
- `final1.0/1.py` — Streamlit app entry point
- `final1.0/LRM.joblib` — Trained Logistic Regression model
- `final1.0/preprocessor.joblib` — Fitted preprocessing pipeline (ColumnTransformer)
- `requirements.txt` — Python dependencies for deployment

## Run locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run final1.0/1.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io and create a new app.
3. Select your repo, branch (e.g. `main`), and set the app path to:
   - `final1.0/1.py`
4. Click Deploy.

Notes
- The app loads `LRM.joblib` and `preprocessor.joblib` from the `final1.0/` folder.
- If you retrain, re-run the notebook cell that saves these artifacts and commit the updated files.
- Ensure your `requirements.txt` matches the libraries used for inference.

# Diabetes Prediction — Streamlit Deployment

This project adds a Streamlit interface to the Logistic Regression model in your notebook.

## Project files
- `app.py` — Streamlit web interface
- `requirements.txt` — Python packages
- `logistic_regression_model.pkl` — trained model exported from your notebook (you must add this)
- `scaler.pkl` — fitted StandardScaler exported from your notebook (you must add this)

## 1. Export the model files
Run the notebook cells that train `logistic_model` and fit `scaler` on `X_train`. Then run:

```python
import joblib
joblib.dump(logistic_model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

Copy both `.pkl` files into the same folder as `app.py`. The scaler must be the one fitted on `X_train` and used to transform the test data.

## 2. Install and run locally
Open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser and test the form.

## 3. Deploy with Streamlit Community Cloud
1. Create a GitHub repository, for example `diabetes-prediction-streamlit`.
2. Upload `app.py`, `requirements.txt`, `logistic_regression_model.pkl`, and `scaler.pkl`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. Deploy, wait for the build to finish, and open the generated app URL.
6. Test the deployed app and submit that URL with your notebook.

## Important
- The notebook loads `diabetes.csv`; keep that file in the notebook's working directory when running the training cells.
- Do not upload private or sensitive datasets to a public repository.
- The app expects these feature names in this exact order: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, matching the standard Pima diabetes dataset and the notebook's `Outcome` target.
- This is an educational model, not a medical diagnostic tool.

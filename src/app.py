import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model + scaler + feature names
model = joblib.load("xgb_bq.pkl")
scaler = joblib.load("scaler_bq.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Patient Readmission Predictor", layout="wide")
st.title("🏥 Predict Patient Readmission Risk")
st.markdown("Upload patient data or fill the form below to get predictions.")

# --- Sample form input ---
with st.form("patient_form"):
    st.subheader("📋 Enter Patient Info")

    age = st.slider("Age", 0, 100, 60)
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 30, 5)
    num_lab_procedures = st.slider("Number of Lab Procedures", 0, 100, 40)
    num_medications = st.slider("Number of Medications", 1, 50, 10)
    number_inpatient = st.slider("Previous Inpatient Visits", 0, 20, 0)
    number_emergency = st.slider("Emergency Visits", 0, 10, 0)
    number_outpatient = st.slider("Outpatient Visits", 0, 20, 0)
    on_insulin = st.selectbox("On Insulin?", ["Yes", "No"])
    meds_changed = st.selectbox("Medications Changed?", ["Yes", "No"])
    race_Caucasian = st.selectbox("Is Race Caucasian?", ["Yes", "No"])
    
    submit = st.form_submit_button("Predict")

# --- Build feature vector if form submitted ---
if submit:
    st.subheader("🔍 Prediction")

    total_prev_visits = number_outpatient + number_emergency + number_inpatient
    is_frequent_user = 1 if total_prev_visits > 5 else 0

    input_dict = {
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_medications': num_medications,
        'number_inpatient': number_inpatient,
        'number_emergency': number_emergency,
        'number_outpatient': number_outpatient,
        'on_insulin': 1 if on_insulin == "Yes" else 0,
        'meds_changed': 1 if meds_changed == "Yes" else 0,
        'age_midpoint': age,
        'total_prev_visits': total_prev_visits,
        'is_frequent_user': is_frequent_user,
        'race_Caucasian': 1 if race_Caucasian == "Yes" else 0
    }

    # Convert to DataFrame and align columns
    df_input = pd.DataFrame([input_dict])
    df_input = df_input.reindex(columns=features, fill_value=0)
    df_scaled = scaler.transform(df_input)

    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    label = "🔁 Likely Readmitted" if pred == 1 else "✅ Not Likely to be Readmitted"
    st.success(f"**Prediction:** {label}")
    st.metric("Readmission Probability", f"{prob:.2%}")

# --- Batch CSV Prediction ---
st.divider()
st.subheader("📂 Upload CSV for Batch Prediction")

csv_file = st.file_uploader("Upload a CSV with same columns as model input", type=["csv"])
if csv_file:
    df_csv = pd.read_csv(csv_file)
    df_csv = df_csv.reindex(columns=features, fill_value=0)
    df_csv = df_csv.astype('float32')
    df_csv_scaled = scaler.transform(df_csv)

    preds = model.predict(df_csv_scaled)
    probs = model.predict_proba(df_csv_scaled)[:, 1]

    results = df_csv.copy()
    results["readmission_pred"] = preds
    results["readmission_proba"] = probs

    st.write(results.head())
    st.download_button("Download Results CSV", results.to_csv(index=False), "predictions.csv", "text/csv")

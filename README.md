# 🏥 Patient Readmission Prediction

This project builds and evaluates machine learning models to predict **30-day hospital readmissions** using a real-world diabetes dataset. The pipeline covers data preprocessing, feature engineering, model training (with SMOTE to address class imbalance), threshold tuning, GCP BigQuery integration, and deployment via a Streamlit app.

---

## 📌 Dataset

- **Source:** [UCI Diabetes 130-US Hospitals Dataset (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Records:** ~100,000 hospital admissions
- **Target:** Whether the patient was readmitted within 30 days
- **Features:** Demographics, diagnoses, medications, lab results, and visit types

---

## ☁️ GCP Integration

This project uses **Google Cloud Platform (GCP)** for:

- Hosting and querying preprocessed healthcare data via **BigQuery**
- Training models directly from BigQuery using the `google-cloud-bigquery` Python SDK
- Supporting reproducibility, scalability, and cloud-hosted data workflows

📁 Script: `src/train_models_from_bigquery.py`

> Requires authentication via GCP service account and access to your BigQuery project.

---

## 🧠 ML Pipeline Overview

- **Data Cleaning & Encoding**
- **Feature Engineering** (custom features like total visits, insulin usage, etc.)
- **Modeling** with:
  - Logistic Regression
  - MLP Classifier
  - Random Forest / Gradient Boosting
  - XGBoost (with and without feature selection)
- **Imbalance Handling** via SMOTE
- **Threshold Tuning** using ROC and Precision-Recall curves
- **Deployment** via a real-time prediction app (Streamlit)

---

## 🧩 Project Structure

```text
healthcare-readmission/
├── data/                          # CSVs, cleaned data, pickled models (excluded from GitHub)
├── models/                        # Saved trained models (.pkl)
├── notebooks/                     # Jupyter notebooks for exploration
├── src/                           # ML and app scripts
│   ├── 1_load_and_explore.py
│   ├── 2_clean_preprocess.py
│   ├── 2.5_feature_selection.py
│   ├── 2.6_feature_engineering.py
│   ├── 3_train_model.py
│   ├── 3_train_model_smote.py
│   ├── 3_train_logistic_mlp_smote.py
│   ├── 3_train_compare_models_engineered.py
│   ├── 3_train_xgboost_smote.py
│   ├── 3_train_xgboost_reduced.py
│   ├── 4_threshold_tuning.py
│   ├── train_models_from_bigquery.py   # 📡 Trains on BigQuery-hosted data
│   └── app.py                          # Streamlit app for user prediction
├── requirements.txt
├── Figure_1.png
├── healthcare-env/                # Local virtual environment (ignored)
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/varad-suryavanshi/Patient-Readmission-Prediction.git
cd Patient-Readmission-Prediction
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare Data (Local Version)

Place `diabetic_data.csv` and `IDS_mapping.csv` inside the `data/` folder.

Run:

```bash
python src/2_clean_preprocess.py
python src/2.6_feature_engineering.py
```

### 4. Train Models (Locally)

```bash
python src/3_train_model_smote.py
python src/3_train_xgboost_smote.py
python src/3_train_logistic_mlp_smote.py
```

### 5. Train from BigQuery (GCP)

Make sure your Google Cloud credentials are configured:

```bash
python src/train_models_from_bigquery.py
```

> This script connects to BigQuery, loads preprocessed feature tables, trains models, and saves them locally.

### 6. Tune Threshold

```bash
python src/4_threshold_tuning.py
```

### 7. Launch Streamlit App

```bash
streamlit run src/app.py
```

---

## 📊 App Features

The Streamlit app supports:
- Manual entry of patient features for individual predictions
- Batch predictions using CSV uploads
- Display of readmission probability and predicted class

---

## 🔮 Future Enhancements

- Deploy Streamlit app to **Streamlit Cloud** or **Hugging Face Spaces**
- Add real-time BigQuery prediction support
- Use Vertex AI for model hosting (GCP)
- Build explainability layer with SHAP or LIME
- Implement cost-based thresholds and patient risk scoring

---

## 📄 License

This project is licensed under the **MIT License**.  
Original dataset available from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

---

Made with 💡 by [Varad Suryavanshi](https://github.com/varad-suryavanshi)  
Integrating cloud-scale data workflows with predictive healthcare modeling.

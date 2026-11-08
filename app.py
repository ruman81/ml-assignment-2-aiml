import os
import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from utils.metrics import MODEL_FILES, compute_metrics, load_model

# ---------------------------------------------------------------------------
# Resolve repo root so paths work both locally and on Streamlit Cloud
# ---------------------------------------------------------------------------
REPO_ROOT = str(pathlib.Path(__file__).resolve().parent)
TARGET_COLUMN = 'target'
CLASS_NAMES = ['Malignant', 'Benign']   # 0 = Malignant, 1 = Benign
EXPECTED_N_FEATURES = 30

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Classification Dashboard",
    layout="wide",
)

st.title("ML Assignment 2 — Classification Model Comparison")
st.markdown(
    "Upload test data, select a model, and view evaluation metrics "
    "and confusion matrix. **Models are pre-trained — no retraining occurs.**"
)

# ---------------------------------------------------------------------------
# Sidebar — CSV Upload (R3.1) and Model Selection (R3.2)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    uploaded_file = st.file_uploader(
        "Upload test CSV (must include all features + target column)",
        type="csv",
        help="Upload test_data.csv from the GitHub repo",
    )

    model_name = st.selectbox(
        "Select Classification Model",
        options=list(MODEL_FILES.keys()),
    )

# ---------------------------------------------------------------------------
# Main area — process uploaded data
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validate target column
    if TARGET_COLUMN not in df.columns:
        st.error(f"Uploaded CSV is missing the target column: '{TARGET_COLUMN}'")
        st.stop()

    X_raw = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Validate feature count
    if X_raw.shape[1] != EXPECTED_N_FEATURES:
        st.error(
            f"Expected {EXPECTED_N_FEATURES} feature columns but found "
            f"{X_raw.shape[1]}. Please upload the unmodified test_data.csv."
        )
        st.stop()

    # Load fitted preprocessor and transform (NO fit — data-leakage safe)
    preprocessor = joblib.load(os.path.join(REPO_ROOT, 'model', 'preprocessor.pkl'))
    X_scaled = preprocessor.transform(X_raw)

    # Load selected model (NO training)
    model = load_model(model_name, repo_root=REPO_ROOT)

    # Compute metrics and predictions
    metrics = compute_metrics(model, X_scaled, y)
    y_pred = model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=CLASS_NAMES, output_dict=True)

    # -------------------------------------------------------------------
    # R3.3 — Evaluation Metrics
    # -------------------------------------------------------------------
    st.subheader(f"Evaluation Metrics — {model_name}")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col1.metric("Accuracy",  f"{metrics['Accuracy']:.4f}")
    col2.metric("AUC",       f"{metrics['AUC']:.4f}")
    col3.metric("Precision", f"{metrics['Precision']:.4f}")
    col4.metric("Recall",    f"{metrics['Recall']:.4f}")
    col5.metric("F1 Score",  f"{metrics['F1']:.4f}")
    col6.metric("MCC",       f"{metrics['MCC']:.4f}")

    metrics_df = pd.DataFrame([metrics])
    st.table(metrics_df.style.format("{:.4f}"))

    # -------------------------------------------------------------------
    # R3.4 — Confusion Matrix & Classification Report
    # -------------------------------------------------------------------
    tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])

    with tab1:
        st.subheader(f"Confusion Matrix — {model_name}")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred: Malignant', 'Pred: Benign'],
            yticklabels=['True: Malignant', 'True: Benign'],
        )
        ax.set_title(f"Confusion Matrix — {model_name}")
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader(f"Classification Report — {model_name}")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}", na_rep="—"))

    # -------------------------------------------------------------------
    # R3.6 — All-Models Comparison
    # -------------------------------------------------------------------
    st.divider()
    st.subheader("All Models — Side-by-Side Comparison")

    all_results = {}
    for name in MODEL_FILES:
        m = load_model(name, repo_root=REPO_ROOT)
        all_results[name] = compute_metrics(m, X_scaled, y)

    comparison_df = pd.DataFrame(all_results).T
    comparison_df.index.name = 'ML Model Name'
    st.dataframe(comparison_df.style.format("{:.4f}").highlight_max(axis=0, color='lightgreen'))

else:
    st.info(
        "Please upload a CSV file using the sidebar to begin. "
        "Use `test_data.csv` from the GitHub repository."
    )

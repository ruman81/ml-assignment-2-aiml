# ML Assignment 2 — Binary Classification with Multiple ML Models

## a. Problem Statement

This project implements five machine learning classification models on the
**Breast Cancer Wisconsin (Diagnostic)** dataset to classify breast tumours as
**Malignant** or **Benign** based on 30 numeric features computed from digitised
images of fine needle aspirate (FNA) of breast masses.

Early and accurate detection of malignant tumours is critical for effective
cancer treatment. This project demonstrates how classical ML algorithms compare
in a real-world binary classification task and provides an interactive web
interface to evaluate model performance on new test data.

---

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** `sklearn.datasets.load_breast_cancer()` (UCI ML Repository)
- **Instances:** 569
- **Features:** 30 (all continuous float64 — computed from FNA images:
  mean, standard error, and worst value of 10 cell-nucleus measurements
  including radius, texture, perimeter, area, smoothness, compactness,
  concavity, concave points, symmetry, fractal dimension)
- **Target column:** `target` — binary (0 = Malignant, 1 = Benign)
- **Class distribution:** 212 Malignant (37.3%), 357 Benign (62.7%)
- **Missing values:** None
- **Train/Test split:** 80% / 20% stratified on target (random_state=42)
  — Training: 455 instances, Test: 114 instances
- **Preprocessing:** StandardScaler (zero-mean, unit-variance normalization)
  fitted on training data only; applied to both train and test sets.
  No encoding or imputation required (all features are numeric, no missing values).

---

## c. GitHub Repository Link

[https://github.com/ruman81/ml-assignment-2-aiml](https://github.com/ruman81/ml-assignment-2-aiml)

---

## d. Models Used and Evaluation Metrics

All 5 models were trained on the **same** 455-instance training set and evaluated
on the **same** 114-instance test set. No retraining was performed during evaluation.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves the highest accuracy (98.25%) and F1 score (0.9861), with near-perfect AUC (0.9954). As a linear model it is fast to train and highly interpretable, and it performs exceptionally well on this linearly-separable dataset. |
| Decision Tree | Lowest performer overall with accuracy of 91.23% and the lowest AUC (0.9157). Prone to overfitting on training data, which explains the noticeable drop in recall (0.9028). Its axis-aligned splits struggle to capture the smooth decision boundary of this dataset. |
| KNN | Strong recall (0.9722) — the second-highest — meaning it rarely misses benign cases. Accuracy (95.61%) and F1 (0.9655) are solid but slightly below Logistic Regression and SVM. Performance is sensitive to the choice of k and the distance metric. |
| Naive Bayes | Good AUC (0.9868), indicating strong probabilistic ranking despite moderate accuracy (92.98%). The conditional-independence assumption limits point-prediction accuracy, but the well-calibrated probability estimates make it a useful baseline and valuable for risk-scoring applications. |
| Support Vector Machine | Ties with Logistic Regression for the best accuracy (98.25%), F1 (0.9861), and MCC (0.9623). AUC is marginally lower (0.9950 vs. 0.9954). The RBF kernel effectively models the decision boundary, though at greater computational cost than Logistic Regression. |
| **Overall Winner** | **Logistic Regression** — achieves the highest AUC (0.9954) and ties for the best accuracy, F1, and MCC, while being the simplest and most interpretable model. SVM matches it on most metrics but at higher computational cost. |

---

## Live Streamlit App Link

*(To be updated after deployment)*

---

## How to Use the Streamlit App

1. Open the live app link above (or run locally with `streamlit run app.py`).
2. In the sidebar, click **Browse files** and upload `test_data.csv`
   (available in this repository).
3. Select a model from the **Select Classification Model** dropdown.
4. View the 6 evaluation metrics displayed as cards and in a summary table.
5. Click the **Confusion Matrix** tab for the heatmap, or
   **Classification Report** for the full per-class breakdown.
6. Scroll down to the **All Models — Side-by-Side Comparison** table to compare
   all 5 models on the same uploaded test data.

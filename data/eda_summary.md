# EDA Summary - Breast Cancer Wisconsin (Diagnostic)

## Dataset Overview
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Instances:** 569
- **Features:** 30 (all continuous float64)
- **Target:** binary (0 = Malignant, 1 = Benign)

## Class Distribution
- Malignant (0): 212 (37.3%)
- Benign (1): 357 (62.7%)
- Mildly imbalanced; stratified split used to maintain ratio.

## Missing Values
- Total missing values: 0 (none)
- No imputation needed.

## Duplicate Rows
- Duplicate rows: 0

## Feature Ranges
Features have vastly different scales:
- `mean area` ranges from 143.5 to 2501.0
- `mean smoothness` ranges from 0.0526 to 0.1634
- StandardScaler is essential for distance-based models (SVM, KNN).

## Highly Correlated Features
- 21 feature pairs with |r| > 0.9
- Examples: radius, perimeter, and area are highly correlated (r > 0.99)
- PCA/feature selection skipped as not required by the assignment.

## Key Observations
- All features are numeric (float64); no encoding needed.
- Malignant tumors tend to have larger mean radius, area, and perimeter.
- `mean concave points` and `worst concave points` show strong class separation.
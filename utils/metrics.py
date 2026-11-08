import os
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

MODEL_FILES = {
    'Logistic Regression':    'model/logistic_regression.pkl',
    'Decision Tree':          'model/decision_tree.pkl',
    'KNN':                    'model/knn.pkl',
    'Naive Bayes':            'model/naive_bayes.pkl',
    'Support Vector Machine': 'model/svm.pkl',
}


def load_model(model_name: str, repo_root: str = '.'):
    """Load a saved model by display name. No training occurs."""
    path = os.path.join(repo_root, MODEL_FILES[model_name])
    return joblib.load(path)


def compute_metrics(model, X, y) -> dict:
    """
    Return a dict with all 6 required metrics.
    Uses predict_proba for AUC when available, falls back to decision_function.
    Binary classification only.
    """
    y_pred = model.predict(X)

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X)
    else:
        y_prob = y_pred

    return {
        'Accuracy':  round(accuracy_score(y, y_pred), 4),
        'AUC':       round(roc_auc_score(y, y_prob), 4),
        'Precision': round(precision_score(y, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y, y_pred, zero_division=0), 4),
        'F1':        round(f1_score(y, y_pred, zero_division=0), 4),
        'MCC':       round(matthews_corrcoef(y, y_pred), 4),
    }

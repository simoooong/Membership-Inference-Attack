import numpy as np
from sklearn.base import BaseEstimator
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_classifier_model(attack_model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluates the trained Membership Classifier and returns key metrics.

    Args:
        attack_model (BaseEstimator): The trained attack classifier.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics.
    """
    print(f"\n--- Evaluating Membership Classifier (Model M') using {attack_model.__class__.__name__} ---")
    
    # Make predictions
    y_pred = attack_model.predict(X_test)
    
    # Check if the model has predict_proba, as some classifiers (e.g., SVC without probability=True) might not
    if hasattr(attack_model, 'predict_proba'):
        y_prob = attack_model.predict_proba(X_test)[:, 1] # Probability of being a member (class 1)
        auc_roc = roc_auc_score(y_test, y_prob)
    else:
        # For classifiers without predict_proba (e.g., some SVMs), AUROC can't be computed this way
        y_prob = None
        auc_roc = float('nan') # Not a Number

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUROC:     {auc_roc:.4f}")
    print("\nEvaluation complete.")

    return {
        "mia_accuracy": accuracy,
        "mia_precision": precision,
        "mia_recall": recall,
        "mia_f1_score": f1,
        "mia_auroc": auc_roc
    }

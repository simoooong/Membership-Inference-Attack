import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_classifier_model(classifier_model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluates the trained Membership Classifier and prints key metrics.

    Args:
        classifier_model (sklearn.linear_model.LogisticRegression): The trained attack classifier.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.
    """
    print("\n--- Evaluating Membership Classifier (Model M') ---")
    
    # Make predictions
    y_pred = classifier_model.predict(X_test)
    y_prob = classifier_model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUROC:     {auc_roc:.4f}")
    print("\nEvaluation complete.")


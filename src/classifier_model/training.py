import numpy as np
from sklearn.base import BaseEstimator

def train_classifier_model(classifier_model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator:
    """
    Trains a given classifier model as the Membership Classifier (Model M').

    Args:
        classifier_model (BaseEstimator): An instantiated scikit-learn compatible classifier model.
        X_train (np.ndarray): Training features (extracted metrics).
        y_train (np.ndarray): Training labels (0 for non-member, 1 for member).

    Returns:
        BaseEstimator: The trained attack classifier.
    """
    print(f"\n--- Training Membership Classifier (Model M') using {classifier_model.__class__.__name__} ---")
    
    # Train the model
    classifier_model.fit(X_train, y_train)
    
    print(f"Membership Classifier ({classifier_model.__class__.__name__}) training complete.")
    return classifier_model

import numpy as np
from sklearn.linear_model import LogisticRegression

def train_classifier_model(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> LogisticRegression:
    """
    Trains a Logistic Regression classifier as the Membership Classifier (Model M').

    Args:
        X_train (np.ndarray): Training features (extracted metrics).
        y_train (np.ndarray): Training labels (0 for non-member, 1 for member).
        random_state (int): Seed for reproducibility of the classifier's internal randomness.

    Returns:
        sklearn.linear_model.LogisticRegression: The trained attack classifier.
    """
    print("\n--- Training Membership Classifier (Model M') ---")

    classifier_model = LogisticRegression(random_state=random_state, max_iter=1000, solver='liblinear')
    classifier_model.fit(X_train, y_train)

    print("Membership Classifier (Model M') training complete.")

    return classifier_model
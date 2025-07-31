from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_classifier_model(X_train, y_train, random_state):
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

def evaluate_classifier_model(classifier_model: LogisticRegression, X_test, y_test):
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


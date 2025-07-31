import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_classifier_dataset(X_member, y_member,
                           X_non_member, y_non_member,
                           test_size, random_state):
    """
    Combines member and non-member metrics, splits them into stratified
    training and testing sets, and standardizes the features.

    Args:
        X_member (np.ndarray): Feature matrix for member samples.
        y_member (np.ndarray): Labels for member samples (all 1s).
        X_non_member (np.ndarray): Feature matrix for non-member samples.
        y_non_member (np.ndarray): Labels for non-member samples (all 0s).
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - X_attack_train_scaled (np.ndarray): Standardized training features for the attack model.
            - X_attack_test_scaled (np.ndarray): Standardized testing features for the attack model.
            - y_attack_train (np.ndarray): Training labels for the attack model.
            - y_attack_test (np.ndarray): Testing labels for the attack model.
    """
    print("\n--- Preparing classifier dataset ---")

    X_combined = np.vstack((X_member, X_non_member))
    y_combined = np.concatenate((y_member, y_non_member))
    
    print(f"Combined dataset shape: X={X_combined.shape}, y={y_combined.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=test_size, random_state=random_state, stratify=y_combined
    )
    
    print(f"Attack training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Attack testing set shape: X={X_test.shape}, y={y_test.shape}")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features standardized based on training set statistics.")

    return X_train_scaled, X_test_scaled, y_train, y_test

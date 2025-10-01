import os
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List
import json
import uuid
import time
import sys

from src.common.utils import set_seed
from src.common.resnet_model import ResNet18 

from src.target_model.data_preprocessing import get_dataset_info, preprocess_dataset
from src.target_model.load_model import load_model 
from src.target_model.training import train_target_model
from src.target_model.evaluation import extract_membership_metrics

from src.classifier_model.data_preprocessing import prepare_classifier_dataset
from src.classifier_model.training import train_classifier_model
from src.classifier_model.evaluation import evaluate_classifier_model

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator


# --- Utility Functions ---

def load_experiment_configs(filepath: str = 'experiment_configs.json') -> List[Dict[str, Any]]:
    """
    Reads the list of experiment configurations from a JSON file.
    """
    if not os.path.exists(filepath):
        print(f"Error: Configuration file not found at {filepath}", file=sys.stderr)
        return []
    try:
        with open(filepath, 'r') as f:
            configs = json.load(f)
        return configs
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}", file=sys.stderr)
        return []

def get_mia_classifier_model(model_name: str, random_state: int) -> BaseEstimator:
    """
    Returns an instantiated classifier model for MIA based on the name.
    """
    model_name_lower = model_name.lower()
    if model_name_lower == 'logisticregression':
        return LogisticRegression(random_state=random_state, max_iter=1000, solver='liblinear')
    elif model_name_lower == 'mlpclassifier':
        return MLPClassifier(random_state=random_state, max_iter=1000, hidden_layer_sizes=(100, 50))
    elif model_name_lower == 'randomforestclassifier':
        return RandomForestClassifier(random_state=random_state, n_estimators=100)
    elif model_name_lower == 'decisiontreeclassifier':
        return DecisionTreeClassifier(random_state=random_state, max_depth=10)
    elif model_name_lower == 'svc_rbf':
        return SVC(kernel='rbf', random_state=random_state, C=1.0, gamma='scale', probability=True)
    else:
        raise ValueError(f"Unknown MIA classifier model name: {model_name}")

def save_result_to_file(data: Dict[str, Any], filename: str):
    """
    Appends a single JSON object to a JSON Lines file.
    """
    with open(filename, 'a') as f:
        f.write(json.dumps(data) + '\n')

# --- Main Logic ---

def single_run(config_dict: Dict[str, Any]):
    """
    Executes a single experiment run using parameters provided in the config dictionary.
    All parameters are accessed directly via config_dict lookups.
    """
    
    # 1. Instantiation of non-JSON serializable objects
    # NOTE: Assuming these specific classes based on the original fixed parameters.
    MODEL_CLASS = ResNet18 
    OPTIMIZER_CLASS = optim.Adam
    TRAIN_CRITERION = nn.CrossEntropyLoss()
    TEST_CRITERION = nn.CrossEntropyLoss(reduction='none')

    # Get essential setup info
    try:
        _load_fn, datapoint_size, num_classes = get_dataset_info(config_dict['dataset_name'])
    except KeyError:
        print(f"Error: 'dataset_name' is missing in the configuration.", file=sys.stderr)
        return

    # Console output
    print(f"\n--- Starting Experiment Run (Plan: {config_dict.get('experiment_plan', 'N/A')}) ---")
    print(f"  Dataset: {config_dict['dataset_name']}, Scale: {config_dict['scale']:.1f}, LR: {config_dict['learning_rate']}, Epochs: {config_dict['num_epochs']}")
    print(f"  MIA Classifier: {config_dict['mia_classifier_choice']}")
    print(f"--------------------------------------------------------------------")

    try:
        # 2. Preprocess the dataset
        D_member_train_normalized, D_member_val_normalized, D_non_member_normalized = preprocess_dataset(
            dataset_name = config_dict['dataset_name'],
            load_fn = _load_fn,
            data_dir = config_dict['data_dir'],
            num_classes = num_classes,
            train_ratio = config_dict['train_ratio'],
            val_ratio = config_dict['val_ratio'],
            scale = config_dict['scale'],
            download = True
        )

        # 3. Target Model Training
        start_time_target_model = time.time()
        
        # model_m is initialized to None and then trained
        model_m, best_val_accuracy, best_val_loss, epochs_trained = train_target_model(
            dataset_name = config_dict['dataset_name'],
            train_dataset = D_member_train_normalized,
            val_dataset = D_member_val_normalized,
            model_class = MODEL_CLASS,
            model_dir = config_dict['model_dir'], 
            num_classes = num_classes,
            train_ratio = config_dict['train_ratio'],
            val_ratio = config_dict['val_ratio'],
            scale = config_dict['scale'], 
            num_epochs = config_dict['num_epochs'],
            learning_rate = config_dict['learning_rate'],
            train_criterion = TRAIN_CRITERION,
            optimizer_class = OPTIMIZER_CLASS,
            batch_size = config_dict['batch_size'],
            patience = config_dict['patience'],
            min_delta = config_dict['min_delta'],
            early_stopping = False if config_dict['EARLY_STOPPING'] == "Deactivated" else True
        )
        
        end_time_target_model = time.time()

        # 4. Extract membership metrics
        X_member, y_member = extract_membership_metrics(
            model = model_m,
            dataset = D_member_train_normalized,
            is_member_label = True,
            test_criterion = TEST_CRITERION,
            batch_size = config_dict['batch_size']
        )

        X_non_member, y_non_member = extract_membership_metrics(
            model = model_m,
            dataset = D_non_member_normalized,
            is_member_label = False,
            test_criterion = TEST_CRITERION,
            batch_size = config_dict['batch_size']
        )

        # 5. Prepare classifier dataset
        X_train, X_test, y_train, y_test = prepare_classifier_dataset(
            X_member = X_member,
            y_member = y_member,
            X_non_member = X_non_member,
            y_non_member = y_non_member,
            test_size = config_dict['test_size'],
            random_state = config_dict['SEED']
        )

        # 6. Train MIA classifier model
        start_time_mia_classifier = time.time()
        mia_classifier = get_mia_classifier_model(config_dict['mia_classifier_choice'], config_dict['SEED'])
        classifier_model = train_classifier_model(
            classifier_model = mia_classifier,
            X_train = X_train,
            y_train = y_train,
        )
        end_time_mia_classifier = time.time()

        # 7. Evaluate Classifier Model
        mia_metrics = evaluate_classifier_model(
            attack_model = classifier_model,
            X_test = X_test,
            y_test = y_test
        )

        # 8. Collect and Save Results
        metrics = {
            "experiment_id": str(uuid.uuid4()),
            "dataset_name": config_dict['dataset_name'],
            "datapoint_size": datapoint_size,
            "num_classes": num_classes,
            "scale": config_dict['scale'],
            "experiment_plan": config_dict.get('experiment_plan', 'N/A'),
            
            # Target Model Params
            "target_model_learning_rate": config_dict['learning_rate'],
            "target_model_epochs_requested": config_dict['num_epochs'],
            "target_model_batch_size": config_dict['batch_size'],
            "target_model_validation_ratio": config_dict['val_ratio'],
            "target_model_member_train_val_split_ratio": config_dict['train_ratio'],
            "target_model_patience": config_dict['patience'],
            "target_model_min_delta": config_dict['min_delta'],
            "seed": config_dict['SEED'],
            
            # Target Model Results
            "target_model_training_time_seconds": end_time_target_model - start_time_target_model,
            "target_model_final_val_accuracy": best_val_accuracy,
            "target_model_final_val_loss": best_val_loss,
            "target_model_epochs_trained": epochs_trained,
            
            # MIA Classifier Params & Results
            "mia_classifier_choice": config_dict['mia_classifier_choice'],
            "mia_training_time_seconds": end_time_mia_classifier - start_time_mia_classifier
        }
        metrics.update(mia_metrics)
        save_result_to_file(metrics, config_dict['results_file'])
        print(f"Successfully saved results for this run to {config_dict['results_file']}.")
        
    except Exception as e:
        print(f"An error occurred during the single experiment run: {e}", file=sys.stderr)


def main():
    """
    Main function to orchestrate the loading and execution of all experiments
    defined in experiment_configs.json.
    """
    CONFIG_FILE = 'experiment_configs.json'
    
    # 1. Load all configurations
    all_configs = load_experiment_configs(CONFIG_FILE)

    if not all_configs:
        print(f"No configurations loaded. Exiting.", file=sys.stderr)
        return

    # Get configuration for setup from the first run (assuming fixed global paths/seed)
    setup_config = all_configs[0]
    set_seed(setup_config['SEED'])

    os.makedirs(setup_config['data_dir'], exist_ok=True)
    os.makedirs(setup_config['model_dir'], exist_ok=True)

    print(f"--- Starting Multi-Experiment Orchestration ---")
    print(f"Total experiment runs to execute: {len(all_configs)}")
    print(f"Results will be stored in: {setup_config['results_file']}")
    print(f"---------------------------------------------")

    # 2. Iterate through and execute each configuration
    for i, config_dict in enumerate(all_configs):
        print(f"\n[{i+1}/{len(all_configs)}] Executing run: {config_dict.get('experiment_plan', 'N/A')}")
        single_run(config_dict)

if __name__ == '__main__':
    main()

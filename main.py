import os
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Type, Dict, Any, List
import json
import uuid
import time

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

@dataclass
class TrainingConfig:
    SEED: int = 42
    data_dir: str = './data'
    model_dir: str = './saved_models'
    results_file: str = './results.jsonl'

    dataset_names: List[str] = field(default_factory=lambda: ['CIFAR-10', 'stl-10'])
    scales: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.05])
    mia_classifier_choices: List[str] = field(default_factory=lambda: ['logisticregression', 'mlpclassifier', 'randomforestclassifier', 'decisiontreeclassifier', 'svc_rbf'])

    val_ratio: float = 0.1
    train_ratio: float = 1 / (2 - val_ratio)

    batch_size: int = 64
    test_size: float = 0.15

    patience = 7
    min_delta = 0.001

    num_classes: int = 10
    num_epochs: int = 50
    learning_rate: float = 0.01
    model_class: Type[nn.Module] = ResNet18
    optimizer_class: Type[optim.Optimizer] = optim.Adam
    train_criterion: nn.Module = nn.CrossEntropyLoss()
    test_criterion: nn.Module = nn.CrossEntropyLoss(reduction='none')

def get_mia_classifier_model(model_name: str, random_state: int) -> BaseEstimator:
    """
    Returns an instantiated classifier model for MIA based on the name.
    """
    if model_name.lower() == 'logisticregression':
        return LogisticRegression(random_state=random_state, max_iter=1000, solver='liblinear')
    elif model_name.lower() == 'mlpclassifier':
        # Added hidden_layer_sizes for a more sensible MLP default
        return MLPClassifier(random_state=random_state, max_iter=1000, hidden_layer_sizes=(100, 50))
    elif model_name.lower() == 'randomforestclassifier':
        return RandomForestClassifier(random_state=random_state, n_estimators=100)
    elif model_name.lower() == 'decisiontreeclassifier':
        return DecisionTreeClassifier(random_state=random_state, max_depth=10) # Set a max_depth to prevent overfitting
    elif model_name.lower() == 'svc_rbf':
        return SVC(kernel='rbf', random_state=random_state, C=1.0, gamma='scale', probability=True)
    else:
        raise ValueError(f"Unknown MIA classifier model name: {model_name}")

def save_result_to_file(data: Dict[str, Any], filename: str):
    """
    Appends a single JSON object to a JSON Lines file.
    """
    with open(filename, 'a') as f:
        f.write(json.dumps(data) + '\n')

def single_run(config: TrainingConfig, current_dataset: str, current_scale: float):
    """
    Executes a single experiment run with the specified dataset, scale, and MIA classifier choice.
    """
    print(f"\n--- Starting Experiment Run for: ---")
    print(f"  Dataset: {current_dataset}")
    print(f"  Scale: {current_scale}")
    print(f"------------------------------------")

    _load_fn, datapoint_size, num_classes = get_dataset_info(current_dataset)

    try:
        # 1. Preprocess the CIFAR-10 dataset to get standardized D_member and D_non_member
        D_member_train_normalized, D_member_val_normalized, D_non_member_normalized = preprocess_dataset(
            dataset_name = current_dataset,
            load_fn = _load_fn,
            data_dir = config.data_dir,
            num_classes = num_classes,
            train_ratio = config.train_ratio,
            val_ratio = config.val_ratio,
            scale = current_scale,
            download = True
        )

        # 2. Load Model if present
        model_m, best_val_accuracy, best_val_loss, epochs_trained = load_model(
            model_dir = config.model_dir,
            model_class = config.model_class,
            num_classes = num_classes,
            dataset_name = current_dataset,
            scale = current_scale,
            train_ratio = config.train_ratio,
            val_ratio = config.val_ratio
        )

        # 3. Train target model if not present
        start_time_target_model = time.time()
        if model_m is None:
            model_m, best_val_accuracy, best_val_loss, epochs_trained = train_target_model(
                dataset_name = current_dataset,
                train_dataset = D_member_train_normalized,
                val_dataset = D_member_val_normalized,
                model_class = config.model_class,
                model_dir = config.model_dir, 
                num_classes = num_classes,
                train_ratio = config.train_ratio,
                val_ratio = config.val_ratio,
                scale = current_scale, 
                num_epochs = config.num_epochs,
                learning_rate = config.learning_rate,
                train_criterion = config.train_criterion,
                optimizer_class = config.optimizer_class,
                batch_size = config.batch_size,
                patience = config.patience,
                min_delta = config.min_delta
            )
        
        end_time_target_model = time.time()

        # 4. Extract_membership_metrics
        X_member, y_member = extract_membership_metrics(
            model = model_m,
            dataset = D_member_train_normalized,
            is_member_label = True,
            test_criterion = config.test_criterion,
            batch_size = config.batch_size
        )

        X_non_member, y_non_member = extract_membership_metrics(
            model = model_m,
            dataset = D_non_member_normalized,
            is_member_label = False,
            test_criterion = config.test_criterion,
            batch_size = config.batch_size
        )

        # 5. Preprocess the membership metrics dataset to get standardized X_train_scaled, X_test_scaled, y_train, y_test
        X_train, X_test, y_train, y_test = prepare_classifier_dataset(
            X_member = X_member,
            y_member = y_member,
            X_non_member = X_non_member,
            y_non_member = y_non_member,
            test_size = config.test_size,
            random_state = config.SEED
        )

        # 6. Train classifier model
        for current_mia_classifier_choice in config.mia_classifier_choices:
            print(f"  MIA Classifier: {current_mia_classifier_choice}")

            start_time_mia_classifier = time.time()
            mia_classifier = get_mia_classifier_model(current_mia_classifier_choice, config.SEED)
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

            metrics = {
                "experiment_id": str(uuid.uuid4()),
                "dataset_name": current_dataset,
                "datapoint_size": datapoint_size,
                "num_classes": num_classes,
                "scale": current_scale,
                "target_model_learning_rate": config.learning_rate,
                "target_model_batch_size": config.batch_size,
                "target_model_validation_ratio": config.val_ratio,
                "target_model_member_train_val_split_ratio": config.train_ratio,
                "target_model_patience": config.patience,
                "target_model_min_delta": config.min_delta,
                "seed": config.SEED,
                "target_model_training_time_seconds": end_time_target_model - start_time_target_model,
                "target_model_final_val_accuracy": best_val_accuracy,
                "target_model_final_val_loss": best_val_loss,
                "target_model_epochs_trained": epochs_trained,
                "mia_classifier_choice": current_mia_classifier_choice,
                "mia_training_time_seconds": end_time_mia_classifier - start_time_mia_classifier
            }
            metrics.update(mia_metrics)
            save_result_to_file(metrics, config.results_file)
            print(f"Successfully saved results for this run to {config.results_file}.")
    except Exception as e:
        print(f"An error occurred during the single experiment run: {e}")

def main():
    base_config = TrainingConfig()
    set_seed(base_config.SEED)

    os.makedirs(base_config.data_dir, exist_ok=True)
    os.makedirs(base_config.model_dir, exist_ok=True)

    print(f"--- Starting Multi-Experiment Orchestration ---")
    print(f"Models will be saved in: {base_config.model_dir}")
    print(f"Results will be stored in: {base_config.results_file}")
    print(f"Iterating over {len(base_config.dataset_names)} datasets, {len(base_config.scales)} scales, and {len(base_config.mia_classifier_choices)} MIA classifiers.")
    print(f"Total runs expected: {len(base_config.dataset_names) * len(base_config.scales) * len(base_config.mia_classifier_choices)}")
    print(f"------------------------------------")

    for dataset_name in base_config.dataset_names:
        for scale in base_config.scales:
            single_run(base_config, dataset_name, scale)

# def main():
#     config = TrainingConfig()
#     set_seed(config.SEED)

#     os.makedirs(config.data_dir, exist_ok=True)
#     os.makedirs(config.model_dir, exist_ok=True)

#     _load_fn, datapoint_size, num_classes = get_dataset_info(config.dataset_name)

#     print(f"--- Starting Single Experiment Run ---")
#     print(f"Dataset: {config.dataset_name}, Datapoint Size: {datapoint_size}, Num Classes: {num_classes}")
#     print(f"Scale: {config.scale}, MIA Classifier: {config.mia_classifier_choice}")
#     print(f"Target Model Learning Rate: {config.learning_rate}, Batch Size: {config.batch_size}")
#     print(f"Results will be stored in: {config.results_file}")
#     print(f"------------------------------------")

#     metrics = {
#         "experiment_id": str(uuid.uuid4()),
#         "dataset_name": config.dataset_name,
#         "datapoint_size": datapoint_size,
#         "num_classes": num_classes,
#         "scale": config.scale,
#         "mia_classifier_choice": config.mia_classifier_choice,
#         "target_model_learning_rate": config.learning_rate,
#         "target_model_batch_size": config.batch_size,
#         "target_model_validation_ratio": config.val_ratio,
#         "target_model_member_train_val_split_ratio": config.train_ratio,
#         "target_model_patience": config.patience,
#         "target_model_min_delta": config.min_delta,
#         "seed": config.SEED
#     }

#     try:
#         # 1. Preprocess the CIFAR-10 dataset to get standardized D_member and D_non_member
#         D_member_train_normalized, D_member_val_normalized, D_non_member_normalized = preprocess_dataset(
#             dataset_name = config.dataset_name,
#             load_fn = _load_fn,
#             data_dir = config.data_dir,
#             num_classes = num_classes,
#             train_ratio = config.train_ratio,
#             val_ratio = config.val_ratio,
#             scale = config.scale,
#             download = True
#         )

#         # 2. Load Model if present
#         model_m, best_val_accuracy, best_val_loss, epochs_trained = load_model(
#             model_dir = config.model_dir,
#             model_class = config.model_class,
#             num_classes = config.num_classes,
#             dataset_name = config.dataset_name,
#             scale = config.scale,
#             train_ratio = config.train_ratio,
#             val_ratio = config.val_ratio
#         )

#         # 3. Train target model if not present
#         start_time_target_model = time.time()
#         if model_m is None:
#             model_m, best_val_accuracy, best_val_loss, epochs_trained = train_target_model(
#                 dataset_name = config.dataset_name,
#                 train_dataset = D_member_train_normalized,
#                 val_dataset = D_member_val_normalized,
#                 model_class = config.model_class,
#                 model_dir = config.model_dir, 
#                 num_classes = num_classes,
#                 train_ratio = config.train_ratio,
#                 val_ratio = config.val_ratio,
#                 scale = config.scale, 
#                 num_epochs = config.num_epochs,
#                 learning_rate = config.learning_rate,
#                 train_criterion = config.train_criterion,
#                 optimizer_class = config.optimizer_class,
#                 batch_size = config.batch_size,
#                 patience = config.patience,
#                 min_delta = config.min_delta
#             )
        
#         end_time_target_model = time.time()
#         metrics["target_model_training_time_seconds"] = end_time_target_model - start_time_target_model
#         metrics["target_model_final_val_accuracy"] = best_val_accuracy
#         metrics["target_model_final_val_loss"] = best_val_loss
#         metrics["target_model_epochs_trained"] = epochs_trained

#         # 4. Extract_membership_metrics
#         X_member, y_member = extract_membership_metrics(
#             model = model_m,
#             dataset = D_member_train_normalized,
#             is_member_label = True,
#             test_criterion = config.test_criterion,
#             batch_size = config.batch_size
#         )

#         X_non_member, y_non_member = extract_membership_metrics(
#             model = model_m,
#             dataset = D_non_member_normalized,
#             is_member_label = False,
#             test_criterion = config.test_criterion,
#             batch_size = config.batch_size
#         )

#         # 5. Preprocess the membership metrics dataset to get standardized X_train_scaled, X_test_scaled, y_train, y_test
#         X_train, X_test, y_train, y_test = prepare_classifier_dataset(
#             X_member = X_member,
#             y_member = y_member,
#             X_non_member = X_non_member,
#             y_non_member = y_non_member,
#             test_size = config.test_size,
#             random_state = config.SEED
#         )

#         # 6. Train classifier model
#         start_time_mia_classifier = time.time()
#         mia_classifier = get_mia_classifier_model(config.mia_classifier_choice, config.SEED)
#         classifier_model = train_classifier_model(
#             classifier_model = mia_classifier,
#             X_train = X_train,
#             y_train = y_train,
#         )
#         end_time_mia_classifier = time.time()
#         metrics["mia_training_time_seconds"] = end_time_mia_classifier - start_time_mia_classifier

#         # 7. Evaluate Classifier Model
#         mia_metrics = evaluate_classifier_model(
#             attack_model = classifier_model,
#             X_test = X_test,
#             y_test = y_test
#         )

#         metrics.update(mia_metrics)
#         save_result_to_file(metrics, config.results_file)
#         print(f"Successfully saved results for the single run.")
#     except Exception as e:
#         print(f"An error occurred during the single experiment run: {e}")

#     print(f"\n--- Single Experiment Run Complete ---")

if __name__ == '__main__':
    main()
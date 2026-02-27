import json
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class FixedParams:
    """
    Configuration parameters that remain constant across all experiments.
    These values are used to record the experimental environment for reproducibility.
    
    NOTE: These parameters are merged directly into the final config dictionary.
    """
    SEED: int = 42
    data_dir: str = './data'
    model_dir: str = './saved_models'
    results_file: str = './artifacts/all_params.jsonl'
    
    # Ratios (train_ratio = 1 / (2 - val_ratio))
    val_ratio: float = 0.1
    train_ratio: float = 1 / (2 - 0.1) # approx 0.5263
    
    batch_size: int = 64
    test_size: float = 0.15
    patience: int = 7
    min_delta: float = 0.001
    
    num_classes: int = 10
    model_class_name: str = "ResNet18"
    optimizer_class_name: str = "Adam"
    train_criterion_name: str = "CrossEntropyLoss"
    test_criterion_name: str = "CrossEntropyLoss(reduction='none')"
    
    EARLY_STOPPING: str = "Deactivated"

@dataclass
class DynamicParams:
    """
    Lists of parameters that will be iterated over to define the experiment space.
    """
    dataset_names: List[str] = field(default_factory=lambda: ['CIFAR-10', 'STL-10'])
    # Scales are used in the Scale impact plan
    scales: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    learning_rates: List[float] = field(default_factory=lambda: [0.1, 0.01, 0.001, 0.0001])
    # Total epochs for training the target model M
    num_epochs: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]) 
    mia_classifier_choices: List[str] = field(default_factory=lambda: [
        'logisticregression', 
        'mlpclassifier', 
        'randomforestclassifier', 
        'decisiontreeclassifier', 
        'svc_rbf'
    ])


def generate_experiment_config() -> List[Dict[str, Any]]:
    """
    Generates a list of experiment configurations by combining fixed and dynamic parameters.
    The list is split into three distinct experimental plans: 
    1. Impact of Learning Rate (Fixed Epochs=15, Fixed Scale=1.0).
    2. Impact of Epochs (Fixed Learning Rate=0.01, Fixed Scale=1.0).
    3. Impact of Scale (Fixed Learning Rate=0.01, Fixed Epochs=15).
    """
    
    fixed = FixedParams()
    dynamic = DynamicParams()
    all_configs = []

    fixed_epochs = 15
    fixed_scale = 0.01
    fixed_lr = 0.01

    # ====================================================================
    # 1. EXPERIMENT PLAN: IMPACT OF LEARNING RATE (Epochs=15, Scale=1.0 Fixed)
    # Total Runs: 2 Datasets * 4 LRs * 1 Scale * 5 Classifiers = 40
    # ====================================================================
    
    for dataset in dynamic.dataset_names:
        for lr in dynamic.learning_rates:
            for mia_clf in dynamic.mia_classifier_choices:
                config = {
                    "experiment_plan": "Impact of Learning Rate",
                    "results_file": "./artifacts/results.jsonl",
                    "dataset_name": dataset,
                    "learning_rate": lr,
                    "scale": fixed_scale,
                    "mia_classifier_choice": mia_clf,
                    "num_epochs" : fixed_epochs,
                    **fixed.__dict__
                }
                all_configs.append(config)
    
    print(f"Generated {len(all_configs)} configs for Learning Rate Impact (Epochs={fixed_epochs}, Scale={fixed_scale}).")


    # ====================================================================
    # 2. EXPERIMENT PLAN: IMPACT OF #EPOCHS (LR=0.01, Scale=1.0 Fixed)
    # Total Runs: 2 Datasets * 10 Epochs * 1 Scale * 5 Classifiers = 100
    # ====================================================================

    initial_config_count = len(all_configs)

    for dataset in dynamic.dataset_names:
        for num_epochs in dynamic.num_epochs:
            for mia_clf in dynamic.mia_classifier_choices:
                config = {
                    "experiment_plan": "Impact of Epochs",
                    "results_file": "./results.jsonl",
                    "dataset_name": dataset,
                    "learning_rate" : fixed_lr,
                    "scale": fixed_scale,
                    "mia_classifier_choice": mia_clf,
                    "num_epochs": num_epochs,
                    **fixed.__dict__
                }
                all_configs.append(config)
    

    new_configs_generated_plan2 = len(all_configs) - initial_config_count
    print(f"Generated {new_configs_generated_plan2} new configs for Epochs Impact (LR={fixed_lr}, Scale={fixed_scale}).")

    
    # ====================================================================
    # 3. EXPERIMENT PLAN: IMPACT OF Scale (LR=0.01, Epochs=15 Fixed)
    # Total Runs: 2 Datasets * 10 Scales * 1 Epochs * 5 Classifiers = 100
    # ====================================================================

    initial_config_count_plan3 = len(all_configs)

    for dataset in dynamic.dataset_names:
        for scale in dynamic.scales:
            for mia_clf in dynamic.mia_classifier_choices:
                config = {
                    "experiment_plan" : "Impact of Scale",
                    "results_file": "./results.jsonl",
                    "dataset_name": dataset,
                    "learning_rate" : fixed_lr,
                    "scale": scale,
                    "mia_classifier_choice": mia_clf,
                    "num_epochs": fixed_epochs,
                    **fixed.__dict__
                }
                all_configs.append(config)


    new_configs_generated_plan3 = len(all_configs) - initial_config_count_plan3
    total_configs = len(all_configs)
    
    print(f"Generated {new_configs_generated_plan3} new configs for Scale Impact (LR={fixed_lr}, Epochs={fixed_epochs}).")
    print(f"Total experiment configurations generated: {total_configs}") # Total: 40 + 100 + 100 = 240
    
    return all_configs


if __name__ == '__main__':
    configs = generate_experiment_config()

    # Save the final configuration list to a JSON file
    output_file = 'experiment_configs.json'
    with open(output_file, 'w') as f:
        # Use indent=4 for human readability
        json.dump(configs, f, indent=4)
        
    print(f"\nSuccessfully saved {len(configs)} configurations to '{output_file}'.")

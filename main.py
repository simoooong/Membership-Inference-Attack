import os
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Type

from src.common.utils import set_seed
from src.common.resnet_model import ResNet18

from src.target_model.data_preprocessing import preprocess_cifar10_dataset
from src.target_model.load_model import load_model
from src.target_model.training import train_target_model
from src.target_model.evaluation import extract_membership_metrics

from src.classifier_model.data_preprocessing import prepare_classifier_dataset
from src.classifier_model.training import train_classifier_model
from src.classifier_model.evaluation import evaluate_classifier_model

@dataclass
class TrainingConfig:
    SEED: int = 42
    data_dir: str = './data'
    model_dir: str = './saved_models'

    train_ratio: float = 0.5
    scale: float = 0.1
    batch_size: int = 64
    test_size: float = 0.15

    num_classes: int = 10
    num_epochs: int = 50
    learning_rate: float = 0.001
    model_class: Type[nn.Module] = ResNet18
    optimizer_class: Type[optim.Optimizer] = optim.Adam
    train_criterion: nn.Module = nn.CrossEntropyLoss()
    test_criterion: nn.Module = nn.CrossEntropyLoss(reduction='none')

def main():
    config = TrainingConfig()
    set_seed(config.SEED)

    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)



    # 1. Preprocess the CIFAR-10 dataset to get standardized D_member and D_non_member
    D_member_normalized, D_non_member_normalized = preprocess_cifar10_dataset(
        data_dir=config.data_dir,
        num_classes=config.num_classes,
        train_ratio=config.train_ratio,
        scale=config.scale,
        download= True
    )

    # 2. Load Model if present
    model_m = load_model(
        model_dir=config.model_dir,
        model_class=config.model_class,
        num_classes=config.num_classes,
        train_ratio=config.train_ratio,
        scale=config.scale
    )

    # 3. Train target model if not present
    if model_m is None:
        model_m = train_target_model(
            D_member_normalized=D_member_normalized,
            model_class=config.model_class,
            model_dir=config.model_dir, 
            num_classes=config.num_classes,
            train_ratio=config.train_ratio,    
            scale=config.scale, 
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            train_criterion=config.train_criterion,
            optimizer_class=config.optimizer_class,
            batch_size=config.batch_size
        )

    # 4. Extract_membership_metrics
    X_member, y_member = extract_membership_metrics(
        model=model_m,
        dataset=D_member_normalized,
        is_member_label=True,
        test_criterion=config.test_criterion,
        batch_size=config.batch_size
    )

    X_non_member, y_non_member = extract_membership_metrics(
        model=model_m,
        dataset=D_non_member_normalized,
        is_member_label=False,
        test_criterion=config.test_criterion,
        batch_size=config.batch_size
    )

    # 5. Preprocess the membership metrics dataset to get standardized X_train_scaled, X_test_scaled, y_train, y_test
    X_train, X_test, y_train, y_test = prepare_classifier_dataset(
        X_member=X_member,
        y_member=y_member,
        X_non_member=X_non_member,
        y_non_member=y_non_member,
        test_size=config.test_size,
        random_state=config.SEED
    )

    # 6. Train classifier model
    classifier_model = train_classifier_model(
        X_train=X_train,
        y_train=y_train,
        random_state=config.SEED
    )

    # 7. Evaluate Classifier Model
    evaluate_classifier_model(
        classifier_model=classifier_model,
        X_test=X_test,
        y_test=y_test
    )

if __name__ == '__main__':
    main()
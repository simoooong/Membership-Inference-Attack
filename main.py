import os
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data_preprocessing import preprocess_cifar10_dataset
from load_model import load_model
from training_target_model import train_target_model, set_seed
from test_target_model import extract_membership_metrics
from preprocess_classifier_data import prepare_classifier_dataset
from train_classifier_model import train_classifier_model, evaluate_classifier_model

if __name__ == '__main__':
    SEED = 42
    set_seed(SEED)

    data_dir = './data'
    model_dir='./saved_models'

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_ratio = 0.5
    scale = 0.1
    num_classes = 10
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 64
    optimizer_class = optim.Adam
    train_criterion = nn.CrossEntropyLoss()
    test_criterion = nn.CrossEntropyLoss(reduction='none')
    test_size = 0.15

    # 1. Preprocess the CIFAR-10 dataset to get standardized D_member and D_non_member
    D_member_normalized, D_non_member_normalized = preprocess_cifar10_dataset(
        data_dir=data_dir,
        train_ratio=train_ratio,
        scale=scale,
        download= True
    )

    # 2. Load Model if present
    model_m = load_model(
        model_dir=model_dir,
        num_classes=num_classes,
        train_ratio=train_ratio,
        scale=scale
    )

    # 3. Train target model if not present
    if model_m is None:
        model_m = train_target_model(
            D_member_normalized=D_member_normalized, 
            model_dir=model_dir, 
            num_classes=num_classes,
            train_ratio=train_ratio,    
            scale=scale, 
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            train_criterion=train_criterion,
            optimizer_class=optimizer_class,
            batch_size=batch_size
        )

    # 4. Extract_membership_metrics
    X_member, y_member = extract_membership_metrics(
        model=model_m,
        dataset=D_member_normalized,
        is_member_label=True,
        test_criterion=test_criterion,
        batch_size=batch_size
    )

    X_non_member, y_non_member = extract_membership_metrics(
        model=model_m,
        dataset=D_non_member_normalized,
        is_member_label=False,
        test_criterion=test_criterion,
        batch_size=batch_size
    )

    # 5. Preprocess the membership metrics dataset to get standardized X_train_scaled, X_test_scaled, y_train, y_test
    X_train, X_test, y_train, y_test = prepare_classifier_dataset(
        X_member=X_member,
        y_member=y_member,
        X_non_member=X_non_member,
        y_non_member=y_non_member,
        test_size=test_size,
        random_state=SEED
    )

    # 6. Train classifier model
    classifier_model = train_classifier_model(
        X_train=X_train,
        y_train=y_train,
        random_state=SEED
    )

    # 7. Evaluate Classifier Model
    evaluate_classifier_model(
        classifier_model=classifier_model,
        X_test=X_test,
        y_test=y_test
    )
    
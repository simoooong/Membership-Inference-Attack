import os
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data_preprocessing import preprocess_cifar10_dataset
from load_model import load_model
from training_target_model import train_target_model, set_seed
from test_target_model import extract_membership_metrics

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

    # 1. Preprocess the CIFAR-10 dataset to get normalized D_member and D_non_member
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

    print(X_member)
    print("_____________________________________________________________")
    print(X_non_member)

    avg_metrics_member = np.mean(X_member, axis=0)
    avg_metrics_non_member = np.mean(X_non_member, axis=0)


    print("\n--- Key Metric Comparison (Averages) ---")
    print(f"Average P_gt: Member={avg_metrics_member[0]:.4f}, Non-Member={avg_metrics_non_member[0]:.4f}")
    print(f"Average L_CE: Member={avg_metrics_member[1]:.4f}, Non-Member={avg_metrics_non_member[1]:.4f}")
    print(f"Average Entropy: Member={avg_metrics_member[2]:.4f}, Non-Member={avg_metrics_non_member[2]:.4f}")
    print(f"Average Top-1 Sorted Probability: Member={avg_metrics_member[3]:.4f}, Non-Member={avg_metrics_non_member[3]:.4f}")
    print(f"Average Top-1 Sorted Logit: Member={avg_metrics_member[3 + num_classes]:.4f}, Non-Member={avg_metrics_non_member[3 + num_classes]:.4f}")
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from typing import Tuple
from scipy.stats import entropy

def extract_membership_metrics(
    model: nn.Module,
    dataset: Dataset,
    is_member_label: bool,
    test_criterion: nn.Module,
    batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts various membership inference metrics for each sample in a dataset
    using a trained model.

    Args:
        model (nn.Module): The trained target model (Model M).
        dataset (Dataset): The dataset to extract metrics from (e.g., D_non_member_normalized).
        is_member_label (bool): True if samples in this dataset are members, False for non-members.
        test_criterion (nn.Module): The loss function to use for calculating Cross-Entropy Loss.
                                   Defaults to nn.CrossEntropyLoss(reduction='none').
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
                                       - X (np.ndarray): Feature matrix (metrics) for the samples.
                                       - y (np.ndarray): Membership labels for the samples (0 or 1).
                                       Metrics include:
                                       - 'p_gt': Ground-Truth Class Probability
                                       - 'l_ce': Cross-Entropy Loss
                                       - 'entropy': Output Probability Distribution Entropy
                                       - 'sorted_probabilities': The full probability vector sorted in descending order.
                                       - 'sorted_logits': The full logits vector sorted in descending order.
    """
     
    device = next(model.parameters()).device
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() if os.cpu_count() else 0)

    all_metrics = []
    all_labels = []
    
    print(f"\n--- Extracting metrics from {len(dataset)} samples ---")

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            probabilities = F.softmax(logits, dim=1)
            losses = test_criterion(logits, labels)

            for i in range(inputs.size(0)):
                true_label = labels[i].item()
                sample_probabilities = probabilities[i]
                sample_logits = logits[i]

                # 1. Ground-Truth Class Probability (P_gt)
                p_gt = sample_probabilities[true_label].item()

                # 2. Cross-Entropy Loss (L_ce)
                l_ce = losses[i].item()

                # 3. Output Probability Distribution Entropy (H)
                h = entropy(sample_probabilities.cpu().numpy(), base=2)

                # 4. Sorted probabilities vector
                sorted_probs, _ = torch.sort(sample_probabilities, descending=True)
                
                # 5. Sorted logits vector
                sorted_logits, _ = torch.sort(sample_logits, descending=True)

                sample_feature = np.concatenate([
                    np.array([p_gt, l_ce, h]),
                    sorted_probs.cpu().numpy(),
                    sorted_logits.cpu().numpy()
                ])
                
                all_metrics.append(sample_feature)
                all_labels.append(1 if is_member_label else 0)
    
    print(f"Metric extraction complete for {len(dataset)} samples.")
    
    X = np.array(all_metrics)
    y = np.array(all_labels)

    return X, y

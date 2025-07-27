import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset, Dataset
import os
import random


def _load_cifar10(data_dir='./data', download=True):
    """
    Loads CIFAR-10 datasets (train and test) without normalization,
    only converting images to tensors.
    """

    trainset_orig = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                           download=download, transform=transforms.ToTensor())

    testset_orig = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                                download=download, transform=transforms.ToTensor())

    full_cifar10_dataset = ConcatDataset([trainset_orig, testset_orig])

    print(f"Loaded original CIFAR-10 training set: {len(trainset_orig)} samples")
    print(f"Loaded original CIFAR-10 testing set: {len(testset_orig)} samples")
    print(f"Combined CIFAR-10 dataset size: {len(full_cifar10_dataset)} samples")

    return full_cifar10_dataset

def _get_mean_std(dataset):
    """
    Calculates the channel-wise mean and standard deviation of a PyTorch dataset.
    Assumes dataset items are already converted to Tensors.
    
    This implementation uses the numerically stable formula for variance: Var(X) = E[X^2] - (E[X])^2
    This is preferred over accumulating standard deviations directly, which is mathematically incorrect.
    """

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count() if os.cpu_count() else 0)

    channels_sum = torch.zeros(3, dtype=torch.float64)
    channels_squared_sum = torch.zeros(3, dtype=torch.float64)
    num_pixels = 0

    print(f"Calculating mean and std from {len(dataset)} samples...")
    for data, _ in loader:
        batch_size, _, height, width = data.shape
        current_batch_pixels = batch_size * height * width

        channels_sum += torch.sum(data, dim=[0, 2, 3])
        channels_squared_sum += torch.sum(data**2, dim=[0, 2, 3])

        num_pixels += current_batch_pixels

    mean = channels_sum / num_pixels
    mean_of_squares = channels_squared_sum / num_pixels
    std = torch.sqrt(mean_of_squares - mean**2)

    print(f"Calculated Mean: {mean.tolist()}")
    print(f"Calculated Std: {std.tolist()}")

    return mean, std

class TransformedSubset(Dataset):
    """
    A wrapper for torch.utils.data.Subset that applies a transform
    during __getitem__. Useful when the transform (e.g., Normalize)
    depends on properties of the subset itself.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def _split_dataset(dataset, train_ratio=0.5, scale=1.0, num_classes=10):
    """
    Splits the full dataset into D_member and D_non_member using stratified sampling
    to ensure equal class distribution in both subsets, considering a 'scale' factor
    for the total data size to be used.

    Args:
        dataset (torch.utils.data.Dataset): The combined CIFAR-10 dataset (e.g., 60,000 samples).
        train_ratio (float): The ratio of 'member' samples within the 'data_size' pool.
                             (e.g., 0.5 for a 50/50 split between member and non-member).
        scale (float): The fraction of the total dataset to use for the member/non-member split.
                       (e.g., 1.0 for 100% of the dataset, 0.5 for 50%).
        num_classes (int): The number of classes in the dataset (e.g., 10 for CIFAR-10).

    Returns:
        tuple: (D_member, D_non_member)
    """

    total_samples_in_full_dataset = len(dataset)
    data_size = int(total_samples_in_full_dataset * scale)

    member_size = int(data_size * train_ratio)
    non_member_size = data_size - member_size

    samples_per_class_member = member_size // num_classes
    samples_per_class_non_member = non_member_size // num_classes

    member_remainder = member_size % num_classes
    non_member_remainder = non_member_size % num_classes

    class_indices = {i: [] for i in range(num_classes)}

    for i in range(total_samples_in_full_dataset):
        _, label = dataset[i]
        class_indices[label].append(i)

    member_indices = []
    non_member_indices = []

    for class_id in range(num_classes):
        current_class_indices = class_indices[class_id]
        random.shuffle(current_class_indices)

        current_member_count = samples_per_class_member + (1 if class_id < member_remainder else 0)
        current_non_member_count = samples_per_class_non_member + (1 if class_id < non_member_remainder else 0)

        member_indices.extend(current_class_indices[:current_member_count])
        non_member_indices.extend(current_class_indices[current_member_count : current_member_count + current_non_member_count])

    random.shuffle(member_indices)
    random.shuffle(non_member_indices)

    D_member = Subset(dataset, member_indices)
    D_non_member = Subset(dataset, non_member_indices)

    print(f"\nCreated D_member with {len(D_member)} samples (stratified, scale={scale}).")
    print(f"Created D_non_member with {len(D_non_member)} samples (stratified, scale={scale}).")

    return D_member, D_non_member

def preprocess_cifar10_dataset(data_dir='./data', num_classes=10, train_ratio=0.5, scale=1.0, download=True):
    """
    Orchestrates the entire CIFAR-10 data preprocessing pipeline:
    1. Loads raw CIFAR-10 data (train and test combined).
    2. Splits the combined dataset into D_member_raw and D_non_member_raw using stratified sampling.
    3. Calculates mean and standard deviation *only* from D_member_raw.
    4. Creates a normalization transform using the calculated stats.
    5. Applies this normalization transform to both D_member_raw and D_non_member_raw.

    Args:
        data_dir (str): Directory to store/load CIFAR-10 data.
        num_classes (int): The number of classes in the dataset (e.g., 10 for CIFAR-10).
        train_ratio (float): Ratio of member samples within the scaled data pool (for D_member).
        scale (float): Fraction of the total CIFAR-10 dataset to use.
        download (bool): Whether to download CIFAR-10 if not found.

    Returns:
        tuple: (D_member_normalized, D_non_member_normalized)
               These are PyTorch Dataset objects.
    """
    print("\n--- Starting CIFAR-10 Preprocessing Pipeline ---")
     
    full_cifar10_dataset = _load_cifar10(data_dir, download)
    
    D_member_raw, D_non_member_raw = _split_dataset(full_cifar10_dataset, train_ratio, scale, num_classes)

    mean, std = _get_mean_std(D_member_raw)

    normalize_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])

    D_member_normalized = TransformedSubset(D_member_raw, normalize_transform)
    D_non_member_normalized = TransformedSubset(D_non_member_raw, normalize_transform)

    print("\n--- CIFAR-10 Preprocessing Pipeline Complete ---")
    print(f"Final D_member dataset size (normalized): {len(D_member_normalized)} samples")
    print(f"Final D_non_member dataset size (normalized): {len(D_non_member_normalized)} samples")

    return D_member_normalized, D_non_member_normalized

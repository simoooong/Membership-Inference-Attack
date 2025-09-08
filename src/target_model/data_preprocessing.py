import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, Subset, Dataset
import os
import random
from typing import Tuple

def _load_cifar10(data_dir: str, download: bool) -> Dataset:
    """
    Loads CIFAR-10 datasets (train and test) without normalization,
    only converting images to tensors.
    """

    trainset_orig = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=download, transform=T.ToTensor())

    testset_orig = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                                download=download, transform=T.ToTensor())

    full_cifar10_dataset = ConcatDataset([trainset_orig, testset_orig])

    print(f"Loaded original CIFAR-10 training set: {len(trainset_orig)} samples")
    print(f"Loaded original CIFAR-10 testing set: {len(testset_orig)} samples")
    print(f"Combined CIFAR-10 dataset size: {len(full_cifar10_dataset)} samples")

    return full_cifar10_dataset

def _load_stl10(data_dir: str, download: bool) -> Dataset:
    """
    Loads STL-10 datasets (train and test) without normalization,
    only converting images to tensors.
    Returns the combined dataset, datapoint_size, and num_classes.
    """

    trainset_orig = torchvision.datasets.STL10(root=data_dir, split='train',
                                               download=download, transform=T.ToTensor())
    
    testset_orig = torchvision.datasets.STL10(root=data_dir, split='test',
                                              download=download, transform=T.ToTensor())
    
    full_dataset = ConcatDataset([trainset_orig, testset_orig])
    print(f"Loaded original STL-10 training set: {len(trainset_orig)} samples")
    print(f"Loaded original STL-10 testing set: {len(testset_orig)} samples")
    print(f"Combined STL-10 dataset size: {len(full_dataset)} samples")
    return full_dataset

def get_dataset_info(dataset_name: str):
    """
    Returns the appropriate data preprocessing function, datapoint_size, and num_classes
    based on the dataset name.
    """

    if dataset_name.lower() == 'cifar-10':
        return _load_cifar10, 32, 10
    elif dataset_name.lower() == 'stl-10':
        return _load_stl10, 96, 10
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Supported: 'CIFAR-10', 'STL-10'")

def _get_mean_std(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(self, subset: Dataset, transform: T.Compose):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def _split_dataset(dataset: Dataset, num_classes: int, train_ratio: float, scale: float) -> Tuple[Dataset, Dataset]:
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
               These are PyTorch Dataset objects.
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
        current_non_member_count = samples_per_class_non_member

        if non_member_remainder > 0 and (len(current_class_indices) - current_member_count - samples_per_class_non_member > 0):
            current_non_member_count += 1
            non_member_remainder -= 1

        member_indices.extend(current_class_indices[:current_member_count])
        non_member_indices.extend(current_class_indices[current_member_count : current_member_count + current_non_member_count])

    random.shuffle(member_indices)
    random.shuffle(non_member_indices)

    D_member = Subset(dataset, member_indices)
    D_non_member = Subset(dataset, non_member_indices)

    return D_member, D_non_member

def preprocess_dataset(
    dataset_name: str,
    load_fn,
    data_dir: str,
    num_classes: int,
    train_ratio: float, 
    val_ratio: float, 
    scale: float, 
    download: bool
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Orchestrates the entire data preprocessing pipeline for a given dataset:
    1. Loads raw data (train and test combined) for the specified dataset.
    2. Scales the total dataset size if `scale` < 1.0.
    3. Splits the scaled dataset into D_member_raw and D_non_member_raw using stratified sampling.
    4. Splits D_member_raw into D_member_train and D_member_val using stratified sampling.
    5. Calculates mean and standard deviation *only* from D_member_train.
    6. Creates a normalization transform using the calculated stats.
    7. Applies this normalization transform to D_member_train, D_member_val, and D_non_member_raw.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'CIFAR-10', 'STL-10', 'Food-101').
        data_dir (str): Directory to store/load dataset data.
        train_ratio (float): Ratio of member samples within the scaled data pool (for D_member_raw).
        val_ratio (float): Ratio of validation samples within D_member_raw (for D_member_val).
        scale (float): Fraction of the total dataset to use.
        download (bool): Whether to download the dataset if not found.

    Returns:
        tuple: (D_member_train_normalized, D_member_val_normalized, D_non_member_normalized)
               These are PyTorch Dataset objects.
    """
    print(f"\n--- Starting Preprocessing Pipeline for {dataset_name} ---")
    
    full_cifar10_dataset = load_fn(data_dir, download)
    
    D_member_raw, D_non_member_raw = _split_dataset(full_cifar10_dataset, num_classes, train_ratio, scale)
    D_member_train, D_member_val = _split_dataset(D_member_raw, num_classes, 1.0 - val_ratio, 1.0)

    mean, std = _get_mean_std(D_member_train)

    normalize_transform = T.Compose([
        T.Normalize(mean, std)
    ])

    D_member_train_normalized = TransformedSubset(D_member_train, normalize_transform)
    D_member_val_normalized = TransformedSubset(D_member_val, normalize_transform)
    D_non_member_normalized = TransformedSubset(D_non_member_raw, normalize_transform)


    print("\n--- CIFAR-10 Preprocessing Pipeline Complete ---")
    print(f"Final D_member_train dataset size (normalized): {len(D_member_train_normalized)} samples")
    print(f"Final D_member_val dataset size (normalized): {len(D_member_val_normalized)} samples")
    print(f"Final D_non_member dataset size (normalized): {len(D_non_member_normalized)} samples")

    return D_member_train_normalized, D_member_val_normalized, D_non_member_normalized

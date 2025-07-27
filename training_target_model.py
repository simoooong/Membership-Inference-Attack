import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import random
import numpy as np

from resnet_model import ResNet18

def set_seed(seed):
    """
    Sets the random seed for reproducibility across multiple libraries.
    """
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    print(f"Random seed set to {seed}")

def train_target_model(D_member_normalized: Dataset,
                       model_dir='./saved_models', 
                       num_classes=10, 
                       train_ratio=0.5, scale=1.0, 
                       num_epochs=50, 
                       learning_rate=0.001,
                       train_criterion=nn.CrossEntropyLoss(),
                       optimizer_class=optim.Adam,
                       batch_size=64):
    """
    Trains the ResNet-18 target model (Model M) on the D_member dataset.

    Args:
        D_member_normalized (Dataset): The preprocessed and normalized dataset for training (members).
        model_dir (str): Base directory to save trained models.
        num_classes (int): The number of classes in the dataset (e.g., 10 for CIFAR-10).
        train_ratio (float): Ratio of member samples within the scaled data pool (for filename).
        scale (float): Fraction of the total CIFAR-10 dataset to use (for filename).
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        train_criterion (nn.Module): The loss function to use for training.
        optimizer_class (torch.optim.Optimizer): The class of the optimizer to use (e.g., optim.Adam).
        batch_size (int): Batch size for DataLoader.
    """

    member_loader = DataLoader(D_member_normalized, batch_size=batch_size, shuffle=False,
                               num_workers=os.cpu_count() if os.cpu_count() else 0)

    model_m = ResNet18(num_classes=num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_m.to(device)
    print(f"Using device: {device}")

    optimizer = optimizer_class(model_m.parameters(), lr=learning_rate)

    print(f"\n--- Starting Training of Target Model (Model M) on D_member ---")
    print(f"Total Epochs: {num_epochs}, Learning Rate: {learning_rate}, Batch Size: {batch_size}")

    for epoch in range(num_epochs):
        model_m.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in member_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_m(inputs)

            loss = train_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = 100 * correct_predictions / total_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    print(f"\n--- Training Complete ---")

    scale_str = f"{scale:.2f}".replace('.', '_')
    train_ratio_str = f"{train_ratio:.2f}".replace('.', '_')
    epoch_accuracy_str = f"{epoch_accuracy:.2f}".replace('.', '_')

    model_filename = (
        f"model_M_scale{scale_str}_trainratio{train_ratio_str}_epochs{num_epochs}"
        f"_trainacc{epoch_accuracy_str}.pth" 
    )

    model_save_path = os.path.join(model_dir, model_filename)

    torch.save(model_m.state_dict(), model_save_path)
    print(f"Trained Model M saved to: {model_save_path}")

    return model_m

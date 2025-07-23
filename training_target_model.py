import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import random
import numpy as np

from resnet_model import ResNet18

def _set_seed(seed):
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

def train_target_model(D_member_normalized, model_dir='./saved_models', num_classes=10,
                       train_ratio=0.5, scale=1.0, num_epochs=50, learning_rate=0.001, batch_size=64):
    """
    Trains the ResNet-18 target model (Model M) on the D_member dataset.

    Args:
        D_member_normalized (Dataset): The preprocessed and normalized dataset for training (members).
        model_dir (str): Path to save the trained model.
        num_classes (int): The number of classes in the dataset (e.g., 10 for CIFAR-10).
        train_ratio (float): Ratio of member samples within the scaled data pool.
        scale (float): Fraction of the total CIFAR-10 dataset to use.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for DataLoader.
    """

    # Set seed for reproducibility
    SEED = 42
    _set_seed(SEED)

    member_loader = DataLoader(D_member_normalized, batch_size=batch_size, shuffle=True,
                               num_workers=os.cpu_count() if os.cpu_count() else 0)

    model_m = ResNet18(num_classes=num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_m.to(device)
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_m.parameters(), lr=learning_rate)

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

            loss = criterion(outputs, labels)
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

    os.makedirs(model_dir, exist_ok=True)

    model_filename = (
        f"model_M_scale{scale:.2f}_trainratio{train_ratio:.2f}_trainacc{epoch_accuracy:.2f}.pth"
    )
    
    model_filename = model_filename.replace('.', '_') 
    model_save_path = os.path.join(model_dir, model_filename)

    torch.save(model_m.state_dict(), model_save_path)
    print(f"Trained Model M saved to: {model_save_path}")

    return model_m



if __name__ == '__main__':
    # This block is for testing the train_target_model function directly.
    # In your full project, you would call train_target_model from a separate main script.

    # Ensure data directory exists for preprocessing
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)

    # 1. Preprocess the CIFAR-10 dataset to get normalized D_member
    # Make sure 'preprocess_data.py' is accessible and contains preprocess_cifar10_dataset
    from data_preprocessing import preprocess_cifar10_dataset

    print("\n--- Running a test training run ---")
    D_member_normalized, _ = preprocess_cifar10_dataset(
        data_dir=data_dir, 
        train_ratio=0.5, 
        scale=0.1, 
        download=True # Set to True to download if not present
    )

    # 2. Train the target model
    model_m = train_target_model(
        D_member_normalized, 
        model_dir='./test_models', 
        num_classes=10,
        train_ratio=0.5, 
        scale=0.1, 
        num_epochs=5, # Reduced epochs for a quicker test run
        learning_rate=0.001, 
        batch_size=64
    )
    print("\n--- Test training run complete. Check './saved_models' for the saved model. ---")


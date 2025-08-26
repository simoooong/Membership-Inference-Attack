import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
import copy
from typing import Type


class EarlyStopping:
    """
    Implements a robust early stopping mechanism, similar to the UP-criteria
    described in the paper. It monitors a metric (e.g., validation loss) and
    stops training only if the metric fails to improve for a specified number
    of consecutive epochs, preventing premature stopping due to noisy, jagged
    validation curves.
    """
    def __init__(self, patience: int, min_delta: float):
        """
        Initializes the early stopper.
        Args:
            patience (int): How many epochs to wait for a new best score.
                            This corresponds to 's' in the paper's UP_s criterion.
            min_delta (float): Minimum change to qualify as an improvement.
                               This handles the "jagged" nature of the error curves.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_score: float, model: nn.Module):
        """
        Checks if the current score is an improvement.
        Args:
            current_score (float): The current score of the monitored metric (e.g., validation loss).
            model (nn.Module): The current state of the model.
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_score is None or current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True

        return self.early_stop

def train_target_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model_class: Type[nn.Module],
    model_dir: str, 
    num_classes: int, 
    train_ratio: float,
    val_ratio: float,
    scale: float, 
    num_epochs: int, 
    learning_rate: float,
    train_criterion: nn.Module,
    optimizer_class: optim.Optimizer,
    batch_size: int,
    patience: int,
    min_delta: int
) -> nn.Module:
    """
    Trains a target model on the D_member training dataset, using the
    validation dataset for early stopping. The function is responsible for
    training and saving the best model based on validation loss, but not for
    final evaluation.

    Args:
        train_dataset (Dataset): The preprocessed and normalized training dataset.
        val_dataset (Dataset): The preprocessed and normalized validation dataset.
        model_class (Type[nn.Module]): The class of the neural network model to instantiate.
        model_dir (str): Base directory to save trained models.
        num_classes (int): The number of classes in the dataset (e.g., 10 for CIFAR-10).
        train_ratio (float): Ratio of member samples within the scaled data pool (for filename).
        validation_ratio (float): Ratio of D_member_normalized to use for validation (for filename).
        scale (float): Fraction of the total CIFAR-10 dataset to use (for filename).
        num_epochs (int): The maximum number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        train_criterion (nn.Module): The loss function to use for training.
        optimizer_class (torch.optim.Optimizer): The class of the optimizer to use (e.g., optim.Adam).
        batch_size (int): Batch size for DataLoader.
        patience (int): Number of epochs with no improvement on validation loss after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """
    print("\n--- Starting Training of Target Model (Model M) ---")
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=os.cpu_count() if os.cpu_count() else 0)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=os.cpu_count() if os.cpu_count() else 0)
    
    model_m = model_class(num_classes=num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_m.to(device)
    print(f"Using device: {device}")
    
    optimizer = optimizer_class(model_m.parameters(), lr=learning_rate)
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    print(f"Total Epochs (max): {num_epochs}, Learning Rate: {learning_rate}, Batch Size: {batch_size}")
    print(f"Early Stopping Patience: {patience}, Min Delta: {min_delta}")
    
    final_epochs = num_epochs
    final_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model_m.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
    
        for inputs, labels in train_loader:
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
    
        train_loss = running_loss / total_samples
        train_acc = 100 * correct_predictions / total_samples
    
        # --- VALIDATION PHASE (Used for Early Stopping) ---
        model_m.eval()
        val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
    
                outputs = model_m(inputs)
                loss = train_criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                correct_val_predictions += (predicted == labels).sum().item()
        
        val_loss /= total_val_samples
        val_acc = 100 * correct_val_predictions / total_val_samples
    
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_acc:.2f}%, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_acc:.2f}%)")
        
        if early_stopper(val_loss, model_m):
            print(f"Early stopping triggered after {early_stopper.counter} epochs with no improvement.")
            print(f"Best validation loss was: {early_stopper.best_score:.4f}")
            
            final_epochs = epoch- early_stopper.patience
            final_val_loss = early_stopper.best_score
            break
        else:
            final_epochs = epoch + 1
            final_val_loss = val_loss

    print(f"\n--- Training Complete ---")
    
    # Load the best model state before saving
    model_m.load_state_dict(early_stopper.best_model_state)
    
    # Save the best model state based on validation loss
    scale_str = f"{scale:.2f}".replace('.', '_')
    train_ratio_str = f"{train_ratio:.2f}".replace('.', '_')
    val_ratio_str = f"{val_ratio:.2f}".replace('.', '_')
    
    model_filename = (
        f"model_M_scale{scale_str}_trainratio{train_ratio_str}_valratio{val_ratio_str}_epochs{final_epochs}"
        f"_bestloss{final_val_loss:.4f}".replace('.', '_') + ".pth"
    )
    
    model_save_path = os.path.join(model_dir, model_filename)
    torch.save(model_m.state_dict(), model_save_path)
    print(f"Trained Model M saved to: {model_save_path}")

    return model_m
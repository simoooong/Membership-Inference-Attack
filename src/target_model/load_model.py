import torch
import torch.nn as nn
import os
import re
from typing import Type, Tuple, Optional

def load_model(
    model_dir: str, 
    model_class: Type[nn.Module],
    num_classes: int, 
    dataset_name: str,
    scale: float,
    train_ratio: float,
    val_ratio: float
) -> Optional[Tuple[nn.Module, float, float, int]]: # Updated return type
    """
    Finds a saved model in the specified directory matching the given parameters
    and loads its state_dict into a provided model class. It also extracts
    the best validation accuracy, best validation loss, and epochs trained from the filename.

    Args:
        model_dir (str): Directory where models are saved.
        model_class (Type[nn.Module]): The class of the model to instantiate.
        num_classes (int): Number of classes the model was trained for.
        train_ratio (float): Train ratio parameter used during training.
        val_ratio (float): Validation ratio parameter used during training.
        scale (float): Scale parameter used during training.

    Returns:
        Optional[Tuple[nn.Module, float, float, int]]: A tuple containing the loaded model,
                                                        its best validation accuracy, best validation loss,
                                                        and epochs trained. Returns None if no matching model is found.
    """

    # FIX: Ensure consistent string formatting for floats to match saved filenames
    scale_str = f"{scale}".replace('.', '_')
    train_ratio_str = f"{train_ratio}".replace('.', '_')
    val_ratio_str = f"{val_ratio}".replace('.', '_')

    # FIX: Pattern matches your stated filename format for _val_loss and _val_acc
    # model_M_scale{scale}_trainratio{train_ratio}_valratio{val_ratio}_epochs{epochs}_val_loss{loss}_val_acc{acc}.pth
    pattern = (
        r"dataset" + re.escape(dataset_name) +
        r"_scale" + re.escape(scale_str) +
        r"_trainratio" + re.escape(train_ratio_str) +
        r"_valratio" + re.escape(val_ratio_str) +
        r"_epochs(\d+)" +         # Capture epochs_trained (Group 1)
        r"_val_loss(\d+_\d+)" +   # Capture best_val_loss (Group 2)
        r"_val_acc(\d+_\d+)\.pth" # Capture best_val_accuracy (Group 3)
    )

    matching_models = []

    if not os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' not found.")
        return None, None, None, None

    for filename in os.listdir(model_dir):
        if re.match(pattern, filename):
            matching_models.append(filename)

    if not matching_models:
        print(f"No model found in '{model_dir}' matching scale={scale:.2f}, train_ratio={train_ratio:.2f}, val_ratio={val_ratio:.2f}.")
        return None, None, None, None
    
    print(matching_models)

    # Logic to pick the best model from matches (e.g., lowest loss if multiple matches with same params)
    best_model_file = None
    lowest_loss_found = float('inf') # Use a distinct name to avoid confusion with the return value

    for model_file in matching_models:
        # FIX: Regex to parse specifically the loss for comparison - now matches _val_loss
        match_loss_for_comparison = re.search(r"_val_loss(\d+_\d+)", model_file)
        if match_loss_for_comparison:
            loss_str_parsed = match_loss_for_comparison.group(1).replace("_", ".")
            current_loss = float(loss_str_parsed)

            if current_loss < lowest_loss_found:
                lowest_loss_found = current_loss
                best_model_file = model_file

    if best_model_file is None:
        print("Could not parse loss from any matching model filenames for comparison. Aborting.")
        return None, None, None, None

    model_path = os.path.join(model_dir, best_model_file)
    print(f"Loading best model from: {model_path}")

    # Extract all metrics from the chosen best_model_file using the full pattern
    match_final = re.match(pattern, best_model_file)
    if not match_final:
        print(f"Error: Could not re-match full pattern for best model file {best_model_file}. Filename structure might be unexpected.")
        return None, None, None, None

    epochs_trained = int(match_final.group(1))
    best_val_loss_str = match_final.group(2).replace("_", ".")
    best_val_loss = float(best_val_loss_str)
    best_val_accuracy_str = match_final.group(3).replace("_", ".")
    best_val_accuracy = float(best_val_accuracy_str)

    model = model_class(num_classes = num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set to eval mode by default when loading

    print(f"Model loaded successfully (Epochs: {epochs_trained}, Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_accuracy:.2f}%).")
    
    return model, best_val_accuracy, best_val_loss, epochs_trained

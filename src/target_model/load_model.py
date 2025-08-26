import torch
import torch.nn as nn
import os
import re
from typing import Type, Optional

def load_model(
    model_dir: str, 
    model_class: Type[nn.Module],
    num_classes: int, 
    train_ratio: float,
    val_ratio: float,
    scale: float
) -> Optional[nn.Module]:
    """
    Finds a saved model in the specified directory matching the given parameters
    and loads its state_dict into a provided model class.

    Args:
        model_dir (str): Directory where models are saved.
        model_class (Type[nn.Module]): The class of the model to instantiate.
        num_classes (int): Number of classes the model was trained for.
        scale (float): Scale parameter used during training.
        train_ratio (float): Train ratio parameter used during training.
    """

    scale_str = f"{scale:.2f}".replace('.', '_')
    train_ratio_str = f"{train_ratio:.2f}".replace('.', '_')
    val_ratio_str = f"{val_ratio:.2f}".replace('.', '_')

    pattern = (
        r"model_M_scale" + re.escape(scale_str) +
        r"_trainratio" + re.escape(train_ratio_str) +
        r"_valratio" + re.escape(val_ratio_str) +
        r"_epochs(\d+)" + 
        r"_bestloss(\d+_\d+)\.pth"
    )


    matching_models = []

    for filename in os.listdir(model_dir):
        if re.match(pattern, filename):
            matching_models.append(filename)

    if not matching_models:
        print(f"No model found in '{model_dir}' matching scale={scale:.2f} and train_ratio={train_ratio:.2f}.")
        return None
    
    best_model_file = None
    lowest_loss = float('inf')

    for model_file in matching_models:
        match = re.search(r"_bestloss(\d+_\d+)\.pth", model_file)
        if match:
            loss_str = match.group(1).replace("_", ".")
            current_loss = float(loss_str)

            if current_loss < lowest_loss:
                lowest_loss = current_loss
                best_model_file = model_file


    model_path = os.path.join(model_dir, best_model_file)
    print(f"Loading best model (lowest loss {lowest_loss:.4f}) from: {model_path}")

    model = model_class(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print(f"Model loaded successfully.")
    return model
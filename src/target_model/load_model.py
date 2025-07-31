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

    pattern = (
        r"model_M_scale" + re.escape(f"{scale:.2f}").replace('.', '_') +
        r"_trainratio" + re.escape(f"{train_ratio:.2f}").replace('.', '_') +
        r"_epochs(\d+)" + 
        r"_trainacc(\d+_\d+)\.pth"
    )

    matching_models = []

    for filename in os.listdir(model_dir):
        if re.match(pattern, filename):
            matching_models.append(filename)

    if not matching_models:
        print(f"No model found in '{model_dir}' matching scale={scale:.2f} and train_ratio={train_ratio:.2f}.")
        return None
    
    best_model_file = None
    best_accuracy = -1.0

    for model_file in matching_models:
        match = re.search(r"_trainacc(\d+_\d+)\.pth", model_file)
        if match:
            acc_str = match.group(1).replace("_", ".")
            current_acc = float(acc_str)

            if current_acc > best_accuracy:
                best_accuracy = current_acc
                best_model_file = model_file


    model_path = os.path.join(model_dir, best_model_file)
    print(f"Loading model from: {model_path}")

    model = model_class(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print(f"Model loaded successfully.")
    return model
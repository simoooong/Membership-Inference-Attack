import torch
import random
import numpy as np

def set_seed(seed: int):
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

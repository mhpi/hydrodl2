import torch
from typing import List

def change_param_range(param: torch.Tensor, bounds: List[float]) -> torch.Tensor:
    """Change the range of a parameter to the specified bounds."""
    out = param * (bounds[1] - bounds[0]) + bounds[0]
    return out
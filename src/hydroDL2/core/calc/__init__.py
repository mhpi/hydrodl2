import torch
from typing import List



def change_param_range(param: torch.Tensor, bounds: List[float]) -> torch.Tensor:
    """Change the range of a parameter to the specified bounds."""
    out = param * (bounds[1] - bounds[0]) + bounds[0]
    return out


def param_bounds_2D(params: torch.Tensor, num: int, bounds: List, ndays: int,
                    nmul: int) -> torch.Tensor:
    """Convert a 2D parameter array to a 3D parameter array.
    
    Parameters
    ----------
    params : torch.Tensor
        The 2D parameter array.
    num : int
        The number of parameters.
    bounds : List[float]
        The parameter bounds.
    ndays : int
        The number of days.
    nmul : int
        The number of parallel models.

    Returns
    -------
    out : torch.Tensor
        The 3D parameter array.
    """
    out_temp = (
            params[:, num * nmul: (num + 1) * nmul]
            * (bounds[1] - bounds[0])
            + bounds[0]
    )
    out = out_temp.unsqueeze(0).repeat(ndays, 1, 1).reshape(
        ndays, params.shape[0], nmul
    )
    return out
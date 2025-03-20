
import torch


def change_param_range(param: torch.Tensor, bounds: list[float]) -> torch.Tensor:
    """Change the range of a parameter to the specified bounds.
    
    Parameters
    ----------
    param
        The parameter.
    bounds
        The parameter bounds.
    
    Returns
    -------
    out
        The parameter with the specified bounds.
    """
    out = param * (bounds[1] - bounds[0]) + bounds[0]
    return out


def param_bounds_2D(
    params: torch.Tensor,
    num: int,
    bounds: list,
    ndays: int,
    nmul: int
) -> torch.Tensor:
    """Convert a 2D parameter array to a 3D parameter array.
    
    Parameters
    ----------
    params
        The 2D parameter array.
    num
        The number of parameters.
    bounds
        The parameter bounds.
    ndays
        The number of days.
    nmul
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

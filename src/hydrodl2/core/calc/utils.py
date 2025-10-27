"""
Note: If adding new public methods, please add them to __all__
at the top of the file and in calc/__init__.py.
"""

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
    torch.Tensor
        The parameter with the specified bounds.
    """
    return param * (bounds[1] - bounds[0]) + bounds[0]


def param_bounds_2d(
    params: torch.Tensor,
    num: int,
    bounds: list,
    ndays: int,
    nmul: int,
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
    torch.Tensor
        The 3D parameter array.
    """
    out_temp = (
        params[:, num * nmul : (num + 1) * nmul] * (bounds[1] - bounds[0]) + bounds[0]
    )
    return (
        out_temp.unsqueeze(0).repeat(ndays, 1, 1).reshape(ndays, params.shape[0], nmul)
    )

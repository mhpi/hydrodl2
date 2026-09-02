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


def trim_warmup(
    outputs: dict[str, torch.Tensor],
    pred_cutoff: int,
    nsteps: int,
) -> dict[str, torch.Tensor]:
    """Drop warm-up timesteps from a model's time-major outputs.

    Models spin up their internal states over a warm-up window that should
    not appear in predictions. Depending on the warm-up strategy, that window
    is either simulated separately (and never enters the outputs) or simulated
    inline and removed afterwards. This helper performs the removal so that a
    model's ``forward`` always returns the same number of timesteps either way.

    Only time-major tensors are trimmed — those whose leading dimension is
    ``nsteps``. Outputs that have already collapsed the time axis (a baseflow
    index summed over time, for instance) are passed through untouched, so
    callers do not have to maintain a list of exceptions.

    Parameters
    ----------
    outputs
        Dictionary of model outputs.
    pred_cutoff
        Number of leading timesteps to drop. Values <= 0 are a no-op.
    nsteps
        Length of the simulated window, used to identify time-major tensors.

    Returns
    -------
    dict[str, torch.Tensor]
        Outputs with the warm-up period removed from every time-major tensor.
    """
    if pred_cutoff <= 0:
        return outputs

    trimmed = {}
    for key, value in outputs.items():
        is_time_major = (
            torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == nsteps
        )
        trimmed[key] = value[pred_cutoff:] if is_time_major else value
    return trimmed

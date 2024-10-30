import json
from pathlib import Path
from typing import Any, List
from torch.nn import Module

__all__ = ["get_models",  "load_model"]


def get_models() -> List:
    """Get a list of all available models in the database.
    
    Returns
    -------
    models : List
        A list of all available models in the database.
    """
    models = ['HBV', 'PRMS']
    return models


def load_model(model: str) -> Module:
    """ Load a model from the database.

    Parameters
    ----------
    model : str
        The name of the model to load.
    """
    if model == 'HBV':
        from hydrodl2.models.hbv import HBV
        return HBV
    elif model == 'PRMS':
        from hydrodl2.models.prms import PRMS
        return PRMS
    else:
        raise ValueError(f"Model {model} not found.")


def load_record(path: Path) -> Record:
    """Load a record from a json file

    Parameters
    ----------
    path : Path
        The path to the json file

    Returns
    -------
    Record
        The record object
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Record(**data)

from typing import List
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

    NOTE: this is hacked just so we can test hydro model imports within dMG
    tutorials. Full implementation should be loader that is aware of all models
    within the hydroDL2.models module and can load them dynamically.

    Parameters
    ----------
    model : str
        The name of the model to load.
    """
    if model == 'HBV':
        from hydroDL2.models.hbv import hbv
        return hbv.HBVMulTDET
    if model == 'HBV':
        from hydroDL2.models.hbv import hbv_capillary
        return hbv_capillary.HBVMulTDET
    elif model == 'PRMS':
        from hydroDL2.models.prms import prms_marrmot
        return prms_marrmot.PRMS
    else:
        raise ValueError(f"Model {model} not found.")

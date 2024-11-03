from typing import List, Dict
from torch.nn import Module
from pathlib import Path
import importlib.util
import os

from hydroDL2.api import get_model_dir
from hydroDL2.core.utils import get_directories, get_files

__all__ = ["available_models",  "load_model"]



def available_models() -> Dict[str, List[str]]:
    """Identify and list all available models in the database.
    
    Returns
    -------
    models : List
        A list of all available models in the database.
    """
    # Path to the models directory
    model_dir = get_model_dir()

    dirs = []
    models = {}

    dirs, _ = get_directories(model_dir)
    for dir in dirs:
        _, file_names = get_files(dir)
        models[dir.name] = file_names
    
    return models


def load_model(model: str, ver_name: str = None) -> Module:
    """Load a model from the models directory.

    Each model file in `models/` directory should only contain one model class.

    Parameters
    ----------
    model : str
        The model name.
    ver_name : str, optional
        The version name (class) of the model to load within the model file.
    
    Returns
    -------
    Module
        The uninstantiated model.
    """
    # Path to the models directory
    parent_dir = get_model_dir()

    # Construct file path
    model_dir = model.split('_')[0].lower()
    model_subpath = os.path.join(model_dir, f'{model.lower()}.py')
    
    # Path to the module file in the models directory
    source = os.path.join(parent_dir, model_subpath)
    
    # Load the model dynamically as a module.
    try:
        spec = importlib.util.spec_from_file_location(model, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ImportError(f"Model '{model}' not found.")
    
    # Retrieve the version name if specified, otherwise get the first class in the module
    if ver_name:
        cls = getattr(module, ver_name)
    else:
        # Find the first class in the module (this may not always be accurate)
        classes = [attr for attr in dir(module) if isinstance(getattr(module, attr), type)]
        if not classes:
            raise ImportError(f"Model version '{model}' not found.")
        cls = getattr(module, classes[0])
    
    return cls

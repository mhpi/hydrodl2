"""
Note: If adding new public methods, please add them to __all__ 
at the top of the file and in api/__init__.py.
"""
import importlib.util
import os

from torch.nn import Module

from hydroDL2.core.utils import *

__all__ = [
    'available_models',
    'available_modules',
    'load_model',
    'load_module'
]


def available_models() -> dict[str, list[str]]:
    """Identify and list all available models in hydroDL2.
    
    Returns
    -------
    list
        A list of available models.
    """
    # Path to the models directory
    model_dir = _get_dir('models')
    models = {}

    dirs, _ = get_model_dirs(model_dir)
    for dir in dirs:
        _, file_names = get_model_files(dir)
        models[dir.name] = file_names
    
    return models


def _list_available_models() -> list[str]:
    """List all available models in hydroDL2 without the dict nesting
    of available_models().
        
    Returns
    -------
    list
        A list of available models.
    """
    model_dir = _get_dir('models')
    models = []
    dirs, _ = get_model_dirs(model_dir)
    for dir in dirs:
        _, file_names = get_model_files(dir)
        for file in file_names:
            models.append(file)  

    return models


def available_modules() -> dict[str, list[str]]:
    """Identify and list all available modules in the hydroDL2.
    
    Returns
    -------
    list
        A list of available modules.
    """
    # Path to the modules directory
    model_dir = _get_dir('modules')
    modules = {}

    dirs, _ = get_model_dirs(model_dir)
    for dir in dirs:
        _, file_names = get_model_files(dir)
        modules[dir.name] = file_names
    
    return modules


def load_model(model: str, ver_name: str = None) -> Module:
    """Load a model from the models directory.

    Each model file in `models/` directory should only contain one model class.

    Parameters
    ----------
    model
        The model name.
    ver_name
        The version name (class) of the model to load within the model file.
    
    Returns
    -------
    Module
        The uninstantiated model.
    """
    # Path to the models directory
    parent_dir = _get_dir('models')

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
        classes = [
            attr for attr in dir(module)
            if isinstance(getattr(module, attr), type) and attr != 'Any'
        ]
        if not classes:
            raise ImportError(f"Model version '{model}' not found.")
        cls = getattr(module, classes[0])
    
    return cls


def load_module():
    """Load a module from the modules directory."""
    raise NotImplementedError("This function is not yet implemented.")

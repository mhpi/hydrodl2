"""
Note: If adding new public methods, please add them to __all__
at the top of the file and in api/__init__.py.
"""
import importlib.util
import logging
import os
import re

from torch.nn import Module

from hydroDL2.core.utils import _get_dir, get_model_dirs, get_model_files

log = logging.getLogger("hydroDL2")


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
    parent_dir = _get_dir('models')

    if ver_name is None:
        ver_name = model  # Default to the model name if no version is specified

    # Construct file path
    model = re.sub(r'([a-z])([A-Z])', r'\1_\2', model).lower() # Convert camelCase to snake_case
    model_dir = model.split('_')[0].lower() # Model class name is first word in snake_case
    model_subpath = os.path.join(model_dir, f'{model}.py')
    
    # Path to the module file in the model directory
    source = os.path.join(parent_dir, model_subpath)
    
    # Load the model dynamically as a module
    try:
        spec = importlib.util.spec_from_file_location(model, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as e:
        raise ImportError(f"Model '{model}' not found.") from e
    
    # Retrieve version name if possible, otherwise get first class in module
    try:
        cls = getattr(module, ver_name)
    except AttributeError as e:
        # Find first class in module (NOTE: not guaranteed accurate)
        classes = [
            attr for attr in dir(module)
            if isinstance(getattr(module, attr), type) and attr != 'Any'
        ]
        if not classes:
            raise ImportError(f"Model version '{model}' not found.") from e

        log.warning(
            f"Model class '{ver_name}' not found in module '{module.__file__}'. "
            f"Falling back to the first available: '{classes[0]}'."
        )
        cls = getattr(module, classes[0])
    
    return cls


def load_module():
    """Load a module from the modules directory."""
    raise NotImplementedError("This function is not yet implemented.")

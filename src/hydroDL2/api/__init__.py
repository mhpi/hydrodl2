from pathlib import Path
import os



def get_model_dir() -> Path:
    """Get the path to the models directory.

    Using this helps avoid path errors when debugging vs. use as a package.

    Returns
    -------
    Path
        Model directory path.
    """
    dir = Path('./models')

    if not os.path.exists(dir):
        dir = Path(os.path.dirname(os.path.abspath(__file__)))
        dir = dir.parent / 'models'
    
    return dir


def get_module_dir() -> Path:
    """Get the path to the modules directory.

    Using this helps avoid path errors when debugging vs. use as a package.

    Returns
    -------
    Path
        Module directory path.
    """
    dir = Path('./modules')

    if not os.path.exists(dir):
        dir = Path(os.path.dirname(os.path.abspath(__file__)))
        dir = dir.parent / 'modules'
    
    return dir

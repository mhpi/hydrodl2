from pathlib import Path
import os



def get_model_dir() -> Path:
    """Get the path to the models directory.

    This is function helps avoid path errors when debugging vs production work.
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

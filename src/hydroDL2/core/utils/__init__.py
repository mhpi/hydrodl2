import os
from pathlib import Path
from typing import List
from typing import Union



def get_directories(directory: Union[Path, str]) -> tuple[List[Path], List[str]]:
    """Get all subdirectories in a given directory.
    
    Parameters
    ----------
    directory : Path or str
        The parent directory.
    """
    if isinstance(directory, str):
        directory = Path(directory)
    
    dirs = []
    dir_names = []
    avoid_list = ['__pycache__']

    for item in directory.iterdir():
        if item.is_dir() and (item.name not in avoid_list):
            dirs.append(item)
            dir_names.append(item.name)
    return dirs, dir_names


def get_files(directory: Union[Path, str]) -> tuple[List[Path], List[str]]:
    """Get all files in a given directory.
    
    Parameters
    ----------
    directory : Path or str
        The parent directory.
    """
    if isinstance(directory, str):
        directory = Path(directory)
    
    files = []
    file_names = []
    avoid_list = ['__init__', '.DS_Store', 'README.md', '.git']
    
    for item in directory.iterdir():
        if item.is_file() and (item.name not in avoid_list):
            files.append(item)

            # Remove file extension
            name = os.path.splitext(item.name)[0]
            file_names.append(name)
    return files, file_names

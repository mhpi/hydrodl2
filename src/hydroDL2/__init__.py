# from hydroDL2._version import __version__
from hydroDL2.api.methods import (available_models, available_modules,
                                  load_model, load_module)
from hydroDL2.models import *

# In case setuptools scm says version is 0.0.0
# assert not __version__.startswith("0.0.0")

# List of all modules to be accessible accross subpackage.
__all__ = ['available_models',
           'available_modules',
           'load_model',
           'load_module'
           ]

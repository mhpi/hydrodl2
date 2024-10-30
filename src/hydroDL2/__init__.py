from hydroDL2._version import __version__
from hydroDL2.api.methods import get_models, load_model

# In case setuptools scm says version is 0.0.0
assert not __version__.startswith("0.0.0")

__all__ = [
    "get_models",
    "load_model",
]

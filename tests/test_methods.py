"""
Test API methods.

Tests will run with pytest when pushed to remote, and can also be run manually
with:
        $ pytest tests/test_methods.py
"""
import pytest
from torch.nn import Module

from hydroDL2 import (available_models, load_model,
                      )
from hydroDL2.api.methods import _list_available_models


def test_available_models():
    models = available_models()
    assert isinstance(models, dict)
    assert all(isinstance(v, list) for v in models.values())
    assert all(isinstance(v, str) for v in models.keys())
    assert len(models) > 0


@pytest.mark.parametrize('model', _list_available_models())
def test_load_model(model):
    loaded_model = load_model(model)
    assert loaded_model is not None, f"Failed to load '{model}'."

    # Check that the loaded model is a class.
    assert isinstance(loaded_model, type), f"Loaded '{model}' is not a class."

    # Check that the model is a subclass of torch.nn.Module.
    assert issubclass(loaded_model, Module)


@pytest.mark.parametrize("model, ver_name", [("prms", "PRMS")])
def test_load_model_with_version(model, ver_name):
    loaded_model = load_model(model, ver_name=ver_name)
    assert loaded_model is not None, f"Failed to load model '{model}' with version '{ver_name}'."
    assert isinstance(loaded_model, type), f"Loaded '{model}' with version '{ver_name}' is not a class."

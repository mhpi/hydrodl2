"""
Test API and utility methods.

Coverage:
- Model loading (load_model, available_models)
- Parameter range scaling (change_param_range)
- Unit hydrograph gamma distribution (uh_gamma)
- Unit hydrograph convolution (uh_conv)
"""

import pytest
import torch
from torch.nn import Module

from hydrodl2 import available_models, load_model
from hydrodl2.api.methods import _list_available_models
from hydrodl2.core.calc import change_param_range, uh_conv, uh_gamma


def test_available_models():
    models = available_models()
    assert isinstance(models, dict)
    assert all(isinstance(v, list) for v in models.values())
    assert all(isinstance(v, str) for v in models.keys())
    assert len(models) > 0


@pytest.mark.parametrize('model', _list_available_models())
def test_load_model(model):
    if model == 'hbv_adj':
        pytest.skip("Skipping 'hbv_adj' model due to known issue.")

    loaded_model = load_model(model)
    assert loaded_model is not None, f"Failed to load '{model}'."

    # Check that the loaded model is a class.
    assert isinstance(loaded_model, type), f"Loaded '{model}' is not a class."

    # Check that the model is a subclass of torch.nn.Module.
    assert issubclass(loaded_model, Module)


@pytest.mark.parametrize("model, ver_name", [("hbv", "Hbv")])
def test_load_model_with_version(model, ver_name):
    loaded_model = load_model(model, ver_name=ver_name)
    assert loaded_model is not None, (
        f"Failed to load model '{model}' with version '{ver_name}'."
    )
    assert isinstance(loaded_model, type), (
        f"Loaded '{model}' with version '{ver_name}' is not a class."
    )


class TestChangeParamRange:
    """Verify change_param_range scales parameters correctly."""

    def test_basic_scaling(self):
        result = change_param_range(torch.tensor([0.5]), [0.0, 10.0])
        assert torch.allclose(result, torch.tensor([5.0]))

    def test_lower_bound(self):
        result = change_param_range(torch.tensor([0.0]), [2.0, 8.0])
        assert torch.allclose(result, torch.tensor([2.0]))

    def test_upper_bound(self):
        result = change_param_range(torch.tensor([1.0]), [2.0, 8.0])
        assert torch.allclose(result, torch.tensor([8.0]))

    def test_negative_bounds(self):
        result = change_param_range(torch.tensor([0.5]), [-2.5, 2.5])
        assert torch.allclose(result, torch.tensor([0.0]))

    def test_preserves_shape(self):
        param = torch.rand(10, 5, 3)
        result = change_param_range(param, [0.0, 100.0])
        assert result.shape == param.shape

    def test_gradient_flows(self):
        param = torch.tensor([0.5], requires_grad=True)
        result = change_param_range(param, [0.0, 10.0])
        result.backward()
        assert param.grad is not None


class TestUhGamma:
    """Verify unit hydrograph gamma distribution calculations."""

    def test_output_shape(self):
        a = torch.ones(15, 5, 1) * 1.0
        b = torch.ones(15, 5, 1) * 2.0
        w = uh_gamma(a, b, lenF=10)
        assert w.shape == (10, 5, 1)

    def test_sums_to_one(self):
        a = torch.ones(15, 5, 1) * 1.5
        b = torch.ones(15, 5, 1) * 3.0
        w = uh_gamma(a, b, lenF=15)
        sums = w.sum(dim=0)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_non_negative(self):
        a = torch.ones(10, 3, 2) * 2.0
        b = torch.ones(10, 3, 2) * 1.5
        w = uh_gamma(a, b, lenF=10)
        assert (w >= 0).all()


class TestUhConv:
    """Verify unit hydrograph convolution calculations."""

    def test_output_shape(self):
        x = torch.ones(3, 1, 20)
        UH = torch.ones(3, 1, 5) / 5.0
        y = uh_conv(x, UH)
        assert y.shape == x.shape

    def test_identity_convolution(self):
        x = torch.rand(2, 1, 15)
        UH = torch.zeros(2, 1, 5)
        UH[:, :, 0] = 1.0
        y = uh_conv(x, UH)
        assert torch.allclose(y, x, atol=1e-5)

    def test_delay_by_one(self):
        x = torch.zeros(1, 1, 10)
        x[0, 0, 3] = 1.0
        UH = torch.zeros(1, 1, 5)
        UH[0, 0, 1] = 1.0
        y = uh_conv(x, UH)
        assert y[0, 0, 4].item() > 0.9

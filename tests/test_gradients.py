"""
Tests for gradient flow and backpropagation.

Coverage:
- Backward pass execution
- Gradient finiteness
- Non-zero gradients
- Gradient flow w.r.t. forcing inputs

NOTE: Tests are parametrized over Hbv, Hbv_1_1p, Hbv_2, and Hbv_2_hourly
via the model_setup fixture.
"""

import torch

from hydrodl2 import load_model
from tests import DEVICE, NGRID, NSTEPS, SEED, _hbv_config_dict, make_forcing


class TestGradientFlow:
    """Test gradient flow through the differentiable model pipeline."""

    def test_backward_runs(self, model_setup):
        model, x_dict, params, sf_key = model_setup
        if isinstance(params, list):
            for p in params:
                p.requires_grad_(True)
        else:
            params.requires_grad_(True)

        result = model(x_dict, params)
        loss = result[sf_key].sum()
        loss.backward()

        if isinstance(params, list):
            for i, p in enumerate(params):
                assert p.grad is not None, f"No gradient for params[{i}]"
        else:
            assert params.grad is not None

    def test_gradients_finite(self, model_setup):
        model, x_dict, params, sf_key = model_setup
        if isinstance(params, list):
            for p in params:
                p.requires_grad_(True)
        else:
            params.requires_grad_(True)

        result = model(x_dict, params)
        loss = result[sf_key].sum()
        loss.backward()

        if isinstance(params, list):
            for i, p in enumerate(params):
                assert torch.isfinite(p.grad).all(), (
                    f"Non-finite gradient for params[{i}]"
                )
        else:
            assert torch.isfinite(params.grad).all()

    def test_gradients_nonzero(self, model_setup):
        model, x_dict, params, sf_key = model_setup
        if isinstance(params, list):
            for p in params:
                p.requires_grad_(True)
        else:
            params.requires_grad_(True)

        result = model(x_dict, params)
        loss = result[sf_key].sum()
        loss.backward()

        if isinstance(params, list):
            assert any(p.grad.abs().max() > 0 for p in params)
        else:
            assert params.grad.abs().max() > 0

    def test_gradient_wrt_forcing(self):
        torch.manual_seed(SEED)
        Hbv = load_model('hbv')
        config = _hbv_config_dict(dynamic_params=['parBETA'])
        model = Hbv(config, device=DEVICE)
        x_phy = make_forcing(NSTEPS, NGRID)
        x_phy.requires_grad_(True)
        x_dict = {'x_phy': x_phy}
        params = torch.randn(NSTEPS, NGRID, model.learnable_param_count)
        result = model(x_dict, params)
        loss = result['streamflow'].sum()
        loss.backward()
        assert x_phy.grad is not None
        assert torch.isfinite(x_phy.grad).all()

"""
Tests for physical constraints in model outputs.

Coverage:
- Non-negative streamflow and flow components
- Evaporation factor bounds [0, 1]
- AET <= PET constraint
- Non-negative SWE
- Descaled parameter bounds
"""

import pytest
import torch

from hydrodl2 import load_model
from tests import SEED, NSTEPS, NGRID, DEVICE


class TestPhysicalConstraints:
    """Verify that model outputs obey physical constraints."""

    def test_streamflow_non_negative(self, model_setup):
        model, x_dict, params, sf_key = model_setup
        result = model(x_dict, params)
        assert (result[sf_key] >= 0).all(), "Streamflow should be non-negative"

    @pytest.mark.parametrize("key", ['srflow', 'ssflow', 'gwflow'])
    def test_flow_components_non_negative(self, simple_model_setup, key):
        model, x_dict, params = simple_model_setup
        result = model(x_dict, params)
        if result[key] is not None:
            assert (result[key] >= 0).all(), f"'{key}' should be non-negative"

    def test_evapfactor_bounded(self, simple_model_setup):
        model, x_dict, params = simple_model_setup
        result = model(x_dict, params)
        ef = result['evapfactor']
        assert (ef >= 0).all() and (ef <= 1.0 + 1e-5).all()

    def test_aet_le_pet(self, simple_model_setup):
        model, x_dict, params = simple_model_setup
        result = model(x_dict, params)
        assert (result['AET_hydro'] <= result['PET_hydro'] + 1e-3).all()

    def test_swe_non_negative(self, simple_model_setup):
        model, x_dict, params = simple_model_setup
        result = model(x_dict, params)
        assert (result['SWE'] >= 0).all()

    def test_descaled_params_within_bounds(self):
        torch.manual_seed(SEED)
        Hbv = load_model('hbv')
        model = Hbv(device=DEVICE)
        raw = torch.randn(NSTEPS, NGRID, model.learnable_param_count)
        phy_params, _ = model._unpack_parameters(raw)
        descaled = model._descale_phy_parameters(phy_params, dy_list=[])
        for name, bounds in model.parameter_bounds.items():
            vals = descaled[name]
            assert (vals >= bounds[0] - 1e-5).all(), f"'{name}' below lower bound"
            assert (vals <= bounds[1] + 1e-5).all(), f"'{name}' above upper bound"

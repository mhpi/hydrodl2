"""
Tests for model forward-pass behavior.

Coverage:
- Output structure (dict type, keys)
- Output tensor shapes (streamflow, BFI)
- Output validity (finite values)
- Forward-pass determinism

NOTE: Tests are parametrized over Hbv, Hbv_1_1p, Hbv_2, and Hbv_2_hourly
via the model_setup and simple_model_setup fixtures.
"""

import torch

from tests import NGRID, NSTEPS


class TestForwardPass:
    """Verify forward-pass output structure and validity."""

    def test_returns_dict(self, model_setup):
        model, x_dict, params, _ = model_setup
        result = model(x_dict, params)
        assert isinstance(result, dict)

    def test_hbv_output_keys(self, simple_model_setup):
        model, x_dict, params = simple_model_setup
        result = model(x_dict, params)
        expected = {
            'streamflow',
            'srflow',
            'ssflow',
            'gwflow',
            'AET_hydro',
            'PET_hydro',
            'SWE',
            'streamflow_no_rout',
            'srflow_no_rout',
            'ssflow_no_rout',
            'gwflow_no_rout',
            'recharge',
            'excs',
            'evapfactor',
            'tosoil',
            'percolation',
            'BFI',
        }
        if 'capillary' in model.flux_names:
            expected.add('capillary')
        assert set(result.keys()) == expected

    def test_streamflow_shape(self, model_setup):
        model, x_dict, params, sf_key = model_setup
        result = model(x_dict, params)
        sf = result[sf_key]
        assert sf.ndim == 3
        assert sf.shape[0] == NSTEPS
        assert sf.shape[2] == 1

    def test_bfi_shape(self, simple_model_setup):
        model, x_dict, params = simple_model_setup
        result = model(x_dict, params)
        assert result['BFI'].shape == (NGRID,)

    def test_all_outputs_finite(self, model_setup):
        model, x_dict, params, _ = model_setup
        result = model(x_dict, params)
        for key, val in result.items():
            assert torch.isfinite(val).all(), f"Non-finite values in '{key}'"

    def test_deterministic(self, model_setup):
        model, x_dict, params, _ = model_setup
        result1 = model(x_dict, params)
        result2 = model(x_dict, params)
        for key in result1:
            assert torch.allclose(result1[key], result2[key], atol=1e-6), (
                f"Non-deterministic output for '{key}'"
            )

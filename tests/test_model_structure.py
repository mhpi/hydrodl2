"""
Tests for model structure and initialization.

Coverage:
- Parameter bounds count per model variant
- State names consistency
- torch.nn.Module inheritance
- Learnable parameter counts (data-driven)
- Flux names and model-specific features
- nmul parameter scaling

NOTE: All parametrized tests cover Hbv, Hbv_1_1p, Hbv_2, and Hbv_2_hourly.
"""

import pytest
import torch

from hydrodl2 import load_model
from tests import (
    DEVICE,
    _hbv_config_dict,
    _hbv_1_1p_config_dict,
    _hbv_2_config_dict,
)


# ---------------------------------------------------------------------------- #
#  Expected learnable parameter counts
#  To add a new model: add an entry here AND in _LEARNABLE_PARAM_CONFIGS.
# ---------------------------------------------------------------------------- #

EXP_LEARNABLE_PARAMS = {
    'Hbv': {'total': 14},
    'Hbv_1_1p': {'total': 16},
    'Hbv_2': {'total': 16, 'dynamic': 3, 'static': 13},
}

_LEARNABLE_PARAM_CONFIGS = {
    'Hbv': ('hbv', _hbv_config_dict, {'dynamic_params': ['parBETA']}),
    'Hbv_1_1p': (
        'hbv_1_1p',
        _hbv_1_1p_config_dict,
        {'dynamic_params': ['parBETA', 'parBETAET']},
    ),
    'Hbv_2': (
        'hbv_2',
        _hbv_2_config_dict,
        {'dynamic_params': ['parBETA', 'parK0', 'parBETAET']},
    ),
}


class TestModelStructure:
    """Verify model initialization and structural properties."""

    @pytest.mark.parametrize(
        "model_name, ver_name, expected_phy, expected_routing",
        [
            ('hbv', 'Hbv', 12, 2),
            ('hbv_1_1p', 'Hbv_1_1p', 14, 2),
            ('hbv_2', 'Hbv_2', 16, 2),
            ('hbv_2_hourly', 'Hbv_2_hourly', 19, 2),
        ],
    )
    def test_parameter_bounds_count(
        self, model_name, ver_name, expected_phy, expected_routing
    ):
        Cls = load_model(model_name, ver_name=ver_name)
        model = Cls(device=DEVICE)
        assert len(model.parameter_bounds) == expected_phy
        assert len(model.routing_parameter_bounds) == expected_routing

    @pytest.mark.parametrize(
        "model_name, ver_name",
        [
            ('hbv', 'Hbv'),
            ('hbv_1_1p', 'Hbv_1_1p'),
            ('hbv_2', 'Hbv_2'),
            ('hbv_2_hourly', 'Hbv_2_hourly'),
        ],
    )
    def test_state_names(self, model_name, ver_name):
        Cls = load_model(model_name, ver_name=ver_name)
        model = Cls(device=DEVICE)
        assert model.state_names == ['SNOWPACK', 'MELTWATER', 'SM', 'SUZ', 'SLZ']

    @pytest.mark.parametrize(
        "model_name, ver_name",
        [
            ('hbv', 'Hbv'),
            ('hbv_1_1p', 'Hbv_1_1p'),
            ('hbv_2', 'Hbv_2'),
            ('hbv_2_hourly', 'Hbv_2_hourly'),
        ],
    )
    def test_is_nn_module(self, model_name, ver_name):
        Cls = load_model(model_name, ver_name=ver_name)
        model = Cls(device=DEVICE)
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize(
        "model_name",
        list(EXP_LEARNABLE_PARAMS.keys()),
        ids=list(EXP_LEARNABLE_PARAMS.keys()),
    )
    def test_learnable_param_count(self, model_name):
        """Learnable parameter counts must match expected values."""
        loader, config_fn, kwargs = _LEARNABLE_PARAM_CONFIGS[model_name]
        Cls = load_model(loader)
        config = config_fn(**kwargs)
        model = Cls(config, device=DEVICE)
        expected = EXP_LEARNABLE_PARAMS[model_name]

        assert model.learnable_param_count == expected['total'], (
            f"Total learnable param count mismatch for {model_name}: "
            f"expected {expected['total']}, got {model.learnable_param_count}"
        )
        if 'dynamic' in expected:
            assert model.learnable_param_count1 == expected['dynamic'], (
                f"Dynamic param count mismatch for {model_name}: "
                f"expected {expected['dynamic']}, got {model.learnable_param_count1}"
            )
            assert model.learnable_param_count2 == expected['static'], (
                f"Static param count mismatch for {model_name}: "
                f"expected {expected['static']}, got {model.learnable_param_count2}"
            )

    def test_hbv_flux_names_count(self):
        Hbv = load_model('hbv')
        model = Hbv(device=DEVICE)
        assert 'streamflow' in model.flux_names
        assert 'BFI' in model.flux_names
        assert len(model.flux_names) == 17

    def test_hbv_1_1p_has_capillary(self):
        Cls = load_model('hbv_1_1p')
        model = Cls(device=DEVICE)
        assert 'capillary' in model.flux_names
        assert 'parC' in model.parameter_bounds

    def test_hbv_nmul_changes_param_count(self):
        Hbv = load_model('hbv')
        config = _hbv_config_dict(dynamic_params=['parBETA'], nmul=2)
        model = Hbv(config, device=DEVICE)
        assert model.learnable_param_count == 26

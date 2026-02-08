"""Fixtures for testing differentiable physics models."""

import pytest
import torch

from hydrodl2 import load_model
from tests import (
    DEVICE,
    SEED,
    _hbv_1_1p_config_dict,
    _hbv_2_config_dict,
    _hbv_2_hourly_config_dict,
    _hbv_config_dict,
    make_hbv2_hourly_inputs,
    make_hbv2_inputs,
    make_hbv_inputs,
)


@pytest.fixture(
    params=['Hbv', 'Hbv_1_1p', 'Hbv_2', 'Hbv_2_hourly'],
    ids=['Hbv', 'Hbv_1_1p', 'Hbv_2', 'Hbv_2_hourly'],
)
def model_setup(request):
    """Yields (model, x_dict, params, streamflow_key) for each model variant."""
    torch.manual_seed(SEED)
    name = request.param

    if name == 'Hbv':
        Cls = load_model('hbv')
        config = _hbv_config_dict()
        model = Cls(config, device=DEVICE)
        x_dict, params = make_hbv_inputs(model)
        sf_key = 'streamflow'
    elif name == 'Hbv_1_1p':
        Cls = load_model('hbv_1_1p')
        config = _hbv_1_1p_config_dict()
        model = Cls(config, device=DEVICE)
        x_dict, params = make_hbv_inputs(model)
        sf_key = 'streamflow'
    elif name == 'Hbv_2':
        Cls = load_model('hbv_2')
        config = _hbv_2_config_dict()
        model = Cls(config, device=DEVICE)
        x_dict, params = make_hbv2_inputs(model)
        sf_key = 'streamflow'
    elif name == 'Hbv_2_hourly':
        Cls = load_model('hbv_2_hourly')
        config = _hbv_2_hourly_config_dict()
        model = Cls(config, device=DEVICE)
        x_dict, params = make_hbv2_hourly_inputs(model)
        sf_key = 'streamflow'
    else:
        raise ValueError(f"Unknown model: {name}")

    return model, x_dict, params, sf_key


@pytest.fixture(
    params=['Hbv', 'Hbv_1_1p'],
    ids=['Hbv', 'Hbv_1_1p'],
)
def simple_model_setup(request):
    """Yields (model, x_dict, params) for simple models with full flux_dict."""
    torch.manual_seed(SEED)
    name = request.param

    if name == 'Hbv':
        Cls = load_model('hbv')
        config = _hbv_config_dict()
        model = Cls(config, device=DEVICE)
    else:
        Cls = load_model('hbv_1_1p')
        config = _hbv_1_1p_config_dict()
        model = Cls(config, device=DEVICE)

    x_dict, params = make_hbv_inputs(model)
    return model, x_dict, params

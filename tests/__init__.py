"""Shared test utilities, constants, and helpers for hydrodl2 tests."""

import pytest
import torch

from hydrodl2 import load_model

# ---------------------------------------------------------------------------- #
#  Constants
# ---------------------------------------------------------------------------- #

SEED = 111111
NSTEPS = 30
NGRID = 5
DEVICE = torch.device('cpu')


# ---------------------------------------------------------------------------- #
#  Config builders
# ---------------------------------------------------------------------------- #


def _hbv_config_dict(
    dynamic_params=None,
    nmul=1,
    routing=True,
    warm_up=0,
    warm_up_states=True,
    cache_states=False,
):
    """Config dict for Hbv with 1 dynamic parameter."""
    if dynamic_params is None:
        dynamic_params = ['parBETA']
    return {
        'warm_up': warm_up,
        'warm_up_states': warm_up_states,
        'dy_drop': 0.0,
        'dynamic_params': {'Hbv': dynamic_params},
        'variables': ['prcp', 'tmean', 'pet'],
        'routing': routing,
        'comprout': False,
        'nearzero': 1e-5,
        'nmul': nmul,
        'cache_states': cache_states,
    }


def _hbv_1_1p_config_dict(
    dynamic_params=None,
    nmul=1,
    routing=True,
    warm_up=0,
    warm_up_states=True,
    cache_states=False,
):
    """Config dict for Hbv_1_1p with 2 dynamic parameters."""
    if dynamic_params is None:
        dynamic_params = ['parBETA', 'parBETAET']
    return {
        'warm_up': warm_up,
        'warm_up_states': warm_up_states,
        'dy_drop': 0.0,
        'dynamic_params': {'Hbv_1_1p': dynamic_params},
        'variables': ['prcp', 'tmean', 'pet'],
        'routing': routing,
        'comprout': False,
        'nearzero': 1e-5,
        'nmul': nmul,
        'cache_states': cache_states,
    }


def _hbv_2_config_dict(
    dynamic_params=None,
    nmul=1,
    routing=False,
    warm_up=0,
    warm_up_states=True,
    cache_states=False,
):
    """Config dict for Hbv_2 with 3 dynamic parameters."""
    if dynamic_params is None:
        dynamic_params = ['parBETA', 'parK0', 'parBETAET']
    return {
        'warm_up': warm_up,
        'warm_up_states': warm_up_states,
        'dy_drop': 0.0,
        'dynamic_params': {'Hbv_2': dynamic_params},
        'variables': ['prcp', 'tmean', 'pet'],
        'routing': routing,
        'comprout': False,
        'nearzero': 1e-5,
        'nmul': nmul,
        'cache_states': cache_states,
    }


def _hbv_2_hourly_config_dict(
    dynamic_params=None,
    nmul=1,
    routing=False,
    warm_up=0,
    warm_up_states=True,
    cache_states=False,
):
    """Config dict for Hbv_2_hourly with 3 dynamic parameters."""
    if dynamic_params is None:
        dynamic_params = ['parBETA', 'parK0', 'parBETAET']
    return {
        'warm_up': warm_up,
        'warm_up_states': warm_up_states,
        'dy_drop': 0.0,
        'dynamic_params': {'Hbv_2_hourly': dynamic_params},
        'variables': ['prcp', 'tmean', 'pet'],
        'routing': routing,
        'comprout': False,
        'nearzero': 1e-5,
        'nmul': nmul,
        'cache_states': cache_states,
    }


# ---------------------------------------------------------------------------- #
#  Data generators
# ---------------------------------------------------------------------------- #


def make_forcing(nsteps, ngrid):
    """Generate mock forcing data with realistic scales."""
    x = torch.rand(nsteps, ngrid, 3)
    x[:, :, 0] *= 10.0
    x[:, :, 1] = x[:, :, 1] * 30.0 - 5.0
    x[:, :, 2] *= 5.0
    return x


def make_hbv_inputs(model, nsteps=NSTEPS, ngrid=NGRID):
    """Create (x_dict, params) for Hbv and Hbv_1_1p models."""
    x_phy = make_forcing(nsteps, ngrid)
    x_dict = {'x_phy': x_phy}
    params = torch.randn(nsteps, ngrid, model.learnable_param_count)
    return x_dict, params


def make_hbv2_inputs(model, nsteps=NSTEPS, ngrid=NGRID):
    """Create (x_dict, params) for Hbv_2 models."""
    x_phy = make_forcing(nsteps, ngrid)
    ac_all = torch.rand(ngrid) * 1000 + 10
    elev_all = torch.rand(ngrid) * 3000
    x_dict = {'x_phy': x_phy, 'ac_all': ac_all, 'elev_all': elev_all}
    # Hbv_2 does not apply sigmoid internally; params must be in [0, 1].
    dynamic_params = torch.rand(nsteps, ngrid, model.learnable_param_count1)
    static_params = torch.rand(ngrid, model.learnable_param_count2)
    params = [dynamic_params, static_params]
    return x_dict, params


def make_hbv2_hourly_inputs(model, nsteps=NSTEPS, ngrid=NGRID, n_gages=2):
    """Create (x_dict, params) for Hbv_2_hourly models."""
    x_phy = make_forcing(nsteps, ngrid)
    ac_all = torch.rand(ngrid) * 1000 + 10
    elev_all = torch.rand(ngrid) * 3000
    outlet_topo = torch.zeros(n_gages, ngrid)
    mid = ngrid // 2
    outlet_topo[0, :mid] = 1
    outlet_topo[1, mid:] = 1
    areas = torch.rand(ngrid) * 100 + 1
    x_dict = {
        'x_phy': x_phy,
        'ac_all': ac_all,
        'elev_all': elev_all,
        'outlet_topo': outlet_topo,
        'areas': areas,
    }
    n_pairs = int((outlet_topo == 1).sum().item())
    # Hbv_2_hourly does not apply sigmoid internally; params must be in [0, 1].
    dynamic_params = torch.rand(nsteps, ngrid, model.learnable_param_count1)
    static_params = torch.rand(ngrid, model.learnable_param_count2)
    distr_params = torch.rand(n_pairs, len(model.distr_parameter_bounds))
    params = [dynamic_params, static_params, distr_params]
    return x_dict, params


# ---------------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------------- #


def _assert_close_to_any(actual, expected_values, stat_name, model_name, rtol=1e-5):
    """Assert actual matches any value in expected_values within tolerance.

    LSTM CPU non-determinism means different hardware produces different (but
    individually reproducible) outputs. Each entry in expected_values is a
    known value from one environment.
    """
    for exp in expected_values:
        if abs(actual - exp) < rtol:
            return
    exp_str = ", ".join(f"{v:.8e}" for v in expected_values)
    pytest.fail(
        f"{stat_name} regression failed for {model_name}: "
        f"got {actual:.8e}, expected one of [{exp_str}].\n"
        f"If this is a new environment, add the value to the appropriate list."
    )


def _create_hbv_deterministic():
    """Create a deterministic Hbv model with seeded inputs."""
    torch.manual_seed(SEED)
    Hbv = load_model('hbv')
    config = _hbv_config_dict(dynamic_params=['parBETA'])
    model = Hbv(config, device=DEVICE)
    x_dict, params = make_hbv_inputs(model)
    return model, x_dict, params


def _create_hbv_1_1p_deterministic():
    """Create a deterministic Hbv_1_1p model with seeded inputs."""
    torch.manual_seed(SEED)
    Cls = load_model('hbv_1_1p')
    config = _hbv_1_1p_config_dict(dynamic_params=['parBETA', 'parBETAET'])
    model = Cls(config, device=DEVICE)
    x_dict, params = make_hbv_inputs(model)
    return model, x_dict, params


def _create_hbv2_deterministic():
    """Create a deterministic Hbv_2 model with seeded inputs."""
    torch.manual_seed(SEED)
    Cls = load_model('hbv_2')
    config = _hbv_2_config_dict(dynamic_params=['parBETA', 'parK0', 'parBETAET'])
    model = Cls(config, device=DEVICE)
    x_dict, params = make_hbv2_inputs(model)
    return model, x_dict, params


def _create_hbv2_hourly_deterministic():
    """Create a deterministic Hbv_2_hourly model with seeded inputs."""
    torch.manual_seed(SEED)
    Cls = load_model('hbv_2_hourly')
    config = _hbv_2_hourly_config_dict(
        dynamic_params=['parBETA', 'parK0', 'parBETAET'],
    )
    model = Cls(config, device=DEVICE)
    x_dict, params = make_hbv2_hourly_inputs(model)
    return model, x_dict, params

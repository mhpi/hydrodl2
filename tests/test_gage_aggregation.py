"""
Tests for Hbv_2 gage-level aggregation (`gage_agg=True`).

Coverage:
- Every full_output/routing/comprout combination runs and returns compact,
  gage-level outputs (per-component outputs are not aggregated to gages).
- In-loop aggregation matches post-hoc area-weighted aggregation of the
  catchment-level output.
"""

import pytest
import torch

from hydrodl2 import load_model
from tests import DEVICE, NSTEPS, SEED, _hbv_2_config_dict, make_hbv2_inputs

N_GAGES = 2


def _agg_inputs(x_dict):
    """Add unit areas (normalized within each gage) and gage incidence matrix."""
    ngrid = x_dict['x_phy'].shape[1]
    gage_of_unit = torch.arange(ngrid) * N_GAGES // ngrid
    outlet_topo = torch.nn.functional.one_hot(gage_of_unit, N_GAGES).float()
    areas = torch.rand(ngrid) * 100 + 1
    areas = areas / (outlet_topo @ (outlet_topo.T @ areas))
    return {**x_dict, 'areas': areas, 'outlet_topo': outlet_topo}


def _model(**overrides):
    config = {**_hbv_2_config_dict(), **overrides}
    return load_model('hbv_2')(config, device=DEVICE)


@pytest.mark.parametrize("full_output", [False, True])
@pytest.mark.parametrize("routing", [False, True])
@pytest.mark.parametrize("comprout", [False, True])
def test_gage_agg_runs_and_returns_gage_level(full_output, routing, comprout):
    torch.manual_seed(SEED)
    model = _model(
        gage_agg=True, full_output=full_output, routing=routing, comprout=comprout
    )
    x_dict, params = make_hbv2_inputs(model)
    out = model(_agg_inputs(x_dict), params)

    assert set(out) == {'streamflow', 'AET_hydro'}
    for key, val in out.items():
        assert val.shape == (NSTEPS, N_GAGES, 1), key
        assert torch.isfinite(val).all(), key


@pytest.mark.parametrize("full_output", [False, True])
def test_gage_agg_matches_post_hoc_aggregation(full_output):
    torch.manual_seed(SEED)
    model_cat = _model(full_output=full_output)
    x_dict, params = make_hbv2_inputs(model_cat)
    x_dict = _agg_inputs(x_dict)
    model_agg = _model(gage_agg=True, full_output=full_output)

    out_cat = model_cat(x_dict, params)
    out_agg = model_agg(x_dict, params)

    areas, outlet_topo = x_dict['areas'], x_dict['outlet_topo']
    for key in ('streamflow', 'AET_hydro'):
        expected = (out_cat[key][..., 0] * areas) @ outlet_topo
        torch.testing.assert_close(out_agg[key][..., 0], expected)

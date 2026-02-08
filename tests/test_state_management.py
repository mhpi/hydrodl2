"""
Tests for model state management and warmup behavior.

Coverage:
- Initial state shapes and values
- get_states / load_states lifecycle
- State validation (wrong type, wrong count)
- State caching after forward pass
- Warmup period behavior (with and without state reuse)

NOTE: Parametrized tests cover Hbv, Hbv_1_1p, Hbv_2, and Hbv_2_hourly.
"""

import pytest
import torch

from hydrodl2 import load_model
from tests import DEVICE, NGRID, NSTEPS, SEED, _hbv_config_dict, make_hbv_inputs


class TestStateManagement:
    """Verify model state initialization, persistence, and loading."""

    @pytest.mark.parametrize(
        "model_name, ver_name",
        [
            ('hbv', 'Hbv'),
            ('hbv_1_1p', 'Hbv_1_1p'),
            ('hbv_2', 'Hbv_2'),
            ('hbv_2_hourly', 'Hbv_2_hourly'),
        ],
    )
    def test_init_states_shape(self, model_name, ver_name):
        Cls = load_model(model_name, ver_name=ver_name)
        model = Cls(device=DEVICE)
        states = model._init_states(NGRID)
        assert len(states) == 5
        for s in states:
            assert s.shape == (NGRID, model.nmul)
            assert torch.allclose(s, torch.full_like(s, 0.001))

    @pytest.mark.parametrize(
        "model_name, ver_name",
        [
            ('hbv', 'Hbv'),
            ('hbv_1_1p', 'Hbv_1_1p'),
            ('hbv_2', 'Hbv_2'),
            ('hbv_2_hourly', 'Hbv_2_hourly'),
        ],
    )
    def test_get_states_before_forward_is_none(self, model_name, ver_name):
        Cls = load_model(model_name, ver_name=ver_name)
        model = Cls(device=DEVICE)
        assert model.get_states() is None

    def test_get_states_after_forward(self, simple_model_setup):
        model, x_dict, params = simple_model_setup
        model(x_dict, params)
        states = model.get_states()
        assert states is not None
        assert len(states) == 5

    def test_load_states_valid(self):
        Hbv = load_model('hbv')
        model = Hbv(device=DEVICE)
        fake_states = tuple(torch.rand(NGRID, 1) for _ in range(5))
        model.load_states(fake_states)
        assert model.states is not None
        for loaded, original in zip(model.states, fake_states):
            assert torch.allclose(loaded, original)

    def test_load_states_wrong_type(self):
        Hbv = load_model('hbv')
        model = Hbv(device=DEVICE)
        with pytest.raises(ValueError, match="must be a tensor"):
            model.load_states(tuple([1.0] * 5))

    def test_load_states_wrong_count(self):
        Hbv = load_model('hbv')
        model = Hbv(device=DEVICE)
        with pytest.raises(ValueError, match="must be a tuple of 5"):
            model.load_states(tuple(torch.rand(NGRID, 1) for _ in range(3)))

    def test_cache_states_persistence(self):
        Hbv = load_model('hbv')
        config = _hbv_config_dict(dynamic_params=['parBETA'], cache_states=True)
        model = Hbv(config, device=DEVICE)
        torch.manual_seed(SEED)
        x_dict, params = make_hbv_inputs(model)
        model(x_dict, params)
        assert model.states is not None
        for s in model.states:
            assert not torch.allclose(s, torch.full_like(s, 0.001)), (
                "States should have changed after forward pass"
            )


class TestWarmUp:
    """Verify warmup period behavior."""

    def test_warm_up_states_mode(self):
        Hbv = load_model('hbv')
        warm_up = 5
        total_steps = NSTEPS + warm_up
        config = _hbv_config_dict(
            dynamic_params=['parBETA'],
            warm_up=warm_up,
            warm_up_states=True,
        )
        model = Hbv(config, device=DEVICE)
        torch.manual_seed(SEED)
        x_dict, params = make_hbv_inputs(model, nsteps=total_steps)
        result = model(x_dict, params)
        assert result['streamflow'].shape[0] == NSTEPS

    def test_no_warm_up_states_mode(self):
        Hbv = load_model('hbv')
        warm_up = 5
        config = _hbv_config_dict(
            dynamic_params=['parBETA'],
            warm_up=warm_up,
            warm_up_states=False,
        )
        model = Hbv(config, device=DEVICE)
        torch.manual_seed(SEED)
        x_dict, params = make_hbv_inputs(model, nsteps=NSTEPS)
        result = model(x_dict, params)
        assert result['streamflow'].shape[0] == NSTEPS - warm_up

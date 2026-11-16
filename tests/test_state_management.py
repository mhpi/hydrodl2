"""
Tests for model state management and warmup behavior.

Coverage:
- Initial state shapes and values
- get_states / load_states lifecycle
- State validation (wrong type, wrong count)
- State caching after forward pass
- Warmup period behavior (with and without state reuse)
- Hbv_2 and Hbv_2_mts runtime state snapshot and restore
  (get/load_runtime_state)

NOTE: Parametrized tests cover Hbv, Hbv_1_1p, Hbv_2, and Hbv_2_hourly.
"""

import pytest
import torch

from hydrodl2 import load_model
from tests import (
    DEVICE,
    NGRID,
    NSTEPS,
    SEED,
    _hbv_2_config_dict,
    _hbv_config_dict,
    make_hbv2_inputs,
    make_hbv_inputs,
)


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

    def test_warmup_states_mode(self):
        Hbv = load_model('hbv')
        warmup = 5
        total_steps = NSTEPS + warmup
        config = _hbv_config_dict(
            dynamic_params=['parBETA'],
            warmup=warmup,
            warmup_states=True,
        )
        model = Hbv(config, device=DEVICE)
        torch.manual_seed(SEED)
        x_dict, params = make_hbv_inputs(model, nsteps=total_steps)
        result = model(x_dict, params)
        assert result['streamflow'].shape[0] == NSTEPS

    def test_no_warmup_states_mode(self):
        Hbv = load_model('hbv')
        warmup = 5
        config = _hbv_config_dict(
            dynamic_params=['parBETA'],
            warmup=warmup,
            warmup_states=False,
        )
        model = Hbv(config, device=DEVICE)
        torch.manual_seed(SEED)
        x_dict, params = make_hbv_inputs(model, nsteps=NSTEPS)
        result = model(x_dict, params)
        assert result['streamflow'].shape[0] == NSTEPS - warmup


class TestMtsRuntimeState:
    """Verify Hbv_2_mts runtime state snapshot and restore."""

    @staticmethod
    def _make_model():
        Cls = load_model('hbv_2_mts')
        return Cls({}, device=DEVICE)

    @staticmethod
    def _random_states(model, nsteps=None):
        shape = (NGRID, model.hif_model.nmul)
        if nsteps is not None:
            shape = (nsteps, *shape)
        return tuple(torch.rand(shape) for _ in model.lof_model.state_names)

    def _warm_model(self):
        torch.manual_seed(SEED)
        model = self._make_model()
        lof = self._random_states(model)
        hif = self._random_states(model)
        model.lof_model.states = lof
        model.states = (lof, hif)
        # A batched pass leaves the full series in the high-frequency cache.
        model._state_cache = [lof, self._random_states(model, nsteps=NSTEPS)]
        model.use_from_cache = True
        model.lof_rollout = True
        model._static_param_cache = ('stale',)
        return model

    def test_roundtrip_restores_states_and_flags(self):
        """A restored model holds the same states and flags."""
        source = self._warm_model()
        target = self._make_model()
        target.load_runtime_state(source.get_runtime_state())

        assert target.use_from_cache is True
        assert target.lof_rollout is True
        for restored, original in zip(target.states, source.states):
            for r, o in zip(restored, original):
                assert torch.equal(r, o)
        for r, o in zip(target.lof_model.states, source.lof_model.states):
            assert torch.equal(r, o)
        for r, o in zip(target._state_cache[0], source._state_cache[0]):
            assert torch.equal(r, o)

    def test_hif_cache_reduced_to_last_step(self):
        """The high-frequency cache restores as the last-step states."""
        source = self._warm_model()
        snapshot = source.get_runtime_state()
        for cached, last in zip(snapshot['state_cache'][1], source.states[1]):
            assert cached is last

    def test_restore_clears_static_param_cache(self):
        """Static parameter cache is keyed by identity, so restore clears it."""
        source = self._warm_model()
        source.load_runtime_state(source.get_runtime_state())
        assert source._static_param_cache is None

    def test_cold_model_roundtrip(self):
        """A model that never ran restores to the same empty state."""
        source = self._make_model()
        target = self._make_model()
        target.load_runtime_state(source.get_runtime_state())
        assert target.states == (None, None)
        assert target._state_cache == [None, None]
        assert target.lof_model.states is None
        assert target.use_from_cache is False

    def test_rejects_unknown_keys(self):
        """A snapshot with missing or extra keys is rejected."""
        model = self._make_model()
        state = model.get_runtime_state()
        state['extra'] = None
        with pytest.raises(ValueError, match='do not match'):
            model.load_runtime_state(state)


class TestHbv2RuntimeState:
    """Verify Hbv_2 runtime state snapshot and restore."""

    @staticmethod
    def _make_model():
        Cls = load_model('hbv_2', ver_name='Hbv_2')
        return Cls(_hbv_2_config_dict(cache_states=True), device=DEVICE)

    @staticmethod
    def _step(model, x_dict, params, t):
        x_t = {**x_dict, 'x_phy': x_dict['x_phy'][t : t + 1]}
        return model(x_t, [params[0][t : t + 1], params[1]])['streamflow']

    def test_resume_matches_continuous(self):
        """A restored model continues stepwise exactly as the original."""
        torch.manual_seed(SEED)
        source = self._make_model()
        x_dict, params = make_hbv2_inputs(source)
        split = NSTEPS // 2

        # Batched spin-up, then a snapshot and stepwise continuation.
        head = {**x_dict, 'x_phy': x_dict['x_phy'][:split]}
        source(head, [params[0][:split], params[1]])
        snapshot = {
            'states': tuple(s.clone() for s in source.get_runtime_state()['states']),
        }
        expected = [self._step(source, x_dict, params, t) for t in range(split, NSTEPS)]

        target = self._make_model()
        target.load_runtime_state(snapshot)
        result = [self._step(target, x_dict, params, t) for t in range(split, NSTEPS)]

        for r, e in zip(result, expected):
            assert torch.equal(r, e)

    def test_cold_model_roundtrip(self):
        """A model that never ran restores to the same empty state."""
        source = self._make_model()
        target = self._make_model()
        target.load_runtime_state(source.get_runtime_state())
        assert target.states is None

    def test_rejects_unknown_keys(self):
        """A snapshot with missing or extra keys is rejected."""
        model = self._make_model()
        state = model.get_runtime_state()
        state['extra'] = None
        with pytest.raises(ValueError, match='do not match'):
            model.load_runtime_state(state)

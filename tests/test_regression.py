"""
Regression tests for deterministic model outputs.

These tests verify that models produce identical numerical results across code
changes. If any test here fails, model numerics have changed.

NOTE: Expected values were generated with seed=111111 and the
mock dataset from conftest.py. If conftest.py mock data generation changes,
these values must be updated.

NOTE: LSTM on CPU is non-deterministic across hardware (e.g. local machine vs.
GitHub Actions runner). Each expected value is therefore a list of known
good values — one per environment. If a test fails on a new environment,
add the reported value in the error to the appropriate list.
"""

import pytest

from tests import (
    _assert_close_to_any,
    _create_hbv2_deterministic,
    _create_hbv2_hourly_deterministic,
    _create_hbv_1_1p_deterministic,
    _create_hbv_deterministic,
)

# ---------------------------------------------------------------------------- #
#  Expected regression statistics
#  To add a new model: add an entry here AND a creator in _CREATORS below.
# ---------------------------------------------------------------------------- #

EXP_SNAPSHOTS = {
    'Hbv': {
        'output_key': 'streamflow',
        'mean': [
            0.02233361080288887,  # Local machine
        ],
        'std': [
            0.045270565897226334,
        ],
    },
    'Hbv_1_1p': {
        'output_key': 'streamflow',
        'mean': [
            0.0013609424931928515,
        ],
        'std': [
            0.002616790821775794,
        ],
    },
    'Hbv_2': {
        'output_key': 'streamflow',
        'mean': [
            0.047415703535079956,
        ],
        'std': [
            0.09584921598434448,
        ],
    },
    'Hbv_2_hourly': {
        'output_key': 'Qs',
        'mean': [
            0.024369191378355026,
        ],
        'std': [
            0.06593040376901627,
        ],
    },
}

_CREATORS = {
    'Hbv': _create_hbv_deterministic,
    'Hbv_1_1p': _create_hbv_1_1p_deterministic,
    'Hbv_2': _create_hbv2_deterministic,
    'Hbv_2_hourly': _create_hbv2_hourly_deterministic,
}


# ---------------------------------------------------------------------------- #
#  Fixture
# ---------------------------------------------------------------------------- #


@pytest.fixture(
    params=list(EXP_SNAPSHOTS.keys()),
    ids=list(EXP_SNAPSHOTS.keys()),
)
def regression_setup(request):
    """Yields (model_name, model, x_dict, params) for each deterministic model."""
    name = request.param
    model, x_dict, params = _CREATORS[name]()
    return name, model, x_dict, params


# ---------------------------------------------------------------------------- #
#  Tests
# ---------------------------------------------------------------------------- #


class TestRegressionSnapshots:
    """Deterministic snapshots that lock down numerical output.

    If any of these fail, model numerics have changed.
    Each stat is checked against a list of known-good values from different
    environments (see EXP_SNAPSHOTS). If a test fails on a new environment,
    add the reported value to the appropriate list.
    """

    def test_output_mean(self, regression_setup):
        """Verify output mean matches expected snapshot value."""
        name, model, x_dict, params = regression_setup
        expected = EXP_SNAPSHOTS[name]
        result = model(x_dict, params)
        key = expected['output_key']
        _assert_close_to_any(
            result[key].mean().item(),
            expected['mean'],
            f'{key}_mean',
            name,
        )

    def test_output_std(self, regression_setup):
        """Verify output std matches expected snapshot value."""
        name, model, x_dict, params = regression_setup
        expected = EXP_SNAPSHOTS[name]
        result = model(x_dict, params)
        key = expected['output_key']
        _assert_close_to_any(
            result[key].std().item(),
            expected['std'],
            f'{key}_std',
            name,
        )

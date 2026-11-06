"""Generate source_parity.npz from the dPLHBVrelease source code.

Run from this directory, in an environment that can import the source code:

    python generate_source_parity.py /path/to/dPLHBVrelease

Expected flows must only ever come from the source models, so that
test_source_parity.py compares hydrodl2 against the source and not against itself.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

NT = 70
WARMUP = 30
NGRID = 4
NMUL = 2
SEED = 111111

# hydrodl2 name: (source name, parameter count, source dynamic parameter indices)
MODELS = {
    'hbv': ('HBV1_0', 13, [1, 13]),
    'hbv_1_1p': ('HBV1_1p', 14, [1, 3, 13]),
}


class _Passthrough(torch.nn.Module):
    """Replaces the source LSTM so raw parameters are supplied directly."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged."""
        return z


def _make_forcing(rng: np.random.Generator) -> np.ndarray:
    """Build [time, grid, var] forcing (prcp, tmean, pet) with rain and snow."""
    t = np.arange(NT)
    season = 12.0 * np.sin(2.0 * np.pi * t / 40.0)[:, None]
    base_temp = np.array([-4.0, 2.0, 8.0, 14.0])
    tmean = base_temp + season + rng.normal(0.0, 1.5, (NT, NGRID))
    prcp = rng.gamma(0.6, 8.0, (NT, NGRID))
    pet = np.clip(0.2 * (tmean + 5.0), 0.0, None)
    return np.stack([prcp, tmean, pet], axis=2).astype(np.float32)


def _make_raw_params(rng: np.random.Generator, n_params: int) -> np.ndarray:
    """Build [time, grid, param] raw outputs that vary slowly through time."""
    n = n_params * NMUL + 2
    base = rng.normal(0.0, 1.5, (1, NGRID, n))
    walk = rng.normal(0.0, 0.05, (NT, NGRID, n)).cumsum(axis=0)
    return (base + walk * 0.2).astype(np.float32)


def _run_source(
    rnn: object,
    name: str,
    n_params: int,
    dynamic: list[int],
    x: np.ndarray,
    raw: np.ndarray,
) -> np.ndarray:
    """Run a source model on CPU and return streamflow [time, grid]."""
    model = rnn.dHBVModel(
        ninv=38,
        nfea=n_params,
        nmul=NMUL,
        hiddeninv=8,
        inittime=WARMUP,
        routOpt=True,
        comprout=False,
        compwts=False,
        staind=-1,
        tdlst=dynamic,
        dydrop=0.0,
        ETMod=True,
        model_name=name,
    )
    model.lstminv = _Passthrough()
    with torch.no_grad():
        flow = model(torch.from_numpy(x), torch.from_numpy(raw))
    return flow[:, :, 0].numpy()


def main(source_dir: str) -> None:
    """Write the fixture next to this script.

    Parameters
    ----------
    source_dir
        Path to the dPLHBVrelease repository.
    """
    # The source models call .cuda() unconditionally.
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.Tensor.cuda = lambda self, *args, **kwargs: self

    sys.path.insert(0, str(Path(source_dir) / 'hydroDL-dev'))
    from hydroDL.model import rnn

    rng = np.random.default_rng(SEED)
    x = _make_forcing(rng)
    arrays = {'x': x, 'warmup': WARMUP, 'nmul': NMUL}

    for key, (name, n_params, dynamic) in MODELS.items():
        raw = _make_raw_params(rng, n_params)
        arrays[f"{key}_raw"] = raw
        arrays[f"{key}_flow"] = _run_source(rnn, name, n_params, dynamic, x, raw)

    path = Path(__file__).parent / 'source_parity.npz'
    np.savez_compressed(path, **arrays)
    log.info("Wrote %s", path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1])

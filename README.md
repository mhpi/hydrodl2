# HydroDL 2.0: Differentiable Hydrological Model Repository

[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-EE4C2C?logo=pytorch)](https://pytorch.org/)

[![Build](https://github.com/mhpi/hydrodl2/actions/workflows/pytest.yaml/badge.svg?branch=master)](https://github.com/mhpi/hydrodl2/actions/workflows/pytest.yaml/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

<!-- <img src="docs/images/hydrodl2_cover_logo.png" alt="hydroOps" width="500" height="500"> -->

HydroDL 2.0 is store for differentiable hydrology models and modules designed to be used in concert with the generic differential modeling package, [dMG](https://github.com/mhpi/generic_deltamodel).

This work is mantained by [MHPI](http://water.engr.psu.edu/shen/) and advised by [Dr. Chaopeng Shen](https://water.engr.psu.edu/shen/). If you find it useful, please cite (dedicated citations are coming):

    Shen, C., et al. (2023). Differentiable modelling to unify machine learning and physical models for geosciences. Nature Reviews Earth & Environment, 4(8), 552–567. <https://doi.org/10.1038/s43017-023-00450-9>.

</br>

## Installation

To install hydrodl2, clone the repo and install in developer mode with [Astral UV](https://docs.astral.sh/uv/) (recommended)

    ```bash
    git clone https://github.com/mhpi/hydrodl2.git

    cd hydrodl2
    uv pip install .
    ```

or with pip

    ```bash
    pip install .
    ```

</br>

## Repo

    ```text
    .
    ├── src/
    |   └── hydrodl2/ 
    │       ├── api/                   # Main API
    │       |   ├── __init__.py        
    │       |   └── methods.py         # Methods exposed to end-users
    |       ├── core/                  # Methods used internally
    │       ├── models/                # Shared models directory
    │       |   └── hbv/               # HBV models
    |       └── modules/               # Augmentations for dMG models
    └── docs/                          
    ```

</br>

## Contributing

We welcome contributions! Please submit changes via a fork and pull requests. For more details, refer to [docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md).

---

*Please submit an [issue](https://github.com/mhpi/hydrodl2/issues) to report any questions, concerns, bugs, etc.*

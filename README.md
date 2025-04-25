# HydroDL 2.0: Differentiable Hydrological Model Repository

[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)]()
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![tests](https://github.com/mhpi/hydroDL2/actions/workflows/pytest.yaml/badge.svg?branch=master)](https://github.com/mhpi/hydroDL2/actions/workflows/pytest.yaml/)
[![image](https://img.shields.io/github/license/saltstack/salt)](https://github.com/mhpi/hydroDL2/blob/master/LICENSE)

<img src="docs/images/hydrodl2_cover_logo.png" alt="hydroOps" width="500" height="500">

HydroDL 2.0 is store for hydrology models and modules designed to be used in concert with the generic differential modeling package, [dMG](https://github.com/mhpi/generic_deltaModel). 


### How to Install:
```shell
git clone https://github.com/mhpi/hydroDL2.git
cd hydrodl2
uv pip install .
```


### Developer Mode Installation:
The same clone as above, but add flag to install the package in editable mode (changes to source code will be reflected in imports)
```shell
pip install -e .
```

### Maintainers:
See Pyproject.toml for information.


### Repository Structure:

    .
    ├── src/
    |   └── hydroDL2/ 
    │       ├── api/                   # Main API
    │       |   ├── __init__.py        
    │       |   └── methods.py         # Methods exposed to end-users
    |       ├── core/                  # Methods used internally
    │       ├── models/                # Shared models directory
    │       |   └── hbv/               # HBV models
    |       └── modules/               # Augmentations for dMG models
    └── docs/                          

### Contributing:
We welcome contributions! Please submit changes via a fork and pull request.

# hydroDL2.0
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://img.shields.io/github/license/saltstack/salt)](https://github.com/mhpi/hydroDL2/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)]()
[![Actions status](https://github.com/astral-sh/ruff/workflows/CI/badge.svg)](https://github.com/astral-sh/ruff/actions)

<img src="docs/images/hydrodl2_cover_logo.png" alt="hydroOps" width="500" height="500">

This repository serves as a store for hydrology models and modules to be used
with the generic differential modeling package, `generic_diffModel`. 

### How to Install:
```shell
git clone https://github.com/mhpi/hydroDL2.git
cd hydrodl2
pip install .
```

### Developer Mode Installation:
The same clone as above, but use hatch's developer mode setting,
```shell
pip install -e .
```

### Maintainers:
See Pyproject.toml for information.

### Contributing:
We request all changes to this repo be made through a fork and PR.


### Repository Structure:

    .
    ├── src/
    |   └── hydroDL2/ 
    │       ├── api/                   # Main API code
    │       |   ├── __init__.py        
    │       |   └── methods.py         # Methods exposed to end-users
    |       ├── core/                  # Methods used internally
    │       ├── models/                # Shared models directory
    │       |   ├── hbv/               # HBV models
    │       |   └── prms/              # Marrmot PRMS models     
    |       └── modules/               # Augmentations for `dMG` models
    ├── tests/                         # Test suite for API and models
    │   ├── __init__.py            
    │   └── test_models.py
    ├── docs/                          
    ├── LICENSE
    ├── mkdocs.yml
    ├── pyproject.toml             
    └── README.md                      

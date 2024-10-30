# hydroDL2
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://img.shields.io/pypi/l/ruff.svg)](https://github.com/astral-sh/ruff/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)]()
[![Actions status](https://github.com/astral-sh/ruff/workflows/CI/badge.svg)](https://github.com/astral-sh/ruff/actions)

<img src="docs/images/hydrodl2_cover_logo.png" alt="hydroOps" width="500" height="500">

This repository serves as a store for hydrology models and modules to be used
with the generic differential modeling package, `generic_diffModel`. 

### How to install:
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


## Repository Structure:

hydrodl2/
├── models/                    # Shared models directory
│   ├── __init__.py            # Initializer
│   ├── hbv/                   # HBV models
│   └── prms/                  # Marrmot PRMS models
├── src/
│   ├── api/                   # Main API code
│   │   ├── __init__.py        
│   │   ├── _version.py        
│   │   ├── config.py          
│   │   ├── methods.py         # Methods exposed to end-users
│   │   └── utils/             # Helper functions
│   └── core/                  # Core utilities.
├── tests/                     # Test suite for API and models
│   ├── __init__.py            
│   └── test_models.py         # Tests for models    
├── .gitignore                 
├── LICENSE
├── mkdocs.yml
├── pyproject.toml             
└── README.md                  

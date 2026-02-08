<h1 align="center">HydroDL2: Differentiable Hydrologic Models</h1>

<!-- <p align="center"><img src="docs/images/hydrodl2.png" alt="HydroDL2" width="500" height="500"></p> -->

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9--3.13-blue?labelColor=333333" alt="Python"></a>
  <a href="https://pypi.org/project/hydrodl2/"><img src="https://img.shields.io/pypi/v/hydrodl2?logo=pypi&logoColor=white&labelColor=333333" alt="PyPI version"></a>
  <a href="https://pypi.org/project/torch/"><img src="https://img.shields.io/badge/dynamic/json?label=PyTorch&query=info.version&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorch%2Fjson&logo=pytorch&color=EE4C2C&logoColor=F900FF&labelColor=333333" alt="PyTorch"></a>
</p>

<p align="center">
  <a href="https://github.com/mhpi/hydrodl2/actions/workflows/pytest.yaml"><img src="https://img.shields.io/github/actions/workflow/status/mhpi/hydrodl2/pytest.yaml?branch=master&logo=github&label=tests&labelColor=333333" alt="Build"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&labelColor=333333" alt="Ruff"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Non--Commercial_(PSU)-yellow?labelColor=333333" alt="License"></a>
</p>

---

</br>

A library of hydrological models developed on PyTorch and designed alongside [𝛿MG](https://github.com/mhpi/generic_deltamodel) for the creation of end-to-end [differentiable models](https://www.nature.com/articles/s43017-023-00450-9), enabling parameter learning, bias correction, missing process representation, and more.

See [`𝛿MG/examples`](https://github.com/mhpi/generic_deltamodel/tree/master/example/hydrology) using HydroDL2-based HBV models for published differentiable parameter learning (dPL) applications, and see [citation](#citation) for details on individual model architectures.

</br>

## Installation

```bash
uv pip install hydrodl2
```

For development installs, see [setup](./docs/setup.md).

## Quick Start

```python
import hydrodl2

# List all available models
hydrodl2.available_models()
# {'hbv': ['hbv', 'hbv_1_1p', 'hbv_2', 'hbv_2_hourly', 'hbv_2_mts', 'hbv_adj']}

# Load a model class
Hbv = hydrodl2.load_model('hbv')

# Instantiate and use in a differentiable pipeline
model = Hbv()
```

Models are standard `torch.nn.Module` subclasses and can be composed with neural networks via [&delta;MG](https://github.com/mhpi/generic_deltamodel) for end-to-end differentiable training.

</br>

## Available Models

| Model | Name | Description |
|-------|------|-------------|
| HBV 1.0 | `hbv` | Base lumped differentiable HBV model |
| HBV Adjoint | `hbv_adj` | Implicit scheme with adjoint-based gradients |
| HBV 1.1p | `hbv_1_1p` | HBV with capillary rise modification |
| HBV 2.0 | `hbv_2` | Multi-scale, distributed HBV with elevation-dependent parameters |
| HBV 2.0 Hourly | `hbv_2_hourly` | Sub-daily variant of HBV 2.0 |
| HBV 2.0 MTS | `hbv_2_mts` | Multi-timescale variant of HBV 2.0 |

</br>

## Repository Structure

```text
.
├── src/
│   └── hydrodl2/
│       ├── api/                   # Main API
│       │   ├── __init__.py
│       │   └── methods.py         # Methods exposed to end-users
│       ├── core/                  # Methods used internally
│       ├── models/                # Shared models directory
│       │   └── hbv/               # HBV model variants
│       └── modules/               # Augmentations for δMG models
├── docs/
├── tests/
└── pyproject.toml
```

## Citation

This work is maintained by [MHPI](http://water.engr.psu.edu/shen/) and advised by [Dr. Chaopeng Shen](https://water.engr.psu.edu/shen/). If you find it useful, please cite:

> Shen, C., Appling, A.P., Gentine, P. et al. Differentiable modelling to unify machine learning and physical models for geosciences. *Nat Rev Earth Environ* **4**, 552–567 (2023). <https://doi.org/10.1038/s43017-023-00450-9>

<details>
<summary>BibTeX</summary>

```bibtex
@article{shen_differentiable_2023,
    title = {Differentiable modelling to unify machine learning and physical models for geosciences},
    volume = {4},
    issn = {2662-138X},
    url = {https://doi.org/10.1038/s43017-023-00450-9},
    doi = {10.1038/s43017-023-00450-9},
    pages = {552--567},
    number = {8},
    journaltitle = {Nature Reviews Earth \& Environment},
    author = {Shen, Chaopeng and Appling, Alison P. and Gentine, Pierre and Bandai, Toshiyuki and Gupta, Hoshin and Tartakovsky, Alexandre and Baity-Jesi, Marco and Fenicia, Fabrizio and Kifer, Daniel and Li, Li and Liu, Xiaofeng and Ren, Wei and Zheng, Yi and Harman, Ciaran J. and Clark, Martyn and Farthing, Matthew and Feng, Dapeng and Kumar, Praveen and Aboelyazeed, Doaa and Rahmani, Farshid and Song, Yalan and Beck, Hylke E. and Bindas, Tadd and Dwivedi, Dipankar and Fang, Kuai and Höge, Marvin and Rackauckas, Chris and Mohanty, Binayak and Roy, Tirthankar and Xu, Chonggang and Lawson, Kathryn},
    date = {2023-08-01},
}
```

</details>

</br>

Models:

- **(HBV)**  Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water Resources Research, 58, e2022WR032404. <https://doi.org/10.1029/2022WR032404>

    <details>
    <summary>BibTeX</summary>

    ```bibtex
    @article{https://doi.org/10.1029/2022WR032404,
        author = {Feng, Dapeng and Liu, Jiangtao and Lawson, Kathryn and Shen, Chaopeng},
        title = {Differentiable, Learnable, Regionalized Process-Based Models With Multiphysical Outputs can Approach State-Of-The-Art Hydrologic Prediction Accuracy},
        journal = {Water Resources Research},
        volume = {58},
        number = {10},
        pages = {e2022WR032404},
        keywords = {rainfall runoff, differentiable programming, machine learning, physical model, differentiable hydrology, LSTM},
        doi = {https://doi.org/10.1029/2022WR032404},
        year = {2022},
    }
    ```

</details>

</br>

- **(HBV Adj.)** Song, Y., Knoben, W. J. M., Clark, M. P., Feng, D., Lawson, K., Sawadekar, K., and Shen, C.: When ancient numerical demons meet physics-informed machine learning: adjoint-based gradients for implicit differentiable modeling, Hydrol. Earth Syst. Sci., 28, 3051–3077, <https://doi.org/10.5194/hess-28-3051-2024>, 2024.

    <details>
    <summary>BibTeX</summary>

    ```bibtex
    @Article{hess-28-3051-2024,
        AUTHOR = {Song, Y. and Knoben, W. J. M. and Clark, M. P. and Feng, D. and Lawson, K. and Sawadekar, K. and Shen, C.},
        TITLE = {When ancient numerical demons meet physics-informed machine learning:
        adjoint-based gradients for implicit differentiable modeling},
        JOURNAL = {Hydrology and Earth System Sciences},
        VOLUME = {28},
        YEAR = {2024},
        NUMBER = {13},
        PAGES = {3051--3077},
        URL = {https://hess.copernicus.org/articles/28/3051/2024/},
        DOI = {10.5194/hess-28-3051-2024}
    }
    ```

</details>

</br>

- **(HBV 1.1p)** Yalan Song, Kamlesh Sawadekar, Jonathan M Frame, et al. Physics-informed, Differentiable Hydrologic  Models for Capturing Unseen Extreme Events  . ESS Open Archive . March 14, 2025. <https://doi.org/10.22541/essoar.172304428.82707157/v2> **[Accepted]**

    <details>
    <summary>BibTeX</summary>

    ```bibtex
    @article{https://doi.org/10.22541/essoar.172304428.82707157/v2,
        author = {Song, Yalan and Sawadekar, Kamlesh and Frame, Jonathan and Pan, Ming and Clark, Martyn and Knoben, Wouter J. M. and Wood W., Andrew and Lawson E., Kathryn and Patel, Trupesh and Shen, Chaopeng},
        title = {Physics-informed, Differentiable Hydrologic  Models for Capturing Unseen Extreme Events},
        journal = {ESS Open Archive},
        volume = {},
        number = {},
        pages = {},
        keywords = {hydrology, differentiable modeling, extremes, physics-informed machine learning, streamflow, streamflow regime},
        doi = {https://doi.org/10.22541/essoar.172304428.82707157/v2},
        year = {2025},
    ```

    </details>

</br>

- **(HBV 2.0)** Song, Y., Bindas, T., Shen, C., Ji, H., Knoben, W. J. M., Lonzarich, L., et al. (2025). High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning. Water Resources Research, 61, e2024WR038928. <https://doi.org/10.1029/2024WR038928>

    <details>
    <summary>BibTeX</summary>

    ```bibtex
    @article{https://doi.org/10.1029/2024WR038928,
        author = {Song, Yalan and Bindas, Tadd and Shen, Chaopeng and Ji, Haoyu and Knoben, Wouter J. M. and Lonzarich, Leo and Clark, Martyn P. and Liu, Jiangtao and van Werkhoven, Katie and Lamont, Sam and Denno, Matthew and Pan, Ming and Yang, Yuan and Rapp, Jeremy and Kumar, Mukesh and Rahmani, Farshid and Thébault, Cyril and Adkins, Richard and Halgren, James and Patel, Trupesh and Patel, Arpita and Sawadekar, Kamlesh Arun and Lawson, Kathryn},
        title = {High-Resolution National-Scale Water Modeling Is Enhanced by Multiscale Differentiable Physics-Informed Machine Learning},
        journal = {Water Resources Research},
        volume = {61},
        number = {4},
        pages = {e2024WR038928},
        keywords = {differentiable modeling, physics-informed machine learning, National Water Model, routing, Muskingum Cunge, multiscale training},
        doi = {https://doi.org/10.1029/2024WR038928},
        year = {2025},
    }
    ```

    </details>

</br>

- **(HBV 2.0 MTS)** Yang, W., Ji, H., Lonzarich, L., Song, Y., Shen, C. (2025). Diffusion-Based Probabilistic Modeling for Hourly Streamflow Prediction and Assimilation. arXiv. <https://arxiv.org/abs/2510.08488> **[Under Review]**

    <details>
    <summary>BibTeX</summary>

    ```bibtex
    @misc{yang2025diffusionbasedprobabilisticmodelinghourly,
          title={Diffusion-Based Probabilistic Modeling for Hourly Streamflow Prediction and Assimilation},
          author={Wencong Yang and Haoyu Ji and Leo Lonzarich and Yalan Song and Chaopeng Shen},
          year={2025},
          eprint={2510.08488},
          archivePrefix={arXiv},
          primaryClass={physics.geo-ph},
          url={https://arxiv.org/abs/2510.08488},
    }
    ```

    </details>

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for details.

---

*Please submit an [issue](https://github.com/mhpi/hydrodl2/issues) to report any questions, concerns, or bugs.*

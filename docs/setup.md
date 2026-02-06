# *HydroDL2* Setup

## 1. System Requirements

HydroDL2 uses PyTorch models and supports both CPU and CUDA (GPU) execution:

- Windows, Linux, or macOS
- Python 3.9--3.13
- NVIDIA GPU(s) supporting CUDA (>12.0 recommended) for GPU-accelerated training

## 2. Install from PyPI

The simplest way to install HydroDL2:

```bash
pip install hydrodl2
```

## 3. Install from Source (Development)

To develop with or contribute to HydroDL2, install from source in editable mode so that changes to the code are immediately reflected without reinstallation.

### Clone the Repository

```bash
git clone https://github.com/mhpi/hydrodl2.git
```

### Create a New Environment and Install

- **UV** (**Recommended** -- UV runs [much faster](https://github.com/astral-sh/uv/blob/main/BENCHMARKS.md) than the alternatives)

  If not already installed, run `pip install uv` or [see here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

  Create a virtual environment:

  ```bash
  uv venv --python 3.12 ./hydrodl2/.venv
  ```

  Activate with `source .venv/bin/activate`, then install HydroDL2:

  ```bash
  uv pip install -e ./hydrodl2
  ```

- **Pip**

  Create a virtual environment in the HydroDL2 directory:

  ```bash
  python3.12 -m venv ./hydrodl2/.venv
  ```

  Activate with `source .venv/bin/activate`. Then install HydroDL2:

  ```bash
  pip install -e ./hydrodl2
  ```

- **Conda**

  Create a base environment for Python versions 3.9--3.13:

  ```bash
  conda env create -n hydrodl2 python=3.x
  ```

  Activate the environment with `conda activate hydrodl2`.

  Install HydroDL2 (editable mode) using pip inside the Conda environment:

  ```bash
  pip install -e ./hydrodl2
  ```

  Note: `conda develop` is deprecated and is not recommended.

  There is a known issue with CUDA failing on new Conda installations. To verify, open a Python instance and check that CUDA is available with PyTorch:

  ```python
  import torch
  print(torch.cuda.is_available())
  ```

  If CUDA is not available, uninstall PyTorch from the environment and reinstall according to your system [specification](https://pytorch.org/get-started/locally/).

### Install Development Dependencies

For linting, testing, and contributing to HydroDL2:

```bash
pip install -e "./hydrodl2[dev]"

# or

uv pip install -e "./hydrodl2[dev]"
```

---

*Please submit an [issue](https://github.com/mhpi/hydrodl2/issues) on GitHub to report any questions, concerns, bugs, etc.*

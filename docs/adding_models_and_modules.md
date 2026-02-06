# Adding Models and Modules to *HydroDL2*

This guide covers how to add new models and modules to the HydroDL2 library so they are automatically discoverable via `load_model()` and `available_models()`.

## Adding a Model

We illustrate this with a hydrology model HBV 1.2 as an example.

1. **Create a model directory** (if it does not already exist) in `src/hydrodl2/models/`, using only lowercase:

   ```text
   src/hydrodl2/models/hbv/       # confirm it exists, or create it
   ```

2. **Create a model file** within your model directory. The filename should match the name exposed to users, converted to lowercase with underscores:

   ```text
   src/hydrodl2/models/hbv/hbv_1_2.py
   ```

3. **Define one model class per file.** The class should be a `torch.nn.Module` subclass. Users will load it with:

   ```python
   Hbv12 = hydrodl2.load_model('hbv_1_2')
   ```

4. **Multiple variants in one file** (discouraged unless differences are small): if you place multiple classes in a single file, users must specify which to load using the `ver_name` parameter. Otherwise, the first class in the file is loaded by default:

   ```python
   Hbv12Beta = hydrodl2.load_model('hbv_1_2', ver_name='Hbv12Beta')
   ```

## Adding a Module

Modules (e.g., data assimilation components) follow the same directory convention under `src/hydrodl2/modules/`:

1. Create a subdirectory for the module category in `src/hydrodl2/modules/` (lowercase).
2. Add a `.py` file containing the module class.

Modules are listed via `available_modules()`.

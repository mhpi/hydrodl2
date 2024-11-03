## Guidlines for Adding Models and Modules to hydroDL2


We illustrate this with a hydrology model HBV v2, or `HBV_v2`.
- In the `models/` directory, create a folder for the model type if it does not already exist, using only lowercase.
    - e.g., we should create `models/hbv/`, or confirm it exists.

- Within `models/<your_model>/`, create a `.py` file for the model taking the name that will be exposed to users, converting to lowercase.
    - e.g., HBV v2 goes to `hbv_v2.py`.

- The model file should only contain one model class. If you want to place multiple variants of your model in the same file, make sure to add a variant specification to your config and indicate
with the `ver_name` flag when loading your model with `load_model()`. Otherwise, the first model listed in your file will be loaded by default.
    - Unless changes between model variants are small, it would be encouraged to use a different file for each.
    
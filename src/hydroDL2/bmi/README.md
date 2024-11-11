A dPLhydro model BMI will live in `./ngen/extern/dpl_model_package` when
operating in NextGen.

Every file for this model that is not in this directory and not standard with
NextGen will be flagged with the identifier `***dPLModel File***`. This way we
can distinguish which files need to be added to a dPLModel module.  



---

### NextGen

When you first git clone https://github.com/NOAA-OWP/ngen.git, run `cd ngen/` and `git submodule update --init --recursive`.

#### To build BMI Python Adapter Test (for Pythonic BMI testing in NextGen)
Then run in `cd ~/ngen/`:
1. `git submodule update --init --recursive -- test/googletest`
2. `cmake -DCMAKE_BUILD_TYPE=Debug -DNGEN_WITH_TESTS:BOOL=ON -B cmake-build-debug -S .` (If UDUNITS2_library is not
found, `conda install udunits2` (confirm that your .conda/[env] dir has `/include`. If not, conda install failed and
should be reattempted.) and run cmake with the following options: 
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DNGEN_WITH_TESTS:BOOL=ON \
      -B cmake-build-debug \
      -S . \
      -DUDUNITS2_LIBRARY=/data/lgl5139/.conda/envs/mulhydrodl/lib/libudunits2.so \
      -DUDUNITS2_INCLUDE_DIR=/data/lgl5139/.conda/envs/mulhydrodl/include)

3. `cmake --build cmake-build-dir --target test_bmi_python -- -j 4` or with options `| less -iR`.
4. run `cmake-build-debug/test/test_bmi_python`


---

### BMI

To test BMI with nextgen:
1. `cd ngen/extern/dpl_model_package`
2. `python run_bmi_unit_test.py`






---

### Steps from `NOAA-OWP/lstm` for running a custom BMI in ngen:

S.D. Peckham
September 12, 2022
November 15, 2022 (updated)

-------------------------------------------
 Steps to run the LSTM Package in NextGen
-------------------------------------------
(1) Download the new LSTM Python package (branch) from:
    https://github.com/NOAA-OWP/lstm/tree/lstm_package

(2) Copy the lstm_package folder into the ngen project tree/repo at:
    ngen/extern/lstm_package

(3) Copy the files in the "ngen_files" subfolder into the corresponding
    locations in the ngen project tree (e.g. into ngen/data/lstm, etc.)

(4) Open a terminal window and set PYTHONPATH (for this session).
    For example:
    (base) % export PYTHONPATH='/Users/peckhams/Dropbox/GitHub/ngen/extern/lstm_py'
    (base) % echo $PYTHONPATH
            /Users/peckhams/Dropbox/GitHub/ngen/extern/lstm_py

    Note:  You may also need to install pytorch, or include its path in PYTHONPATH.
    Note:  Virtual environments in NextGen don't work yet.
    Note:  As a result of setting PYTHONPATH, if you now type:
           (base) % pip list
           you will see "lstm" in the list of packages.
    Note:  You can check that PYTHONPATH is set correctly via:
           (base) % python
           >>> import lstm

(5) In the original version of LSTM, it was assumed that you would be running
    LSTM from its own repo folder, and therefore some file paths are set relative
    to that folder.  For example, in the LSTM model config (YML) files, the
    "train_cfg_file" entry is initially set to:
        ./trained_neuralhydrology_models/hourly_slope_mean_precip/config.yml
    To run LSTM from the top level of the ngen repo folder, this must be changed to:
        ./extern/lstm_py/trained_neuralhydrology_models/hourly_slope_mean_precip/config.yml
    However, LSTM also uses some Python pickle files (created during training) that
    make the same assumption, and you can't edit these.  One easy way to resolve this
    filepath issue --- so that you can run LSTM from the ngen repo folder --- is to
    cd to the ngen repo folder and then create a symbolic link with the command:  
 
    % ln -s ./extern/lstm_py/trained_neuralhydrology_models ./trained_neuralhydrology_models

(7) Run NextGen for the HUC01 catchment "cat-67" with the commands:
    (base) % cd <ngen_repo_tree>
    (base) % ./cmake_build/ngen ./data/lstm/spatial/catchment_data_cat67.geojson "cat-67" ./data/lstm/spatial/nexus_data_nex65.geojson "nex-65" ./data/lstm/rc_files/realization_config_lstm_cat67b.json

    The output should look like:

#### NextGen Output ####
NGen Framework 0.1.0
Building Nexus collection
Building Catchment collection
Catchment topology is dendridic.
Running Models
Running timestep 0
Running timestep 100
Running timestep 200
Running timestep 300
Running timestep 400
Running timestep 500
Running timestep 600
Running timestep 700
Finished 720 timesteps.
ngen(54702,0x10ddc0dc0) malloc: *** error for object 0x7f95afd54740: pointer being freed was not allocated
ngen(54702,0x10ddc0dc0) malloc: *** set a breakpoint in malloc_error_break to debug
zsh: abort      ./cmake_build/ngen ./data/catchment_data_cat67.geojson "cat-67"  "nex-65" 

#### Output files generated in ngen folder ####
cat-67.csv  (34537 bytes)
nex-65_output.csv (24288 bytes)

Note: It appears that the "Pointer" error reported at the end is a NextGen problem.


(8) Run NextGen for 3 test catchments (in one CAMELS basin) with the commands:
    (base) % cd <ngen_repo_tree>
    (base) % ./cmake_build/ngen ./data/lstm/spatial/catchment_data_CAMELS-test.geojson "cat-67" ./data/lstm/spatial/nexus_data_CAMELS-test.geojson "nex-65" ./data/lstm/rc_files/realization_config_lstm_CAMELS-test.json

------------------------------------------------
 Notes about realization config files for LSTM
------------------------------------------------

(1) Set "name" in the formulations block to "bmi_python".

(2) Set "python_type" to "lstm.bmi_lstm.bmi_LSTM",
    which has the form:  "package-name/module-name/class-name".
    Note:  The file: ngen/extern/lstm_py/lstm/__init__.py should be empty.

(3) Set "model_type_name" to "bmi_LSTM", which is the model class name.

(4) Set "init_config" as the complete path to an LSTM model config file.
    For example:
        "./data/lstm/yml_files/HUC01/cat-67.yml"
    or
        "./extern/lstm_py/bmi_config_files/cat-67.yml"
    It can also contain a regular expression to match many files:
        "./data/lstm/yml_files/HUCO01/{{id}}.yml"      

(5) Set "main_output_variable" to
        "land_surface_water__runoff_volume_flux",

(6) In the "variable_names_map" block, notice the line:
        "streamflow_cms: "land_surface_water__runoff_volume_flux",
    You do *not* need to add:
        "water_input": "atmosphere_water__liquid_equivalent_precipitation_rate",

(7) In the "forcing" block, you no longer need to use a forcing file that
    only contains data for the time range of interest (start_time to end_time).
    For example, you can set "path" to:
        "./data/forcing/HUC01-test/cat-67.csv"
    or
        "./data/forcing/CAMELS-test/{{id}}.csv"

(8) In the "time" block, make sure that "start_time" and "end_time" fall into
    the range that is spanned by entries in the CSV forcing file.


--------------------------------------------------------------------
 Steps to run the LSTM Package in NextGen for All HUC01 Catchments
--------------------------------------------------------------------

(1) From the Amazon S3 bucket, download the folder that has LSTM YML config files
    for every HUC01 catchment:  formulations-dev > HUC01 > LSTM  
    (This can be done with Cyberduck, as explained in another doc.)

(2) Copy this LSTM folder to:
        ngen/extern/lstm_py/bmi_config_files/HUC01
    or
        ngen/data/lstm/yml_files/HUC01

(3) Download the files "catchment_data.geojson" and "nexus_data.geojson" from:
    formulations-dev > HUC01
    Copy them into:  ngen/data/lstm/spatial  with the new names:
        catchment_data_HUC01.geojson
        nexus_data_HUC01.geojson

    NOTE:  Each ID (e.g. "cat-27") that appears in:
        ngen/data/lstm/realization_config_lstm.json
    must have a corresponding entry in these two files.
    (So older versions of these geojson files may not work.)

(4) Each YML file in:  ngen/extern/lstm_py/bmi_config_files/HUC01
    has the line:
    train_cfg_file: ./trained_neuralhydrology_models/hourly_slope_mean_precip_temp/config.yml
    Since we'll be running NextGen from the ngen folder, first make sure that the
    trained_neuralhydrology_models folder contains this folder and file, then
    create a symbolic link in the ngen folder with the command:
    % ln -s ./extern/lstm_py/trained_neuralhydrology_models ./trained_neuralhydrology_models

(5) Create a realization config file (ngen/data/lstm/realization_config_lstm_HUC01.json) for
    LSTM that uses the "{{id}}" regular expression to set "init_config", and that uses
    the same forcing file (December 2015) for all catchments.  Here it is:

{
    "global": {
      "formulations":
      [
        {
          "name": "bmi_python",
          "params": {
              "python_type": "lstm.bmi_LSTM",
              "model_type_name": "bmi_LSTM",
              "init_config": "./extern/lstm_py/bmi_config_files/HUC01/{{id}}.yml",
              "main_output_variable": "land_surface_water__runoff_volume_flux",
              "uses_forcing_file": false,
              "variables_names_map" : {
                  "streamflow_cms": "land_surface_water__runoff_volume_flux"},
              "pytorch_model_path": "./data/lstm/sugar_creek_trained.pt",
              "normalization_path": "./data/lstm/input_scaling.csv",
              "initial_state_path": "./data/lstm/initial_states.csv",
              "useGPU": false
        }
      }
      ],
        "forcing": {
            "path": "./data/forcing/HUC01-test/cat-67.csv"
        }
    },
    "time": {
        "start_time": "2015-12-01 00:00:00",
        "end_time": "2015-12-30 23:00:00",
        "output_interval": 3600
    }
}
  
(6) Now type this command, where "" indicates all catchments:

    (base) % cd <ngen-repo-folder>
    (base) % ./cmake_build/ngen ./data/lstm/spatial/catchment_data_HUC01.geojson "" ./data/lstm/spatial/nexus_data_HUC01.geojson "" ./data/lstm/rc_files/realization_config_lstm_HUC01.json

--------------------------------
 Note about model output files
--------------------------------
At this time, NextGen does not support setting a "output directory" for
model output.  So all output files will be written to the same folder
where you ran NextGen from.
See: https://github.com/NOAA-OWP/ngen/issues/374
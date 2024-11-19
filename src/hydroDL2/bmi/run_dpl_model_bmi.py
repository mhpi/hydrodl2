"""
This is a validation script for testing a physics-informed, differentiable ML
model BMI that is NextGen and NOAA-OWP operation-ready.

BMI Docs: https://csdms.colorado.edu/wiki/BMI

Note:
- The current setup is capable of BMI forward on both CAMELS (671 basins) and
    CONUS (3200) MERIT data. For different datasets, `.set_value()` mappings
    must be modeified to the respective forcing + attribute key values.

Author: Leo Lonzarich, 15 Jul. 2024
"""
import sys

package_path = '/data/lgl5139/hydro_multimodel/dPLHydro_multimodel'
sys.path.append(package_path)

import logging
import os
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from conf.config import Config
from core.data import take_sample_test
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from ruamel.yaml import YAML

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import time

# from torch.custom_autograd import BMIBackward
from bmi_dpl_model import dPLModelBMI
from core.data.dataset_loading import get_data_dict

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from bmipy import Bmi

log = logging.getLogger(__name__)


def execute() -> None:
    # Path to BMI config.
    config_path = '/data/lgl5139/ngen/extern/dpl_model_package/conf/model_config.yaml'


    ################## Initialize the BMI ##################
    # Create instance of BMI model.-
    log.info("Creating dPL model BMI instance")
    model = BMIdPLModel(config_filepath=config_path, verbose=True)




    ################## Get test data/forward BMI ##################
    log.info(f"Collecting attribute and forcing data")

    # TODO: Adapt this PMI data loader to ngen, less a function iceberg.
    dataset_dict, _ = get_data_dict(model.config, train=False)

    ## NOTE: This should be fixed in the data loading function for Yalan's CAMELS extraction.
    # Fixing typo in CAMELS dataset: 'geol_porostiy'.
    # (Written into config somewhere inside get_data_dict...)
    var_c_nn = model.config['observations']['var_c_nn']
    if 'geol_porostiy' in var_c_nn:
        model.config['observations']['var_c_nn'][var_c_nn.index('geol_porostiy')] = 'geol_porosity'
        
    # n_timesteps = model.config['end_timestep']
    n_timesteps = dataset_dict['inputs_nn_scaled'].shape[0]
    # n_basins = dataset_dict['c_nn'].shape[0]
    rho = model.config['rho']  # For routing

    # debugging ----- #
    # n_timesteps = 400
    # n_basins = 671
    # --------------- #

    # Store attributes and forcings in BMI if end_timestep > 100 days.
    # NOTE: maybe data gets passed in this step no matter what, but for <100 day
    # periods, it's only seen by dPL model on update call.
    
    # if n_timesteps > 100:
    if model.config['seq_mode'] == True:
        for t in range(n_timesteps - rho):
            # NOTE: for each timestep in this loop, the data assignments below are of
            # arrays of basins. e.g., forcings['key'].shape = (rho + 1, # basins).

            ################## Map forcings + attributes into BMI ##################
            # Set NN forcings...
            for i, var in enumerate(model.config['observations']['var_t_nn']):
                standard_name = model._var_name_map_short_first[var]
                model.set_value(standard_name, dataset_dict['inputs_nn_scaled'][t, :, i], model='nn')
            n_forc = i
            
            # Set NN attributes...
            for i, var in enumerate(model.config['observations']['var_c_nn']):
                standard_name = model._var_name_map_short_first[var]
                model.set_value(standard_name, dataset_dict['inputs_nn_scaled'][t, :, n_forc + i + 1], model='nn') 

            # Set physics model forcings...
            for i, var in enumerate(model.config['observations']['var_t_hydro_model']):
                standard_name = model._var_name_map_short_first[var]
                model.set_value(standard_name, dataset_dict['x_phy'][t, :, i], model='pm') 

            # Set physics model attributes...
            for i, var in enumerate(model.config['observations']['var_c_hydro_model']):
                standard_name = model._var_name_map_short_first[var]
                # NOTE: These don't have a time dimension.
                model.set_value(standard_name, dataset_dict['c_hydro_model'][:, i], model='pm') 
            
        # [CONTROL FUNCTION] Initialize the BMI, forward model in this step if set in config
        log.info(f"INITIALIZING BMI")
        model.initialize()

    else:
        # [CONTROL FUNCTION] Initialize the BMI, forward model in this step if set in config
        log.info(f"INITIALIZING BMI")
        model.initialize()

    ################## Forward model for 1 or multiple timesteps ##################
    log.info(f"BEGIN BMI FORWARD: {n_timesteps} timesteps...")

    # TODO: Add model-internal compiler directives that skip over these steps
    # when the model is run in nextgen. See here: https://github.com/NOAA-OWP/noah-owp-modular/blob/5be0faae07637ffb44235d4783b5420478ff0e9f/src/RunModule.f90#L284

    # Loop through and return streamflow at each timestep.
    for t in range(n_timesteps - rho):
        # NOTE: MHPI models use a warmup period and routing in their forward pass,
        # so we cannot simply pass one timestep to these, but rather warmup or
        # rho + 1 timesteps up to the step we want to predict.

        ################## Map forcings + attributes into BMI ##################
        # Set NN forcings...
        for i, var in enumerate(model.config['observations']['var_t_nn']):
            standard_name = model._var_name_map_short_first[var]
            model.set_value(standard_name, dataset_dict['inputs_nn_scaled'][t:rho + t + 1, :, i], model='nn')
        n_forc = i
        
        # Set NN attributes...
        for i, var in enumerate(model.config['observations']['var_c_nn']):
            standard_name = model._var_name_map_short_first[var]
            model.set_value(standard_name, dataset_dict['inputs_nn_scaled'][t:rho + t + 1, :, n_forc + i + 1], model='nn') 

        # Set physics model forcings...
        for i, var in enumerate(model.config['observations']['var_t_hydro_model']):
            standard_name = model._var_name_map_short_first[var]
            model.set_value(standard_name, dataset_dict['x_phy'][t:rho + t + 1, :, i], model='pm') 

        # Set physics model attributes...
        for i, var in enumerate(model.config['observations']['var_c_hydro_model']):
            standard_name = model._var_name_map_short_first[var]
            # NOTE: These don't have a time dimension.
            model.set_value(standard_name, dataset_dict['c_hydro_model'][:,:, i], model='pm') 

        # [CONTROL FUNCTION] Update the model at all basins for one timestep.
        # Return predictions computed in .initialize() if seq_mode.
        model.update()

        sf = model.streamflow_cms.cpu().detach().numpy()
        print(f"Streamflow at time {t} is {np.average((sf))}")
        print(f"BMI process time: {model.bmi_process_time}")


    # flow_sim = model.get_value('land_surface_water__runoff_volume_flux')
    flow_sim = model.preds[model.config['hydro_models'][0]]['flow_sim']
    

    ################## DA code here ##################
    # Add step here to pass gradients back into BMI.
    # During the BMI update() pass, gradients will be updated and then passed
    # loss = MeanSquaredLoss()

    # optim = "not implemented"  # Some sort of optimizer.

    # # back externally.
    # loss = BMIBackward(MeanSquaredLoss(flow_sim))
    # loss.backward()
    # optim.step()
    # optim.zero_grad()

    # # model.grads
    # # exit()

    ## ------- ##


    ################## Finalize BMI ##################
    # [CONTROL FUNCTION] wrap up BMI run, deallocate mem.
    log.info(f"FINALIZE BMI")
    model.finalize()


def query_bmi_var(model: 'BMI', name: str) -> np.ndarray:
    """
    Args:
        model (Bmi): the Bmi model to query.
        name (str): the name of the variable to query.

    Returns:
        ndarray: numpy array with the value of the variable marshalled through BMI.
    """
    #TODO most (if not all) of this can be cached...unless the grid can change?
    rank = model.get_var_rank(name)
    grid = model.get_var_grid(name)
    shape = np.zeros(rank, dtype=int)
    model.get_grid_shape(grid, shape)
    #TODO call model.get_var_type(name) and determine the correct type of nd array to create
    result = model.get_value(name, np.zeros(shape))
    return result


def initialize_config(cfg: DictConfig) -> Tuple[Config, Dict[str, Any]]:
    """
    Convert config into a dictionary and a Config object for validation.
    """
    try:
        config_dict: Union[Dict[str, Any], Any] = OmegaConf.to_container(
            cfg, resolve=True)
        config = Config(**config_dict)
    except ValidationError as e:
        log.exception("Configuration validation error", exc_info=e)
        raise e
    return config, config_dict


class MeanSquaredLoss(torch.nn.Module):
    """
    Mean squared loss function for BMI Backward.
    """
    def __init__(self) -> None:
        pass

    def forward(self):
        pass
    

if __name__ == '__main__':
    start = time.time()
    execute()

    print(f"completion time: {time.time() - start}")

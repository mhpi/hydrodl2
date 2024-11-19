"""
Physics-informed Differentiable ML Model with BMI Interface
This example demonstrates how to integrate a BMI model that interacts with physical
model data and attributes for hydrological forecasting.

Author: Leo Lonzarich, 2024
"""


from bmi_dpl_model import dPLModelBMI
from data_loader import get_data_dict

CONFIG_PATH = '../bmi_config.yaml'
""" Read in forcings """
forcings = get_data_dict(CONFIG_PATH)

""" Initialize BMI object """
bmi = dPLModelBMI()

""" Control Function """
bmi.initialize(CONFIG_PATH)
    
""" Read data into BMI and forward """
for t in range(bmi.config.timesteps):
    # Map forcings into BMI at each timestep.
    for i, var in enumerate(forcings.keys()):
        standard_name = bmi.get_csdms_name(var)
        bmi.set_value(standard_name, forcings[var][t, :])  #[time,basins]

    """ Control Function - Forward internal PyTorch model """
    bmi.update()

    streamflow = bmi.get_value('land_surface_water__runoff_volume_flux')

""" Control Function - e.g., memory deallocation """
bmi.finalize()


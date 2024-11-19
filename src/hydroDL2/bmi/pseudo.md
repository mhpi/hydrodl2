### BMI Application Pseudo Code

Note: this is for the full-sequence forward BMI, where all data is passed
to BMI after `init`, then enclosed PyTorch model is forwarded in `.initialize()`.

#### BMI
--------------------------------------------
config_path = path to BMI/torch model settings

# 1 - init BMI instance;
bmi_model = BMI(config_path) ((
    # Load configurations, initialize empty input/output var arrays.
))


# 2 - Load forcings & attributes to dictionary;
dataset_dicts = load_datasets()


# 3 - Assign forcings/attributes to internal BMI variables... loop over 
#      timesteps = internal vars have shape [days, basins];
for each timestep in range(# timesteps):
    for var in forcings_nn_dict:  # NN
        bmi_model.set_value(var, forcings_dict[var])

    for var in attributes_nn_dict:  # NN
        bmi_model.set_value(var, attributes_dict[var])

    for var in forcings_hydro_dict:  # physics model
        bmi_model.set_value(var, forcings_hydro_dict[var])


*** All internal and external variables are NumPy arrays ***


# 4 - Initialize the BMI model [BMI control function]
bmi_model.initialize() ((
    # Load a Pytorch model [PMI-BMI linkage].
    torch_model = PMI_model_handler()

    # Reconstruct datset dict with internal values.
    bmi_dataset_dict = {values[forcing/attribute]}

    *** All data for torch_model: numpy -> torch ***
    torch.from_numpy(bmi_dataset_dict)

    # 4.1 - Forward batches of data through pytorch model.
    predictions = []
    batches = batch_basins()
    for i in range(len(batches))
        # Get data for basin sample;
        data_sample = bmi_dataset_dict[basins in batches[i]]

        # Forward pass through pytorch model in PMI.
        value[y*] += torch_model.forward(data_sample)

    # Scale predictions and store internally: torch -> numpy;
    value[y*_np] = predictions['streamflow'].to_numpy()
    value[y*_grad] = value[y*].detach_gradient().to_numpy()
))


y*_combined = torch.zeros(# timesteps, # basins)


# 5 - Loop over timesteps to get outputs
for each timestep in range(# timesteps):

    # 5.1 - [BMI control function]
    bmi_model.update() ((
        # No operation in this setup...
        # Normally does single timestep torch_model forward.
    ))

    # Output predictions and grad from BMI.
    y* = bmi_model.get_value(y*_np)
    y*_grad = bmi_model.get_value(y*_grad)

    *** Convert y* back to torch tensor with gradients ***
    y*_torch = torch.from_numpy(y*)
    y*_torch.grad = y*_grad
    
    y*_combined = torch.cat(y*_torch)

*** Now external y*_combined == predictions['streamflow'] in BMI ***


*** 6 - Get obeservations and convert to tensor; numpy -> torch. ***
y = torch.from_numpy(dataset dict[observations])


# 7. finalize and de-init BMI model [BMI control function];
bmi_model.finalize()
--------------------------------------------













#### Data Assimilation (DA) + BMI
--------------------------------------------
# Config path to BMI/PyTorch model settings
config_path = path_to_BMI_torch_model_settings

# 1 - Initialize BMI instance
bmi_model = BMI()


# 2 - Load forcings and attributes into a dictionary
dataset_dict = load_datasets()


# 3 - Initialize the BMI model [BMI control function]
bmi_model.initialize(config_path) ((
    # Load PyTorch model
    torch_model = PMI_model_handler()

    # Load configurations and initialize empty input/output variable arrays.
))

# 4 - [Start of DA part] Loop through basin batches
k_list = [] 
batches = batch_basins() # Either single basin and parallelized, or batches of basins

for basin in range(# basins):

    # 4.1 - Sliding window loop
    for t in range(# timesteps - window_size + 1):
        
        # 4.2 - Initialize correction factor for current window, adjust precipitation.
        # NOTE: confirm if adjusted_prcp also needs to be a tensor for its interaction
        # with k. Probably true.
        k = torch.tensor([1.0])
        adjusted_prcp = k * torch.from_numpy(dataset_dict[prcp][basin:basin + 1, t:t + window_size])
        dataset_dict[prcp][basin:basin + 1, t:t + window_size] = adjusted_P_input

        *** adjusted_prcp now has grad inherited from k ***


        # Initialize optimizer for k
        optimizer = torch.optim.Adam([k])
    

        # 4.3 loop over epochs
        for epoch in range(# epochs):

            optimizer.zero_grad()
            y*_combined = torch.zeros(# timesteps, # basins)


            # 4.4 - Update internal forcing variables in the BMI model.
            for each timestep in range(window_size):
                
                *** Convert torch -> numpy, save grad ***
                bmi_model.set_value(prcp, forcings_dict[prcp])
                bmi_model.set_value(prcp_grad, predictions['streamflow'].detach_gradient().to_numpy())

                for var in forcings_nn_dict:  # NN
                    bmi_model.set_value(var, forcings_dict[var!=prcp]) ((
                        *** Convert numpy -> torch, reattach grads.
                        values[prcp] = torch.from_numpy()
                        values[prcp].grad = values[prcp_grad]
                    ))

                for var in attributes_nn_dict:  # NN
                    bmi_model.set_value(var, attributes_dict[var])

                for var in forcings_hydro_dict:  # physics model
                    bmi_model.set_value(var, forcings_hydro_dict[var])


                # 4.5 - Forward model [BMI control function].
                bmi_model.update() ((
                    # Reconstruct datset dict with internal values.
                    bmi_dataset_dict = {values[forcing/attribute]}

                    # Forward pass through pytorch model in PMI to get y*.
                    value[y*] += torch_model.forward(bmi_dataset_dict)

                    *** Convert torch -> numpy ***
                    value[y*_np] = y*.to_numpy()
                    value[y*_grad] = y*.detach_gradient().to_numpy()
                ))


                # Output predictions and grad from BMI.
                y* = bmi_model.get_value(y*)
                y*_grad = bmi_model.get_value(y*_grad)

                *** Convert y* back to torch tensor with gradients ***
                y*_torch = torch.from_numpy(y*)
                y*_torch.grad = y*_grad
                
                y*_combined = torch.cat(y*_torch)


            y = torch.tensor(dataset_dict[streamflow_obs][basin:basin + 1, i:i + window_size, :])

            loss = lossFun(y*_combined, y)

            *** #5 - Backprop with custom backward ***
            loss.backward()
            optimizer.step()

        k_list.append(k.item())
        torch.cuda.empty_cache()

# 6 - Finalize and deinitialize the BMI model
bmi_model.finalize()
--------------------------------------------

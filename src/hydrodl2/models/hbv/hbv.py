from typing import Any, Optional, Union

import torch

from hydrodl2.core.calc import change_param_range, uh_conv, uh_gamma


class Hbv(torch.nn.Module):
    """HBV 1.0 ~.

    Multi-component, differentiable PyTorch HBV model with option to run without
    internal state warmup.

    Authors
    -------
    -   Farshid Rahmani & Yalan Song, Leo Lonzarich
    -   (Original NumPy HBV ver.) Beck et al., 2020 (http://www.gloh2o.org/hbv/).
    -   (HBV-light Version 2) Seibert, 2005
        (https://www.geo.uzh.ch/dam/jcr:c8afa73c-ac90-478e-a8c7-929eed7b1b62/HBV_manual_2005.pdf).
    
    Publication
    -----------
    -   Dapeng Feng, Jiangtao Liu, Kathryn Lawson, Chaopeng Shen. "Differentiable,
        learnable, regionalized process-based models with multiphysical outputs
        can approach state-of-the-art hydrologic prediction accuracy." Water
        Resources Research (2020), 58, e2022WR032404.
        https://doi.org/10.1029/2022WR032404.

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        Device to run the model on.
    """
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = 'HBV 1.0'
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.dynamic_params = []
        self.dy_drop = 0.0
        self.variables = ['prcp', 'tmean', 'pet']
        self.routing = True
        self.comprout = False
        self.nearzero = 1e-5
        self.nmul = 1
        self.device = device
        self.parameter_bounds = {
            'parBETA': [1.0, 6.0],
            'parFC': [50, 1000],
            'parK0': [0.05, 0.9],
            'parK1': [0.01, 0.5],
            'parK2': [0.001, 0.2],
            'parLP': [0.2, 1],
            'parPERC': [0, 10],
            'parUZL': [0, 100],
            'parTT': [-2.5, 2.5],
            'parCFMAX': [0.5, 10],
            'parCFR': [0, 0.1],
            'parCWH': [0, 0.2],
        }
        self.routing_parameter_bounds = {
            'rout_a': [0, 2.9],
            'rout_b': [0, 6.5],
        }

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            # Overwrite defaults with config values.
            self.warm_up = config.get('warm_up', self.warm_up)
            self.warm_up_states = config.get('warm_up_states', self.warm_up_states)
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.dynamic_params = config['dynamic_params'].get(self.__class__.__name__, self.dynamic_params)
            self.variables = config.get('variables', self.variables)
            self.routing = config.get('routing', self.routing)
            self.comprout = config.get('comprout', self.comprout)
            self.nearzero = config.get('nearzero', self.nearzero)
            self.nmul = config.get('nmul', self.nmul)
            if 'parBETAET' in self.dynamic_params:
                self.parameter_bounds['parBETAET'] = [0.3, 5]
        self.set_parameters()

    def set_parameters(self) -> None:
        """Get physical parameters."""
        self.phy_param_names = self.parameter_bounds.keys()
        if self.routing:
            self.routing_param_names = self.routing_parameter_bounds.keys()
        else:
            self.routing_param_names = []

        self.learnable_param_count = len(self.phy_param_names) * self.nmul \
            + len(self.routing_param_names)

    def unpack_parameters(
        self,
        parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract physical model and routing parameters from NN output.
        
        Parameters
        ----------
        parameters
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of physical and routing parameters.
        """
        phy_param_count = len(self.parameter_bounds)
        
        # Physical parameters
        phy_params = torch.sigmoid(
            parameters[:, :, :phy_param_count * self.nmul]).view(
                parameters.shape[0],
                parameters.shape[1],
                phy_param_count,
                self.nmul,
            )
        # Routing parameters
        routing_params = None
        if self.routing:
            routing_params = torch.sigmoid(
                parameters[-1, :, phy_param_count * self.nmul:],
            )
        return phy_params, routing_params

    def descale_phy_parameters(
        self,
        phy_params: torch.Tensor,
        dy_list:list,
    ) -> torch.Tensor:
        """Descale physical parameters.
        
        Parameters
        ----------
        phy_params
            Normalized physical parameters.
        dy_list
            List of dynamic parameters.
        
        Returns
        -------
        dict
            Dictionary of descaled physical parameters.
        """
        n_steps = phy_params.size(0)
        n_grid = phy_params.size(1)

        # TODO: Fix; if dynamic parameters are not entered in config as they are
        # in HBV params list, then descaling misamtch will occur. Confirm this
        # does not happen.
        param_dict = {}
        pmat = torch.ones([1, n_grid, 1]) * self.dy_drop
        for i, name in enumerate(self.parameter_bounds.keys()):
            staPar = phy_params[-1, :, i,:].unsqueeze(0).repeat([n_steps, 1, 1])
            if name in dy_list:
                dynPar = phy_params[:, :, i,:]
                drmask = torch.bernoulli(pmat).detach_().cuda()
                comPar = dynPar * (1 - drmask) + staPar * drmask
                param_dict[name] = change_param_range(
                    param=comPar,
                    bounds=self.parameter_bounds[name],
                )
            else:
                param_dict[name] = change_param_range(
                    param=staPar,
                    bounds=self.parameter_bounds[name],
                )
        return param_dict

    def descale_rout_parameters(
        self,
        routing_params: torch.Tensor,
    ) -> torch.Tensor:
        """Descale routing parameters.
        
        Parameters
        ----------
        routing_params
            Normalized routing parameters.

        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(self.routing_parameter_bounds.keys()):
            param = routing_params[:, i]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.routing_parameter_bounds[name],
            )
        return parameter_dict

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: torch.Tensor,
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """Forward pass for HBV.
        
        Parameters
        ----------
        x_dict
            Dictionary of input forcing data.
        parameters
            Unprocessed, learned parameters from a neural network.
        
        Returns
        -------
        Union[tuple, dict]
            Tuple or dictionary of model outputs.
        """
        # Unpack input data.
        x = x_dict['x_phy']
        self.muwts = x_dict.get('muwts', None)

        # Unpack parameters.
        phy_params, routing_params = self.unpack_parameters(parameters)
        
        if self.routing:
            self.routing_param_dict = self.descale_rout_parameters(routing_params)

        # Initialization
        if self.warm_up_states:
            warm_up = self.warm_up
        else:
            # No state warm up - run the full model for warm_up days.
            self.pred_cutoff = self.warm_up
            warm_up = 0
        
        n_grid = x.size(1)

        # Initialize model states.
        SNOWPACK = torch.zeros([n_grid, self.nmul],
                                dtype=torch.float32,
                                device=self.device) + 0.001
        MELTWATER = torch.zeros([n_grid, self.nmul],
                                dtype=torch.float32,
                                device=self.device) + 0.001
        SM = torch.zeros([n_grid, self.nmul],
                         dtype=torch.float32,
                         device=self.device) + 0.001
        SUZ = torch.zeros([n_grid, self.nmul],
                          dtype=torch.float32,
                          device=self.device) + 0.001
        SLZ = torch.zeros([n_grid, self.nmul],
                          dtype=torch.float32,
                          device=self.device) + 0.001

        # Warm-up model states - run the model only on warm_up days first.
        if warm_up > 0:
            with torch.no_grad():
                phy_param_warmup_dict = self.descale_phy_parameters(
                    phy_params[:warm_up,:,:],
                    dy_list=[],
                )
                # Save current model settings.
                initialize = self.initialize
                routing  = self.routing

                # Set model settings for warm-up.
                self.initialize =  True
                self.routing = False

                SNOWPACK, MELTWATER, SM, SUZ, SLZ = self.PBM(
                    x[:warm_up, :, :],
                    [SNOWPACK, MELTWATER, SM, SUZ, SLZ],
                    phy_param_warmup_dict,
                )

                # Restore model settings.
                self.initialize = initialize
                self.routing = routing
        
        phy_params_dict = self.descale_phy_parameters(
            phy_params[warm_up:,:,:],
            dy_list=self.dynamic_params,
        )
        
        # Run the model for the remainder of simulation period.
        return self.PBM(
                    x[warm_up:, :, :],
                    [SNOWPACK, MELTWATER, SM, SUZ, SLZ],
                    phy_params_dict,
                )

    def PBM(
        self,
        forcing: torch.Tensor,
        states: tuple,
        full_param_dict: dict,
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """Run the HBV model forward.
        
        Parameters
        ----------
        forcing
            Input forcing data.
        states
            Initial model states.
        full_param_dict
            Dictionary of model parameters.
        
        Returns
        -------
        Union[tuple, dict]
            Tuple or dictionary of model outputs.
        """
        SNOWPACK, MELTWATER, SM, SUZ, SLZ = states

        # Forcings
        P = forcing[:, :, self.variables.index('prcp')]  # Precipitation
        T = forcing[:, :, self.variables.index('tmean')]  # Mean air temp
        PET = forcing[:, :, self.variables.index('pet')] # Potential ET

        # Expand dims to accomodate for nmul models.
        Pm = P.unsqueeze(2).repeat(1, 1, self.nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, self.nmul)
        PETm = PET.unsqueeze(-1).repeat(1, 1, self.nmul)

        n_steps, n_grid = P.size()

        # Apply correction factor to precipitation
        # P = parPCORR.repeat(n_steps, 1) * P

        # Initialize time series of model variables in shape [time, basins, nmul].
        Qsimmu = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.001
        Q0_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        Q1_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        Q2_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001

        # AET = PET_coef * PET
        AET = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        recharge_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        excs_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        evapfactor_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        tosoil_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        PERC_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        SWE_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)

        snowpack = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        meltwater = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        soil_moisture = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        upper_zone = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        lower_zone = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)

        param_dict ={}
        for t in range(n_steps):
            # Get dynamic parameter values per timestep.
            for key in full_param_dict.keys():
                param_dict[key] = full_param_dict[key][t, :, :]

            # Separate precipitation into liquid and solid components.
            PRECIP = Pm[t, :, :]
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= param_dict['parTT']).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < param_dict['parTT']).type(torch.float32))

            # Snow -------------------------------
            SNOWPACK = SNOWPACK + SNOW
            melt = param_dict['parCFMAX'] * (Tm[t, :, :] - param_dict['parTT'])
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = param_dict['parCFR'] * param_dict['parCFMAX'] * (
                param_dict['parTT'] - Tm[t, :, :]
                )
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (param_dict['parCWH'] * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation -------------------------------
            soil_wetness = (SM / param_dict['parFC']) ** param_dict['parBETA']
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            SM = SM + RAIN + tosoil - recharge

            excess = SM - param_dict['parFC']
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # parBETAET only has effect when it is a dynamic parameter.
            evapfactor = (SM / (param_dict['parLP'] * param_dict['parFC']))
            if 'parBETAET' in param_dict:
                evapfactor = evapfactor ** param_dict['parBETAET']
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PETm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=self.nearzero)

            # Groundwater boxes -------------------------------
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, param_dict['parPERC'])
            SUZ = SUZ - PERC
            Q0 = param_dict['parK0'] * torch.clamp(SUZ - param_dict['parUZL'], min=0.0)
            SUZ = SUZ - Q0
            Q1 = param_dict['parK1'] * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = param_dict['parK2'] * SLZ
            SLZ = SLZ - Q2

            Qsimmu[t, :, :] = Q0 + Q1 + Q2
            Q0_sim[t, :, :] = Q0
            Q1_sim[t, :, :] = Q1
            Q2_sim[t, :, :] = Q2
            AET[t, :, :] = ETact
            SWE_sim[t, :, :] = SNOWPACK

            recharge_sim[t, :, :] = recharge
            excs_sim[t, :, :] = excess
            evapfactor_sim[t, :, :] = evapfactor
            tosoil_sim[t, :, :] = tosoil
            PERC_sim[t, :, :] = PERC

            # Record model states
            snowpack[t, :, :] = SNOWPACK
            meltwater[t, :, :] = MELTWATER
            soil_moisture[t, :, :] = SM
            upper_zone[t, :, :] = SUZ
            lower_zone[t, :, :] = SLZ

        # Get the overall average or weighted average using learned weights.
        if self.muwts is None:
            Qsimavg = Qsimmu.mean(-1)
        else:
            Qsimavg = (Qsimmu * self.muwts).sum(-1)

        # Run routing
        if self.routing:
            # Routing for all components or just the average.
            if self.comprout:
                # All components; reshape to [time, gages * num models]
                Qsim = Qsimmu.view(n_steps, n_grid * self.nmul)
            else:
                # Average, then do routing.
                Qsim = Qsimavg

            UH = uh_gamma(
                self.routing_param_dict['rout_a'].repeat(n_steps, 1).unsqueeze(-1),
                self.routing_param_dict['rout_b'].repeat(n_steps, 1).unsqueeze(-1),
                lenF=15,
            )
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])  # [gages,vars,time]
            UH = UH.permute([1, 2, 0])  # [gages,vars,time]
            Qsrout = uh_conv(rf, UH).permute([2, 0, 1])

            # Routing individually for Q0, Q1, and Q2, all w/ dims [gages,vars,time].
            rf_Q0 = Q0_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q0_rout = uh_conv(rf_Q0, UH).permute([2, 0, 1])
            rf_Q1 = Q1_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q1_rout = uh_conv(rf_Q1, UH).permute([2, 0, 1])
            rf_Q2 = Q2_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q2_rout = uh_conv(rf_Q2, UH).permute([2, 0, 1])

            if self.comprout:
                # Qs is now shape [time, [gages*num models], vars]
                Qstemp = Qsrout.view(n_steps, n_grid, self.nmul)
                if self.muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp *self.muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else:
            # No routing, only output the average of all model sims.
            Qs = torch.unsqueeze(Qsimavg, -1)
            Q0_rout = Q1_rout = Q2_rout = None

        if self.initialize:
            # If initialize is True, only return warmed-up storages.
            return SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # Baseflow index (BFI) calculation
            BFI_sim = 100 * (torch.sum(Q2_rout, dim=0) / (
                torch.sum(Qs, dim=0) + self.nearzero))[:,0]
            
            # Return all sim results.
            out_dict = {
                'streamflow': Qs,  # Routed Streamflow
                'srflow': Q0_rout,  # Routed surface runoff
                'ssflow': Q1_rout,  # Routed subsurface flow
                'gwflow': Q2_rout,  # Routed groundwater flow
                'AET_hydro': AET.mean(-1, keepdim=True),  # Actual ET
                'PET_hydro': PETm.mean(-1, keepdim=True),  # Potential ET
                'SWE': SWE_sim.mean(-1, keepdim=True),  # Snow water equivalent
                'streamflow_no_rout': Qsim.unsqueeze(dim=2),  # Streamflow
                'srflow_no_rout': Q0_sim.mean(-1, keepdim=True),  # Surface runoff
                'ssflow_no_rout': Q1_sim.mean(-1, keepdim=True),  # Subsurface flow
                'gwflow_no_rout': Q2_sim.mean(-1, keepdim=True),  # Groundwater flow
                'recharge': recharge_sim.mean(-1, keepdim=True),  # Recharge
                'excs': excs_sim.mean(-1, keepdim=True),  # Excess stored water
                'evapfactor': evapfactor_sim.mean(-1, keepdim=True),  # Evaporation factor
                'tosoil': tosoil_sim.mean(-1, keepdim=True),  # Infiltration
                'percolation': PERC_sim.mean(-1, keepdim=True),  # Percolation
                'BFI': BFI_sim,  # Baseflow index
            }
            # state_dict = {
            #     'snowpack': snowpack.mean(-1, keepdim=True),
            #     'meltwater': meltwater.mean(-1, keepdim=True),
            #     'soil_moisture': soil_moisture.mean(-1, keepdim=True),
            #     'upper_zone': upper_zone.mean(-1, keepdim=True),
            #     'lower_zone': lower_zone.mean(-1, keepdim=True),
            # }

            # For surrogate model training, return full parameter dictionary.
            full_param_dict['rout_a'] = self.routing_param_dict['rout_a'].repeat(n_steps, 1).unsqueeze(-1)
            full_param_dict['rout_b'] = self.routing_param_dict['rout_b'].repeat(n_steps, 1).unsqueeze(-1)

            if not self.warm_up_states:
                for key in out_dict.keys():
                    if key != 'BFI':
                        out_dict[key] = out_dict[key][self.pred_cutoff:, :, :]
            return out_dict #, state_dict, full_param_dict
